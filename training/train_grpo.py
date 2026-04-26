"""Multi-Agent ATC GRPO Training with Unsloth.

Architecture:
  - Single LLM (Qwen2.5-7B-Instruct, 4-bit QLoRA) plays 4 roles via system prompts
  - GRPO: group-relative advantage  A_i = (r_i - mean(group)) / (std(group) + eps)
  - Four independent reward functions (AMAN, DMAN, GENERATOR, SUPERVISOR)
  - Potential-based reward shaping per role (policy-gradient safe, Ng et al. 1999)
  - Adaptive curriculum: ChallengeGenerator escalates difficulty as agents improve
  - Per-role reward curves saved to reward_curves.json for demo

Training loop:
  Episode -> Generator mutates task -> AMAN bids -> DMAN bids ->
  Negotiate (if conflicts) -> Grade -> Per-agent GRPO update

Colab T4 resource profile:
  Model:        Qwen2.5-7B-Instruct, 4-bit QLoRA
  LoRA rank:    16 (q_proj, v_proj, k_proj, o_proj)
  Batch size:   2, gradient accumulation 4 -> effective batch 8
  Generations:  4 per prompt (GRPO group size — minimum for stable advantage estimate)
  Max tokens:   512 per completion
  Training:     ~200 episodes ≈ 800 samples ≈ 2 hr on T4

For JSON parse reliability on a cold base model, run SFT first (``training/build_sft_dataset.py`` +
``training/train_sft.py``) then continue from the saved adapter.

Usage:
  python training/train_grpo.py [--episodes 200] [--model Qwen/Qwen2.5-7B-Instruct]

Colab one-liner:
  !python training/train_grpo.py --episodes 100 --output_dir /content/atc-multiagent
"""

from __future__ import annotations

import argparse
import inspect
import json
import logging
import os
import re
import sys
import time
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _require_training_deps():
    if sys.version_info >= (3, 14):
        print("[ERROR] Python 3.14 not supported. Use 3.11 or 3.12.")
        sys.exit(1)
    try:
        import torch
    except ImportError as e:
        print(f"[ERROR] Training deps missing: {e}")
        print("Install: pip install torch")
        sys.exit(1)
    try:
        # Unsloth should be imported before TRL/Transformers/PEFT.
        import unsloth  # noqa: F401
        from unsloth import FastLanguageModel
    except Exception as e:
        print(f"[ERROR] unsloth import failed: {e}")
        print("Install: pip install unsloth unsloth-zoo")
        sys.exit(1)
    try:
        from trl import GRPOConfig, GRPOTrainer
    except ImportError as e:
        print(f"[ERROR] Training deps missing: {e}")
        print("Install: pip install trl transformers")
        sys.exit(1)
    return torch, FastLanguageModel, GRPOConfig, GRPOTrainer


from training.dataset import (
    build_episode_dataset,
    parse_aman_action,
    parse_dman_action,
    parse_generator_action,
)
from training.reward_functions import (
    aman_reward_fn,
    dman_reward_fn,
    generator_reward_fn,
    supervisor_reward_fn,
    adapt_reward_fn,
)
from multi_agent.environment import MultiAgentATCEnvironment
from multi_agent.generator import ChallengeGenerator
from multi_agent.models import AgentRole, SupervisorProfileName
from multi_agent.supervisor import SupervisorAgent
from tasks import task_catalog, ordered_tasks


# ── Hyperparameters ───────────────────────────────────────────────────────────

DEFAULT_MODEL  = "Qwen/Qwen2.5-7B-Instruct"
DEFAULT_OUTPUT = "./outputs/atc-multiagent"
LORA_RANK      = 16
LORA_ALPHA     = 32
LORA_TARGETS   = ["q_proj", "v_proj", "k_proj", "o_proj"]
MAX_SEQ_LEN    = 4096
MAX_NEW_TOKENS = 384
TEMPERATURE    = 0.7
# Role-aware generation budgets. GRPO supports one global max token cap, so we
# use the max here and enforce per-role compactness in rewards/logging.
ROLE_MAX_NEW_TOKENS = {
    AgentRole.AMAN.value: 384,
    AgentRole.DMAN.value: 384,
    AgentRole.GENERATOR.value: 224,
    AgentRole.SUPERVISOR.value: 160,
    AgentRole.ADAPT.value: 384,
}
# 8 generations per prompt.
# Research basis (GRPO dynamics, arXiv 2503.06639): the adaptive advantage weight
# w_i = (1-p)/√[p(1-p)] relies on an accurate estimate of p (per-prompt success
# rate).  With N=4 the estimate variance is high enough to produce misleading
# advantage signs on low-success prompts.  N=8 halves the estimation variance
# while keeping VRAM cost manageable on T4 (1.5B model, LoRA rank 16).
# DAPO / NGRPO finding: groups where pass@N ∈ {0, N} have std≈0 → zero GRPO
# gradient.  Larger N reduces the probability of all-correct or all-wrong groups
# on intermediate-difficulty tasks.
N_GENERATIONS  = 8
BATCH_SIZE     = 4
GRAD_ACCUM     = 2           # effective batch = 8
LR             = 5e-5
# In trl==0.16.0 + unsloth==2026.4.7 with PEFT, non-zero KL can fail when
# ref_per_token_logps is absent in the fast path (ref=None crash).
KL_COEFF       = 0.0
WARMUP_RATIO   = 0.05
LOGGING_STEPS  = 1
SAVE_STEPS     = 50
SAVE_TOTAL_LIMIT = 3         # keep only 3 checkpoints on disk
EVAL_EPISODES  = 3
STAGE_EPOCHS = {
    "stage_a": 0.45,  # AMAN + DMAN only
    "stage_b": 0.30,  # add sampled generator/supervisor
    "stage_c": 0.25,  # full 4-role mix
}


def _configure_runtime_warnings() -> None:
    """Hide repetitive upstream warnings that don't affect correctness."""
    warnings.filterwarnings(
        "ignore",
        message=r"Both `max_new_tokens` \(=.*\) and `max_length`\(=.*\) seem to have been set\..*",
    )
    warnings.filterwarnings(
        "ignore",
        message=r"Passing `generation_config` together with generation-related arguments=.* is deprecated.*",
    )
    warnings.filterwarnings(
        "ignore",
        message=r"The attention mask API under `transformers\.modeling_attn_mask_utils`.*deprecated.*",
        category=FutureWarning,
    )
    # Some transformers builds emit this through logging, not warnings.
    class _SuppressMaxLenWarning(logging.Filter):
        def filter(self, record: logging.LogRecord) -> bool:
            msg = record.getMessage()
            return "Both `max_new_tokens`" not in msg

    for name in (
        "transformers.generation.utils",
        "transformers.generation.configuration_utils",
    ):
        logging.getLogger(name).addFilter(_SuppressMaxLenWarning())


def _auto_tune_for_gpu(torch_module) -> Dict[str, int]:
    """Return tuned batch/accum/token settings based on detected VRAM."""
    tuned = {
        "batch_size": BATCH_SIZE,
        "grad_accum": GRAD_ACCUM,
        "max_new_tokens": MAX_NEW_TOKENS,
    }
    if not torch_module.cuda.is_available():
        return tuned
    try:
        vram_gb = torch_module.cuda.get_device_properties(0).total_memory / (1024 ** 3)
    except Exception:
        return tuned

    # 80GB-class GPUs: increase throughput while keeping rollout quality.
    if vram_gb >= 70:
        tuned["batch_size"] = max(tuned["batch_size"], 8)
        tuned["grad_accum"] = 1
        tuned["max_new_tokens"] = min(tuned["max_new_tokens"], 384)
    elif vram_gb >= 40:
        tuned["batch_size"] = max(tuned["batch_size"], 6)
        tuned["grad_accum"] = min(tuned["grad_accum"], 2)
        tuned["max_new_tokens"] = min(tuned["max_new_tokens"], 384)
    return tuned


def _prefer_local_model_path(model_name: str) -> str:
    """Use local HF cache path when available to avoid network flakiness."""
    if os.path.isdir(model_name):
        return model_name
    try:
        from huggingface_hub import snapshot_download

        local_path = snapshot_download(repo_id=model_name, local_files_only=True)
        print(f"[INFO] Using local model snapshot cache: {local_path}")
        return local_path
    except Exception:
        return model_name


def _is_transient_network_error(exc: Exception) -> bool:
    msg = str(exc).lower()
    needles = (
        "temporary failure in name resolution",
        "name resolution",
        "connecterror",
        "connection error",
        "failed to establish a new connection",
    )
    return any(n in msg for n in needles)


def _load_model_with_fallback(
    FastLanguageModel,
    model_source: str,
    *,
    max_seq_length: int,
):
    """Load model/tokenizer, retrying in offline mode on DNS/network failures."""
    kwargs = {
        "model_name": model_source,
        "max_seq_length": max_seq_length,
        "load_in_4bit": True,
        "dtype": None,
    }
    try:
        return FastLanguageModel.from_pretrained(**kwargs)
    except Exception as exc:
        if not _is_transient_network_error(exc):
            raise
        print("[WARN] Network/DNS issue while loading tokenizer/model. Retrying from local cache...")
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
        os.environ["HF_HUB_OFFLINE"] = "1"
        kwargs["local_files_only"] = True
        return FastLanguageModel.from_pretrained(**kwargs)


# ── Role-dispatch table ───────────────────────────────────────────────────────

REWARD_FN_DISPATCH = {
    AgentRole.AMAN.value:       aman_reward_fn,
    AgentRole.DMAN.value:       dman_reward_fn,
    AgentRole.GENERATOR.value:  generator_reward_fn,
    AgentRole.SUPERVISOR.value: supervisor_reward_fn,
    AgentRole.ADAPT.value:      adapt_reward_fn,
}


def _reward_failure_mode() -> str:
    mode = os.getenv("REWARD_FAILURE_MODE", "strict").strip().lower()
    return mode if mode in {"strict", "penalize"} else "strict"


def _config_supports(param: str, config_cls) -> bool:
    try:
        return param in inspect.signature(config_cls.__init__).parameters
    except Exception:
        return False


def _trainer_supports(param: str, trainer_cls) -> bool:
    try:
        return param in inspect.signature(trainer_cls.__init__).parameters
    except Exception:
        return False


def _maybe_patch_trainer_sampler(trainer) -> None:
    """Handle TRL/Transformers sampler signature drift across versions."""
    try:
        sampler = getattr(type(trainer), "_get_train_sampler", None)
        if sampler is None:
            return
        # Old TRL versions expose _get_train_sampler(self) while newer
        # Transformers call sampler_fn(dataset). Patch only that old form.
        if len(inspect.signature(sampler).parameters) == 1:
            from types import MethodType

            original = trainer._get_train_sampler

            def _compat_get_train_sampler(self, train_dataset=None):
                return original()

            trainer._get_train_sampler = MethodType(_compat_get_train_sampler, trainer)
            print("[WARN] Applied sampler compatibility shim for this TRL/Transformers pair.")
    except Exception as exc:
        print(f"[WARN] Could not apply sampler compatibility shim: {exc}")


def _maybe_patch_unsloth_grad_accum(trainer) -> None:
    """Provide missing attribute expected by some Unsloth GRPO trainer builds."""
    if hasattr(trainer, "current_gradient_accumulation_steps"):
        return
    steps = 1
    try:
        args = getattr(trainer, "args", None)
        if args is not None:
            steps = int(getattr(args, "gradient_accumulation_steps", 1) or 1)
    except Exception:
        steps = 1
    trainer.current_gradient_accumulation_steps = steps
    print(
        "[WARN] Applied Unsloth GRPO compatibility shim: "
        f"current_gradient_accumulation_steps={steps}"
    )


def _maybe_patch_unsloth_loss_type(trainer) -> None:
    """Ensure loss_type exists for Unsloth/TRL compatibility.

    Some compiled trainer paths branch on hasattr(self.args, "loss_type").
    If missing, they may take an older unpack path that mismatches newer
    return signatures and crashes with "too many values to unpack".
    """
    try:
        args = getattr(trainer, "args", None)
        if args is None:
            return
        if getattr(args, "loss_type", None) is None:
            setattr(args, "loss_type", "grpo")
            print("[WARN] Applied Unsloth GRPO compatibility shim: loss_type='grpo'")
    except Exception as exc:
        print(f"[WARN] Could not apply loss_type compatibility shim: {exc}")


def _maybe_patch_unsloth_runtime_attrs(trainer) -> None:
    """Backfill runtime attrs expected by some Unsloth compiled trainer builds."""
    try:
        args = getattr(trainer, "args", None)

        if not hasattr(trainer, "importance_sampling_level"):
            level = "token"
            if args is not None:
                level = getattr(args, "importance_sampling_level", level) or level
            setattr(trainer, "importance_sampling_level", level)
            print(
                "[WARN] Applied Unsloth GRPO compatibility shim: "
                f"importance_sampling_level={level!r}"
            )

        if not hasattr(trainer, "epsilon_low"):
            epsilon_low = 0.2
            if args is not None:
                epsilon_low = float(getattr(args, "epsilon", epsilon_low) or epsilon_low)
            setattr(trainer, "epsilon_low", epsilon_low)
            print(
                "[WARN] Applied Unsloth GRPO compatibility shim: "
                f"epsilon_low={epsilon_low}"
            )

        if not hasattr(trainer, "epsilon_high"):
            epsilon_high = getattr(trainer, "epsilon_low", 0.2)
            if args is not None:
                epsilon_high = float(
                    getattr(args, "epsilon_high", epsilon_high) or epsilon_high
                )
            setattr(trainer, "epsilon_high", epsilon_high)
            print(
                "[WARN] Applied Unsloth GRPO compatibility shim: "
                f"epsilon_high={epsilon_high}"
            )

        if not hasattr(trainer, "vllm_importance_sampling_cap"):
            cap = 2.0
            if args is not None:
                cap = float(
                    getattr(args, "vllm_importance_sampling_cap", cap) or cap
                )
            setattr(trainer, "vllm_importance_sampling_cap", cap)
            print(
                "[WARN] Applied Unsloth GRPO compatibility shim: "
                f"vllm_importance_sampling_cap={cap}"
            )

        if not hasattr(trainer, "loss_type"):
            loss_type = "grpo"
            if args is not None:
                loss_type = getattr(args, "loss_type", loss_type) or loss_type
            setattr(trainer, "loss_type", loss_type)
            print(
                "[WARN] Applied Unsloth GRPO compatibility shim: "
                f"loss_type={loss_type!r}"
            )
    except Exception as exc:
        print(f"[WARN] Could not apply runtime attr compatibility shim: {exc}")


def _maybe_patch_unsloth_args_attrs(trainer) -> None:
    """Backfill args fields expected by some Unsloth compiled trainer paths."""
    try:
        args = getattr(trainer, "args", None)
        if args is None:
            return

        if not hasattr(args, "delta"):
            setattr(args, "delta", None)
            print("[WARN] Applied Unsloth GRPO compatibility shim: args.delta=None")

        if not hasattr(args, "temperature"):
            setattr(args, "temperature", TEMPERATURE)
            print(
                "[WARN] Applied Unsloth GRPO compatibility shim: "
                f"args.temperature={TEMPERATURE}"
            )

        if not hasattr(args, "max_completion_length"):
            setattr(args, "max_completion_length", MAX_NEW_TOKENS)
            print(
                "[WARN] Applied Unsloth GRPO compatibility shim: "
                f"args.max_completion_length={MAX_NEW_TOKENS}"
            )
    except Exception as exc:
        print(f"[WARN] Could not apply args attr compatibility shim: {exc}")


def _maybe_patch_nanmin_symbols() -> None:
    """Provide nanmin/nanmax symbols expected by some generated trainer code."""
    try:
        import builtins
        import torch as _torch

        def _compat_nanmin(x, *args, **kwargs):
            if hasattr(_torch, "nanmin"):
                return _torch.nanmin(x, *args, **kwargs)
            x2 = _torch.nan_to_num(x, nan=float("inf"))
            if args or kwargs:
                return _torch.amin(x2, *args, **kwargs)
            return _torch.min(x2)

        def _compat_nanmax(x, *args, **kwargs):
            if hasattr(_torch, "nanmax"):
                return _torch.nanmax(x, *args, **kwargs)
            x2 = _torch.nan_to_num(x, nan=float("-inf"))
            if args or kwargs:
                return _torch.amax(x2, *args, **kwargs)
            return _torch.max(x2)

        if not hasattr(builtins, "nanmin"):
            builtins.nanmin = _compat_nanmin
            print("[WARN] Applied compatibility shim: builtins.nanmin")
        if not hasattr(builtins, "nanmax"):
            builtins.nanmax = _compat_nanmax
            print("[WARN] Applied compatibility shim: builtins.nanmax")
    except Exception as exc:
        print(f"[WARN] Could not apply nanmin/nanmax compatibility shim: {exc}")


def _resolve_num_generations(batch_size: int, requested: int) -> int:
    requested = max(1, requested)
    for candidate in range(min(requested, batch_size), 0, -1):
        if batch_size % candidate == 0:
            return candidate
    return 1


def _effective_kl_coeff() -> float:
    raw = os.getenv("ATC_KL_COEFF", str(KL_COEFF)).strip()
    try:
        value = float(raw)
    except ValueError:
        print(f"[WARN] Invalid ATC_KL_COEFF={raw!r}; using default {KL_COEFF}.")
        return KL_COEFF

    if value < 0.0:
        print(f"[WARN] Negative KL coeff {value} is invalid; clamping to 0.0.")
        value = 0.0

    if value > 0.0:
        print(
            "[WARN] Non-zero KL enabled. On trl==0.16.0 + unsloth==2026.4.7 this may "
            "trigger ref_per_token_logps=None errors with PEFT."
        )
    return value


def _select_sample_value(value: Any, index: int) -> Any:
    if isinstance(value, list):
        if not value:
            return None
        return value[index] if index < len(value) else value[-1]
    return value


def _expand_to_completion_length(
    value: Any,
    target_len: int,
    *,
    field_name: str,
    num_generations: int,
) -> List[Any]:
    """Align prompt-level metadata with completion-level batches."""
    if not isinstance(value, list):
        return [value] * target_len
    current_len = len(value)
    if current_len == target_len:
        return value
    if current_len == 0:
        return [None] * target_len
    if current_len * max(1, num_generations) == target_len:
        expanded: List[Any] = []
        for item in value:
            expanded.extend([item] * max(1, num_generations))
        return expanded
    if target_len % current_len == 0:
        factor = target_len // current_len
        expanded = []
        for item in value:
            expanded.extend([item] * factor)
        return expanded
    raise RuntimeError(
        f"Metadata alignment error for '{field_name}': len={current_len}, "
        f"target={target_len}, num_generations={num_generations}"
    )


def _build_curriculum_slices(
    dataset_raw: List[Dict[str, Any]],
    seed: int,
    *,
    adapt_focus: bool = False,
) -> Dict[str, List[Dict[str, Any]]]:
    """Create staged role mixes for pure-GRPO curriculum."""
    import random

    rng = random.Random(seed)
    adapt_rows = [
        row for row in dataset_raw
        if row.get("agent_role") == AgentRole.ADAPT.value
    ]
    stage_a = [
        row for row in dataset_raw
        if row.get("agent_role") in (AgentRole.AMAN.value, AgentRole.DMAN.value)
    ]
    if adapt_focus and adapt_rows:
        # ADAPT is otherwise only in stage_c; include it early to emphasize domain transfer.
        stage_a = stage_a + list(adapt_rows)
        rng.shuffle(stage_a)
    gen_sup = [
        row for row in dataset_raw
        if row.get("agent_role") in (AgentRole.GENERATOR.value, AgentRole.SUPERVISOR.value)
    ]
    rng.shuffle(gen_sup)
    keep = int(0.35 * len(gen_sup))
    stage_b = stage_a + gen_sup[:keep]
    stage_c = list(dataset_raw)
    return {"stage_a": stage_a, "stage_b": stage_b, "stage_c": stage_c}


def _tail_rate(values: List[int], window: int = 128) -> float:
    if not values:
        return 0.0
    tail = values[-min(window, len(values)) :]
    return sum(tail) / max(1, len(tail))


def _parse_quality_gates(parse_log: Dict[str, List[int]], fallback_log: Dict[str, List[int]]) -> Dict[str, float]:
    return {
        "parse_aman": _tail_rate(parse_log.get("AMAN", []), 128),
        "parse_dman": _tail_rate(parse_log.get("DMAN", []), 128),
        "fallback_aman": _tail_rate(fallback_log.get("AMAN", []), 128),
        "fallback_dman": _tail_rate(fallback_log.get("DMAN", []), 128),
    }


# ── Unified reward dispatcher ─────────────────────────────────────────────────

def combined_reward_fn(completions: List[str], **kwargs) -> List[float]:
    """Route each completion to its role-specific reward function.

    TRL GRPOTrainer calls this with a batch of completions.
    kwargs contains per-sample metadata from the dataset.
    """
    target_len = len(completions)
    normalized_kwargs: Dict[str, List[Any]] = {}
    for key, value in kwargs.items():
        normalized_kwargs[key] = _expand_to_completion_length(
            value,
            target_len,
            field_name=key,
            num_generations=N_GENERATIONS,
        )

    roles = normalized_kwargs.get("agent_role", [AgentRole.AMAN.value] * target_len)
    if not roles:
        roles = [AgentRole.AMAN.value] * target_len

    rewards: List[float] = []
    failure_mode = _reward_failure_mode()

    for i, (completion, role) in enumerate(zip(completions, roles)):
        fn = REWARD_FN_DISPATCH.get(role, aman_reward_fn)
        sample_kwargs = {k: [_select_sample_value(v, i)] for k, v in normalized_kwargs.items()}
        try:
            r = fn([completion], **sample_kwargs)
            if not r:
                raise RuntimeError(f"empty reward list for role={role}")
            rewards.append(r[0])
        except Exception as exc:
            msg = f"reward_fn({role}) failed at index={i}: {exc}"
            if failure_mode == "strict":
                raise RuntimeError(msg) from exc
            print(f"[WARN] {msg}")
            rewards.append(-1.0)

    return rewards


# ── Training entry point ──────────────────────────────────────────────────────

def train(
    model_name:   str  = DEFAULT_MODEL,
    output_dir:   str  = DEFAULT_OUTPUT,
    n_episodes:   int  = 200,
    lora_rank:    int  = LORA_RANK,
    seed:         int  = 42,
    eval_episodes: int = EVAL_EPISODES,
    push_to_hub:  bool = False,
    hub_model_id: Optional[str] = None,
    run_eval:     bool = True,
    *,
    domain_episode_ratio: float = 0.30,
    adapt_focus: bool = False,
    domain_stratify: bool = True,
    stage_epoch_scale: float = 1.0,
    adapt_eval_episodes: Optional[int] = None,
) -> None:
    torch, FastLanguageModel, GRPOConfig, GRPOTrainer = _require_training_deps()
    _configure_runtime_warnings()
    from transformers import TrainerCallback

    domain_episode_ratio = max(0.0, min(1.0, float(domain_episode_ratio)))

    tuned = _auto_tune_for_gpu(torch)
    batch_size = tuned["batch_size"]
    grad_accum = tuned["grad_accum"]
    max_new_tokens = max(tuned["max_new_tokens"], max(ROLE_MAX_NEW_TOKENS.values()))

    num_generations = _resolve_num_generations(batch_size, N_GENERATIONS)
    if num_generations != N_GENERATIONS:
        print(
            f"[WARN] Adjusted num_generations {N_GENERATIONS} -> {num_generations} "
            f"to satisfy GRPO batch-size divisibility constraint."
        )

    print(f"\n{'='*60}")
    print(f"  ATC Multi-Agent GRPO Training")
    print(f"  Model:        {model_name}")
    print(f"  Episodes:     {n_episodes}")
    print(f"  Eval eps:     {eval_episodes}")
    print(f"  Generations:  {num_generations} per prompt")
    print(f"  Output:       {output_dir}")
    device_str = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
    print(f"  Device:       {device_str}")
    print(
        f"  Tune:         batch={batch_size}, accum={grad_accum}, "
        f"max_new_tokens={max_new_tokens}, temp={TEMPERATURE}, logging_steps={LOGGING_STEPS}"
    )
    print(
        "  Role tokens:  "
        f"AMAN={ROLE_MAX_NEW_TOKENS[AgentRole.AMAN.value]}, "
        f"DMAN={ROLE_MAX_NEW_TOKENS[AgentRole.DMAN.value]}, "
        f"GEN={ROLE_MAX_NEW_TOKENS[AgentRole.GENERATOR.value]}, "
        f"SUP={ROLE_MAX_NEW_TOKENS[AgentRole.SUPERVISOR.value]}, "
        f"ADAPT={ROLE_MAX_NEW_TOKENS[AgentRole.ADAPT.value]}"
    )
    _aes = adapt_eval_episodes if adapt_eval_episodes is not None else max(3, eval_episodes)
    print(
        f"  ADAPT:        domain_ratio={domain_episode_ratio}  focus={adapt_focus}  "
        f"stratify_domains={domain_stratify}  stage_epoch_scale={stage_epoch_scale}  "
        f"adapt_eval_eps={_aes}"
    )
    _spg = os.getenv("ATC_STRICT_PARSE_GATE", "0").strip().lower() in {"1", "true", "yes", "on"}
    print(
        f"  Parse abort:  {'ON (Stage A can RuntimeError)' if _spg else 'OFF (warn only; set ATC_STRICT_PARSE_GATE=1 to enforce)'}"
    )
    print(f"{'='*60}\n")

    # ── 1. Capture pre-training baseline metrics ──────────────────────────────
    if run_eval:
        print("[0/5] Capturing pre-training baseline metrics...")
        baseline = _quick_heuristic_eval(n_episodes=min(10, n_episodes))
        _save_json(baseline, Path(output_dir) / "baseline_metrics.json")
        print(f"    Baseline composite: {baseline['mean_composite']:.3f}")

    # ── 2. Load model ─────────────────────────────────────────────────────────
    print("[1/5] Loading model with Unsloth 4-bit QLoRA...")
    model_source = _prefer_local_model_path(model_name)
    model, tokenizer = _load_model_with_fallback(
        FastLanguageModel,
        model_source,
        max_seq_length=MAX_SEQ_LEN,
    )
    # Prevent generate() ambiguity warnings from inherited max_length defaults.
    if hasattr(model, "generation_config") and model.generation_config is not None:
        model.generation_config.max_length = None
    model = FastLanguageModel.get_peft_model(
        model,
        r=lora_rank,
        lora_alpha=LORA_ALPHA,
        target_modules=LORA_TARGETS,
        lora_dropout=0.0,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=seed,
    )
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"    LoRA rank={lora_rank}, trainable params: {trainable:,}")

    # ── 2b. Base model eval (before any gradient steps) ──────────────────────
    base_model_metrics: Optional[Dict[str, Any]] = None
    if run_eval:
        print("\n[1.5/5] Measuring base model score (untrained LoRA)...")
        model.eval()
        base_model_metrics = _run_model_episodes(
            model, tokenizer, n_episodes=eval_episodes, tag="BASE MODEL (no fine-tune)"
        )
        base_adapt = _run_adapt_domain_eval(
            model,
            tokenizer,
            n_episodes=_aes,
            tag="BASE ADAPT (domain pipeline)",
        )
        base_model_metrics = _merge_adapt_eval(base_model_metrics, base_adapt)
        model.train()
        _save_json(base_model_metrics, Path(output_dir) / "base_model_metrics.json")
        print(f"    Base model composite: {base_model_metrics['mean_composite']:.3f}"
              f"  (AMAN {base_model_metrics['mean_aman_reward']:.3f}"
              f" / DMAN {base_model_metrics['mean_dman_reward']:.3f})")
        print(
            f"    Base ADAPT pipeline (domain→ATC): "
            f"{base_model_metrics.get('mean_adapt_pipeline_composite', 0.0):.3f}"
        )

    # ── 3. Build training dataset ─────────────────────────────────────────────
    print(f"\n[2/5] Building {n_episodes}-episode multi-agent dataset...")
    t0 = time.time()
    dataset_raw = build_episode_dataset(
        n_episodes=n_episodes,
        seed=seed,
        include_generator=True,
        include_supervisor=True,
        include_adapt=True,
        domain_episode_ratio=domain_episode_ratio,
        domain_stratify=domain_stratify,
    )

    print(f"    Dataset: {len(dataset_raw)} samples ({time.time()-t0:.1f}s)")

    role_counts: Dict[str, int] = {}
    for s in dataset_raw:
        r = s.get("agent_role", "unknown")
        role_counts[r] = role_counts.get(r, 0) + 1
    for role, count in sorted(role_counts.items()):
        print(f"    {role}: {count} samples")

    try:
        from datasets import Dataset
        stage_slices = _build_curriculum_slices(
            dataset_raw, seed=seed, adapt_focus=adapt_focus
        )
        stage_datasets = {
            name: Dataset.from_list(rows) for name, rows in stage_slices.items() if rows
        }
    except ImportError:
        print("[ERROR] pip install datasets")
        sys.exit(1)
    print("    Curriculum:")
    for stage_name in ("stage_a", "stage_b", "stage_c"):
        ds = stage_datasets.get(stage_name)
        if ds is not None:
            print(f"      {stage_name}: {len(ds)} samples")

    # ── 4. GRPO config ────────────────────────────────────────────────────────
    kl_coeff = _effective_kl_coeff()
    print(
        f"\n[3/5] Configuring GRPO (group_size={num_generations}, lr={LR}, kl={kl_coeff})..."
    )
    grpo_kwargs: Dict[str, Any] = {
        "num_generations":              num_generations,
        "temperature":                  TEMPERATURE,
        "learning_rate":                LR,
        "per_device_train_batch_size":  batch_size,
        "gradient_accumulation_steps":  grad_accum,
        "num_train_epochs":             1,
        "warmup_ratio":                 WARMUP_RATIO,
        "logging_steps":                LOGGING_STEPS,
        "save_steps":                   SAVE_STEPS,
        "save_total_limit":             SAVE_TOTAL_LIMIT,
        "output_dir":                   output_dir,
        "run_name":                     f"atc-multiagent-grpo-{int(time.time())}",
        "bf16":                         torch.cuda.is_bf16_supported(),
        "fp16":                         not torch.cuda.is_bf16_supported(),
        "gradient_checkpointing":       True,
        "optim":                        "paged_adamw_8bit",
    }

    if _wandb_available():
        grpo_kwargs["report_to"] = "wandb"
    else:
        grpo_kwargs["report_to"] = "none"

    # Compatibility shims for different TRL versions
    if _config_supports("max_completion_length", GRPOConfig):
        grpo_kwargs["max_completion_length"] = max_new_tokens
    elif _config_supports("max_new_tokens", GRPOConfig):
        grpo_kwargs["max_new_tokens"] = max_new_tokens

    if _config_supports("beta", GRPOConfig):
        grpo_kwargs["beta"] = kl_coeff
    elif _config_supports("kl_coeff", GRPOConfig):
        grpo_kwargs["kl_coeff"] = kl_coeff

    if _config_supports("use_vllm", GRPOConfig):
        grpo_kwargs["use_vllm"] = False

    if _config_supports("logging_strategy", GRPOConfig):
        grpo_kwargs["logging_strategy"] = "steps"
    if _config_supports("logging_first_step", GRPOConfig):
        grpo_kwargs["logging_first_step"] = True
    if _config_supports("disable_tqdm", GRPOConfig):
        grpo_kwargs["disable_tqdm"] = False

    grpo_config = GRPOConfig(**grpo_kwargs)

    # ── 5. Per-role reward logger ─────────────────────────────────────────────
    print("\n[4/5] Setting up per-role reward logging...")

    # Curriculum generator: receives real GRPO group rewards so the adaptive
    # curriculum can trigger rescue mode and tier changes based on actual training
    # performance rather than the simulated scores used during dataset building.
    curriculum_generator = ChallengeGenerator(seed=seed)

    # Separate lists so we can show per-role curves in the demo
    reward_log: Dict[str, List[float]] = {
        "AMAN": [], "DMAN": [], "GENERATOR": [], "SUPERVISOR": [], "ADAPT": [], "composite": []
    }
    parse_log: Dict[str, List[int]] = {
        "AMAN": [], "DMAN": [], "GENERATOR": [], "SUPERVISOR": [], "ADAPT": []
    }
    fallback_log: Dict[str, List[int]] = {
        "AMAN": [], "DMAN": [], "GENERATOR": [], "SUPERVISOR": [], "ADAPT": []
    }
    overflow_log: Dict[str, List[int]] = {
        "AMAN": [], "DMAN": [], "GENERATOR": [], "SUPERVISOR": [], "ADAPT": []
    }
    parse_fail_samples: Dict[str, str] = {}
    parse_fail_counts: Dict[str, int] = {
        "AMAN": 0, "DMAN": 0, "GENERATOR": 0, "SUPERVISOR": 0, "ADAPT": 0
    }
    reward_call_count = 0
    # Per-tier reward accumulator for ZPD logging
    tier_reward_log: Dict[int, List[float]] = {i: [] for i in range(5)}

    class RewardLogger:
        __name__ = "combined_reward_fn"

        def __call__(self, *args, **kwargs):
            nonlocal reward_call_count
            # TRL <0.17: reward_func(completions=..., **kwargs)
            # TRL >=0.17: reward_func(prompts, completions, **kwargs)
            if "completions" in kwargs:
                completions = kwargs.pop("completions")
            elif len(args) >= 2:
                completions = args[1]
            elif args:
                completions = args[0]
            else:
                completions = []
            kwargs.pop("prompts", None)
            kwargs.pop("prompt_ids", None)

            # Some TRL versions pass conversational turns. Extract the assistant text.
            if completions and isinstance(completions[0], list):
                flattened = []
                for c in completions:
                    if c and isinstance(c[-1], dict) and "content" in c[-1]:
                        flattened.append(c[-1]["content"])
                    else:
                        flattened.append(str(c))
                completions = flattened

            rewards = combined_reward_fn(completions, **kwargs)
            reward_call_count += 1

            # ── Degenerate-group detection + live curriculum update ────────────
            # When all N_GENERATIONS completions in a group have near-identical
            # rewards, std(group) ≈ 0 and GRPO produces zero gradient.
            # Two common causes:
            #   (a) All-wrong: task too hard — every completion parse-fails.
            #   (b) All-right: task too easy — every completion is correct.
            # We pass real group rewards into the curriculum generator so it can
            # trigger rescue mode (all-wrong) or advance tier (all-right) based
            # on actual training performance, not the simulated scores used during
            # dataset building.  (Research: VCRL arXiv 2509.19803, DAPO 2025)
            if len(rewards) >= 2:
                import statistics as _stats
                try:
                    grp_std = _stats.stdev(rewards)
                except _stats.StatisticsError:
                    grp_std = 0.0
                grp_mean = sum(rewards) / len(rewards)
                # Feed every group into the curriculum generator (not just degenerate ones)
                curriculum_generator.update(grp_mean, rewards)
                if grp_std < 0.01:
                    mode_tag = "ALL-WRONG" if grp_mean < -0.3 else "ALL-CORRECT"
                    if reward_call_count % 20 == 0:
                        print(
                            f"[ZPD_WARN] {mode_tag} group detected "
                            f"(std={grp_std:.4f}, mean={grp_mean:.3f}) "
                            f"at call={reward_call_count}.  "
                            f"{curriculum_generator.curriculum_summary()}"
                        )

            # ── Per-tier reward accumulation ──────────────────────────────────
            tier = kwargs.get("curriculum_tier", [curriculum_generator.current_tier] * len(rewards))
            if not isinstance(tier, list):
                tier = [tier] * len(rewards)
            for r, t in zip(rewards, tier):
                t_int = int(t) if isinstance(t, (int, float)) else curriculum_generator.current_tier
                tier_reward_log[t_int].append(r)

            roles = kwargs.get("agent_role", [])
            if not isinstance(roles, list):
                roles = [roles] * len(rewards)
            elif len(roles) < len(rewards):
                roles = roles + [
                    roles[-1] if roles else AgentRole.AMAN.value
                ] * (len(rewards) - len(roles))

            def _is_parse_ok(role: str, completion: Any) -> int:
                if role == AgentRole.AMAN.value:
                    return 1 if parse_aman_action(completion) is not None else 0
                if role == AgentRole.DMAN.value:
                    return 1 if parse_dman_action(completion) is not None else 0
                if role == AgentRole.GENERATOR.value:
                    return 1 if parse_generator_action(completion) is not None else 0
                if role == AgentRole.SUPERVISOR.value:
                    text = str(completion)
                    if re.search(r'"score"\s*:\s*-?\d+(?:\.\d+)?', text):
                        return 1
                    return 0
                if role == AgentRole.ADAPT.value:
                    from multi_agent.adapt import parse_adapt_action
                    return 1 if parse_adapt_action(completion) is not None else 0
                return 0

            def _is_fallback(role: str, parse_ok: int, reward_value: float) -> int:
                if role in (AgentRole.AMAN.value, AgentRole.DMAN.value):
                    return 0 if parse_ok else 1
                if role == AgentRole.ADAPT.value:
                    return 0 if parse_ok else 1
                if role == AgentRole.GENERATOR.value:
                    return 1 if reward_value <= -0.49 else 0
                if role == AgentRole.SUPERVISOR.value:
                    return 1 if reward_value <= -0.49 else 0
                return 0

            def _overflow(role: str, completion: Any) -> int:
                budget = ROLE_MAX_NEW_TOKENS.get(role, max_new_tokens)
                approx_tokens = len(str(completion).split())
                return 1 if approx_tokens > budget else 0

            for completion, r, role in zip(completions, rewards, roles):
                parse_ok = _is_parse_ok(role, completion)
                if role in reward_log:
                    reward_log[role].append(r)
                if role in parse_log:
                    parse_log[role].append(parse_ok)
                if role in fallback_log:
                    fallback_log[role].append(_is_fallback(role, parse_ok, float(r)))
                if role in overflow_log:
                    overflow_log[role].append(_overflow(role, completion))
                if role in parse_fail_counts and parse_ok == 0:
                    parse_fail_counts[role] += 1
                    if role in (AgentRole.AMAN.value, AgentRole.DMAN.value, AgentRole.ADAPT.value):
                        txt = re.sub(r"\s+", " ", str(completion)).strip()
                        parse_fail_samples[role] = txt[:220]
                reward_log["composite"].append(r)

            # Reward-hacking detection: warn when composite rises but per-role variance
            # collapses (all roles getting same score = likely gaming)
            if len(reward_log["composite"]) % 50 == 0 and len(reward_log["composite"]) > 50:
                _check_reward_hacking(reward_log)
            if reward_call_count % 25 == 0:
                for role in (AgentRole.AMAN.value, AgentRole.DMAN.value, AgentRole.ADAPT.value):
                    sample = parse_fail_samples.get(role)
                    if sample:
                        print(
                            f"[PARSE_FAIL_SAMPLE] role={role} count={parse_fail_counts[role]} "
                            f"sample='{sample}'"
                        )

            return rewards

    reward_logger = RewardLogger()

    # ── 6. Train ──────────────────────────────────────────────────────────────
    print("\n[5/5] Starting GRPO training...")
    trainer_kwargs: Dict[str, Any] = {
        "model":            model,
        "processing_class": tokenizer,
        "reward_funcs":     [reward_logger],
        "train_dataset":    stage_datasets.get("stage_a"),
    }
    if _trainer_supports("args", GRPOTrainer):
        trainer_kwargs["args"] = grpo_config
    else:
        trainer_kwargs["config"] = grpo_config

    trainer = GRPOTrainer(**trainer_kwargs)

    def _print_live_metrics(args, state, logs=None, *, prefix: str = "[LIVE]") -> None:
        """One stdout line of trainer + reward tail stats (used each log step and each epoch)."""
        logs = logs or {}
        if prefix == "[LIVE]" and not logs:
            return
        step = int(logs.get("step", getattr(state, "global_step", 0)) or 0)
        max_steps = int(getattr(state, "max_steps", 0) or 0)
        loss = logs.get("loss")
        lr = logs.get("learning_rate")

        def _avg_last(key: str, window: int = 64) -> float:
            vals = reward_log.get(key, [])
            if not vals:
                return float("nan")
            tail = vals[-min(window, len(vals)) :]
            return sum(tail) / max(1, len(tail))

        def _fmt(v: float) -> str:
            return "n/a" if v != v else f"{v:.3f}"

        def _parse_rate(role: str, window: int = 64) -> float:
            vals = parse_log.get(role, [])
            if not vals:
                return float("nan")
            tail = vals[-min(window, len(vals)) :]
            return sum(tail) / max(1, len(tail))

        def _tail_rate(store: Dict[str, List[int]], role: str, window: int = 64) -> float:
            vals = store.get(role, [])
            if not vals:
                return float("nan")
            tail = vals[-min(window, len(vals)) :]
            return sum(tail) / max(1, len(tail))

        comp = _avg_last("composite")
        aman = _avg_last("AMAN")
        dman = _avg_last("DMAN")
        gen = _avg_last("GENERATOR")
        sup = _avg_last("SUPERVISOR")
        adapt_r = _avg_last("ADAPT")
        p_aman = _parse_rate("AMAN")
        p_dman = _parse_rate("DMAN")
        p_gen = _parse_rate("GENERATOR")
        p_sup = _parse_rate("SUPERVISOR")
        p_adapt = _parse_rate("ADAPT")
        f_aman = _tail_rate(fallback_log, "AMAN")
        f_dman = _tail_rate(fallback_log, "DMAN")
        f_adapt = _tail_rate(fallback_log, "ADAPT")
        o_aman = _tail_rate(overflow_log, "AMAN")
        o_dman = _tail_rate(overflow_log, "DMAN")
        o_adapt = _tail_rate(overflow_log, "ADAPT")
        print(
            f"{prefix} "
            f"step={step}/{max_steps} "
            f"loss={loss if loss is not None else 'n/a'} "
            f"lr={lr if lr is not None else 'n/a'} "
            f"comp64={_fmt(comp)} AMAN={_fmt(aman)} DMAN={_fmt(dman)} "
            f"ADAPT={_fmt(adapt_r)} GEN={_fmt(gen)} SUP={_fmt(sup)} "
            f"parse64[A={_fmt(p_aman)} D={_fmt(p_dman)} T={_fmt(p_adapt)} G={_fmt(p_gen)} S={_fmt(p_sup)}] "
            f"fb64[A={_fmt(f_aman)} D={_fmt(f_dman)} T={_fmt(f_adapt)}] "
            f"ovf64[A={_fmt(o_aman)} D={_fmt(o_dman)} T={_fmt(o_adapt)}]",
            flush=True,
        )

    class LiveMetricsCallback(TrainerCallback):
        """Stream concise live metrics into notebook/stdout while training."""

        def on_log(self, args, state, control, logs=None, **kwargs):
            _print_live_metrics(args, state, logs, prefix="[LIVE]")

        def on_epoch_end(self, args, state, control, **kwargs):
            ep = float(getattr(state, "epoch", 0.0) or 0.0)
            gs = int(getattr(state, "global_step", 0) or 0)
            hist = list(getattr(state, "log_history", []) or [])
            last = hist[-1] if hist else {}
            loss = last.get("loss")
            lr = last.get("learning_rate")
            loss_s = f"{loss:.4f}" if isinstance(loss, (int, float)) else str(loss)
            lr_s = f"{lr:.2e}" if isinstance(lr, (int, float)) else str(lr)
            print(
                f"\n{'='*60}\n"
                f"  [GRPO] EPOCH END  epoch={ep:.2f}  global_step={gs}\n"
                f"  last_log: loss={loss_s}  lr={lr_s}\n"
                f"{'='*60}",
                flush=True,
            )
            _print_live_metrics(args, state, last, prefix="[GRPO EPOCH]")

    if hasattr(trainer, "add_callback"):
        live_cb = LiveMetricsCallback()
        # Fail fast before trainer.train() if callback API is incompatible.
        missing = [name for name in ("on_train_begin", "on_log", "on_train_end") if not hasattr(live_cb, name)]
        if missing:
            raise RuntimeError(
                "LiveMetricsCallback is incompatible with Trainer API; "
                f"missing methods: {missing}"
            )
        trainer.add_callback(live_cb)
    
    # ── CRITICAL: Apply ALL compatibility patches BEFORE any training ──
    _maybe_patch_trainer_sampler(trainer)
    _maybe_patch_unsloth_grad_accum(trainer)
    _maybe_patch_unsloth_loss_type(trainer)
    _maybe_patch_unsloth_runtime_attrs(trainer)
    _maybe_patch_unsloth_args_attrs(trainer)
    _maybe_patch_nanmin_symbols()
    
    # ── ADD THIS: Extra safety check for any remaining missing attrs ──
    # Some Unsloth compiled paths access these directly on the trainer
    _safety_attrs = {
        "importance_sampling_level": "token",
        "epsilon_low": 0.2,
        "epsilon_high": 0.2,
        "vllm_importance_sampling_cap": 2.0,
        "current_gradient_accumulation_steps": getattr(trainer, "current_gradient_accumulation_steps", 1),
    }
    for attr, default in _safety_attrs.items():
        if not hasattr(trainer, attr):
            setattr(trainer, attr, default)
            print(f"[SAFETY] Added missing trainer.{attr} = {default!r}")
    
    # ── ADD THIS: Delete stale compiled cache so Unsloth rebuilds with patched attrs ──
    import shutil
    compiled_cache_dirs = [
        Path.cwd() / "unsloth_compiled_cache",
        Path.home() / ".cache" / "unsloth" / "compiled_cache",
    ]
    for cache_dir in compiled_cache_dirs:
        if cache_dir.exists():
            shutil.rmtree(cache_dir, ignore_errors=True)
            print(f"[INFO] Deleted stale compiled cache: {cache_dir}")

    strict_gates = os.getenv("ATC_STRICT_GATES", "1").strip().lower() in {"1", "true", "yes", "on"}
    # Stage-A parse abort is opt-in: short Colab runs + mixed ADAPT batches rarely hit 0.75 DMAN in one stage.
    strict_parse_gate = os.getenv("ATC_STRICT_PARSE_GATE", "0").strip().lower() in {
        "1", "true", "yes", "on",
    }
    se_scale = max(0.01, min(4.0, float(stage_epoch_scale)))
    stage_sequence = ["stage_a", "stage_b", "stage_c"]
    stage_a_retries = 0
    # Treat high domain ratio like adapt_focus for thresholds (even if --adapt_focus was omitted).
    mixed_stage_a = bool(adapt_focus) or float(domain_episode_ratio) >= 0.25
    quick_run = se_scale < 0.5

    def _safe_float(v: Any, default: float = 0.0) -> float:
        try:
            return float(v)
        except Exception:
            return default

    def _optimization_gates() -> Dict[str, float]:
        history = list(getattr(getattr(trainer, "state", None), "log_history", []) or [])
        reward_std_vals = [
            _safe_float(entry.get("reward_std"), 0.0)
            for entry in history
            if "reward_std" in entry
        ]
        clip_vals = [
            _safe_float(entry.get("clip_ratio/region_mean"), 0.0)
            for entry in history
            if "clip_ratio/region_mean" in entry
        ]
        clip_nonzero_frac = (
            sum(1 for v in clip_vals if v > 1e-8) / max(1, len(clip_vals))
            if clip_vals else 0.0
        )
        reward_std_med = (
            sorted(reward_std_vals)[len(reward_std_vals) // 2]
            if reward_std_vals else 0.0
        )
        return {"clip_nonzero_frac": clip_nonzero_frac, "reward_std_median": reward_std_med}

    for stage_name in stage_sequence:
        ds = stage_datasets.get(stage_name)
        if ds is None or len(ds) == 0:
            continue
        if hasattr(trainer, "train_dataset"):
            trainer.train_dataset = ds
        if hasattr(trainer, "_train_dataloader"):
            trainer._train_dataloader = None
        if hasattr(trainer, "args"):
            trainer.args.num_train_epochs = STAGE_EPOCHS.get(stage_name, 0.25) * se_scale
        print(
            f"\n[STAGE] {stage_name} samples={len(ds)} "
            f"epochs={getattr(getattr(trainer, 'args', None), 'num_train_epochs', 'n/a')}"
        )
        trainer.train()

        parse_gates = _parse_quality_gates(parse_log, fallback_log)
        opt_gates = _optimization_gates()
        print(
            f"[GATE] {stage_name} "
            f"parseA={parse_gates['parse_aman']:.3f} parseD={parse_gates['parse_dman']:.3f} "
            f"fbA={parse_gates['fallback_aman']:.3f} fbD={parse_gates['fallback_dman']:.3f} "
            f"clipNonZero={opt_gates['clip_nonzero_frac']:.3f} "
            f"rewardStdMed={opt_gates['reward_std_median']:.4f}"
        )

        if stage_name == "stage_a":
            # Softer retry trigger when stage_a mixes ADAPT or when epochs are scaled down.
            if mixed_stage_a or quick_run:
                retry_bar_a, retry_bar_d = 0.72, 0.72
            else:
                retry_bar_a, retry_bar_d = 0.85, 0.85
            parse_ok = (
                parse_gates["parse_aman"] >= retry_bar_a
                and parse_gates["parse_dman"] >= retry_bar_d
            )
            if not parse_ok and stage_a_retries < 1:
                stage_a_retries += 1
                if hasattr(trainer, "args"):
                    trainer.args.num_train_epochs = 0.20 * se_scale
                print("[GATE] Stage A parse gate missed; running one extra controller-only pass.")
                trainer.train()
                parse_gates = _parse_quality_gates(parse_log, fallback_log)
                print(
                    f"[GATE] stage_a_retry parseA={parse_gates['parse_aman']:.3f} "
                    f"parseD={parse_gates['parse_dman']:.3f}"
                )
            # Abort only if ATC_STRICT_PARSE_GATE=1 (see banner at train() start).
            if mixed_stage_a or quick_run:
                req_a, req_d = 0.68, 0.42
            else:
                req_a, req_d = 0.75, 0.75
            pa, pd = parse_gates["parse_aman"], parse_gates["parse_dman"]
            if pa < req_a or pd < req_d:
                print(
                    f"[WARN] Stage A parse tail: AMAN={pa:.3f} (guide {req_a:.2f}), "
                    f"DMAN={pd:.3f} (guide {req_d:.2f}) — "
                    "continuing (stages B/C add more controller steps). "
                    "To hard-fail here: export ATC_STRICT_PARSE_GATE=1."
                )
            if strict_gates and strict_parse_gate and not (pa >= req_a and pd >= req_d):
                raise RuntimeError(
                    "Parse gate failed after Stage A (ATC_STRICT_PARSE_GATE=1): "
                    f"AMAN={pa:.3f}, DMAN={pd:.3f}"
                )
        # Short / mixed curricula often lack stable clip stats — skip when adapt_focus or very fast stages.
        if strict_gates and not adapt_focus and se_scale >= 0.35 and (
            opt_gates["clip_nonzero_frac"] < 0.20
            or opt_gates["reward_std_median"] < 0.02
        ):
            raise RuntimeError(
                f"Optimization gate failed in {stage_name}: "
                f"clip_nonzero_frac={opt_gates['clip_nonzero_frac']:.3f}, "
                f"reward_std_median={opt_gates['reward_std_median']:.4f}"
            )
        if strict_gates and adapt_focus and se_scale >= 0.35 and (
            opt_gates["clip_nonzero_frac"] < 0.20
            or opt_gates["reward_std_median"] < 0.02
        ):
            print(
                f"[WARN] Optimization gate soft (adapt_focus): "
                f"clip_nonZero={opt_gates['clip_nonzero_frac']:.3f}, "
                f"rewardStdMed={opt_gates['reward_std_median']:.4f} — continuing."
            )

    # ── Save ──────────────────────────────────────────────────────────────────
    print(f"\nSaving model to {output_dir}...")
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)

    curves_path = Path(output_dir) / "reward_curves.json"
    _save_json(reward_log, curves_path)
    print(f"Reward curves -> {curves_path}")

    # Save per-tier reward summary and curriculum state
    tier_summary = {
        str(tier): {
            "n_samples": len(rws),
            "mean":      round(sum(rws) / len(rws), 4) if rws else None,
            "first_q":   round(sum(rws[:max(1, len(rws)//4)]) / max(1, len(rws)//4), 4) if rws else None,
            "last_q":    round(sum(rws[max(0,3*len(rws)//4):]) / max(1, len(rws)-3*len(rws)//4), 4) if rws else None,
        }
        for tier, rws in tier_reward_log.items()
    }
    tier_curves_path = Path(output_dir) / "tier_reward_summary.json"
    _save_json(tier_summary, tier_curves_path)

    curriculum_report = curriculum_generator.mastery_report()
    _save_json(curriculum_report, Path(output_dir) / "curriculum_state.json")

    print("\n── Curriculum Summary ──────────────────────────────────────")
    print(curriculum_generator.curriculum_summary())
    print("Per-tier mean rewards:")
    for tier, stats in tier_summary.items():
        if stats["n_samples"] > 0:
            print(f"  Tier {tier}: n={stats['n_samples']}  mean={stats['mean']}  "
                  f"first_q={stats['first_q']}  last_q={stats['last_q']}")
    print("────────────────────────────────────────────────────────────\n")

    _print_final_stats(reward_log)

    # ── Post-training eval ────────────────────────────────────────────────────
    if run_eval:
        print("\n[Post] Measuring trained model score...")
        FastLanguageModel.for_inference(model)  # fuse LoRA weights for faster generation
        trained_model_metrics = _run_model_episodes(
            model, tokenizer, n_episodes=eval_episodes, tag="TRAINED MODEL"
        )
        trained_adapt = _run_adapt_domain_eval(
            model, tokenizer, n_episodes=_aes, tag="TRAINED ADAPT (domain pipeline)"
        )
        trained_model_metrics = _merge_adapt_eval(trained_model_metrics, trained_adapt)
        _save_json(trained_model_metrics, Path(output_dir) / "trained_model_metrics.json")
        if base_model_metrics is not None:
            _plot_payload = {
                "base":    _plot_payload_from_metrics(base_model_metrics),
                "trained": _plot_payload_from_metrics(trained_model_metrics),
            }
            _save_json(_plot_payload, Path(output_dir) / "grpo_before_after.json")
            _print_improvement(base_model_metrics, trained_model_metrics)
            delta_comp = float(trained_model_metrics.get("mean_composite", 0.0)) - float(
                base_model_metrics.get("mean_composite", 0.0)
            )
            delta_aman = float(trained_model_metrics.get("mean_aman_reward", 0.0)) - float(
                base_model_metrics.get("mean_aman_reward", 0.0)
            )
            delta_dman = float(trained_model_metrics.get("mean_dman_reward", 0.0)) - float(
                base_model_metrics.get("mean_dman_reward", 0.0)
            )
            delta_adapt = float(trained_model_metrics.get("mean_adapt_pipeline_composite", 0.0)) - float(
                base_model_metrics.get("mean_adapt_pipeline_composite", 0.0)
            )
            print(
                f"[GATE] quality delta_comp={delta_comp:+.3f} "
                f"delta_aman={delta_aman:+.3f} delta_dman={delta_dman:+.3f} "
                f"delta_adapt={delta_adapt:+.3f}"
            )
            if strict_gates and not adapt_focus and not (
                delta_comp >= 0.0 and (delta_aman > 0.0 or delta_dman > 0.0)
            ):
                raise RuntimeError(
                    "Quality gate failed: expected non-negative composite delta and at least one "
                    "controller reward improvement."
                )
        else:
            # Fallback: compare heuristic baseline vs trained model
            _print_improvement(
                {**baseline, "tag": "HEURISTIC BASELINE"},
                {**trained_model_metrics, "tag": "TRAINED MODEL"},
            )

    if push_to_hub and hub_model_id:
        print(f"\nPushing to Hub: {hub_model_id}")
        trainer.push_to_hub(hub_model_id)

    return trainer


# ── Quick heuristic eval (no LLM needed — uses planner baseline) ──────────────

def _quick_heuristic_eval(n_episodes: int = 6) -> Dict[str, Any]:
    """Run heuristic-only multi-agent episodes (client=None → deterministic planner).

    Uses run_episode so metrics are AMAN/DMAN rewards from the real multi-agent
    environment — not single-agent grades. Same format as _run_model_episodes so
    _print_improvement can compare them directly.
    """
    from multi_agent.inference import run_episode as _run_ep

    env = MultiAgentATCEnvironment(seed=99)
    sup = SupervisorAgent()

    # Fixed task list — no generator mutations for a stable repeatable baseline
    eval_tasks = ["delhi_monsoon_recovery_easy", "bengaluru_irrops_hard"]

    composites, aman_rews, dman_rews, conflict_list, emg_list = [], [], [], [], []

    for ep in range(n_episodes):
        task_id = eval_tasks[ep % len(eval_tasks)]
        try:
            r = _run_ep(
                task_id      = task_id,
                client       = None,   # heuristic mode — no LLM
                env          = env,
                generator    = None,
                supervisor   = sup,
                episode_id   = ep,
                use_generator= False,
            )
            composites.append(float(r.get("composite", 0)))
            aman_rews.append(float(r.get("aman_reward", 0)))
            dman_rews.append(float(r.get("dman_reward", 0)))
            conflict_list.append(int(r.get("conflicts", 0)))
            emg_list.append(int(r.get("emg_arr_ok", 0)) + int(r.get("emg_dep_ok", 0)))
        except Exception as exc:
            print(f"  [WARN] Heuristic eval ep {ep} failed: {exc}")

    def _mean(lst: list) -> float:
        return round(sum(lst) / max(1, len(lst)), 3) if lst else 0.0

    return {
        "tag":              "HEURISTIC BASELINE",
        "n_episodes":       n_episodes,
        "mean_composite":   _mean(composites),
        "mean_aman_reward": _mean(aman_rews),
        "mean_dman_reward": _mean(dman_rews),
        "mean_conflicts":   _mean(conflict_list),
        "mean_emg_handled": _mean(emg_list),
        "scores":           [round(s, 3) for s in composites],
    }


def _print_improvement(
    before: Dict[str, Any], after: Dict[str, Any]
) -> None:
    tag_b = before.get("tag", "BEFORE")
    tag_a = after.get("tag", "AFTER")
    rows = [
        ("mean_composite",   "Composite score"),
        ("mean_aman_reward", "AMAN reward"),
        ("mean_dman_reward", "DMAN reward"),
        ("mean_adapt_pipeline_composite", "ADAPT pipeline (domain→ATC)"),
        ("mean_conflicts",   "Avg conflicts"),
        ("mean_emg_handled", "Emg handled"),
    ]
    width = 56
    print(f"\n{'='*width}")
    print(f"  BEFORE vs AFTER TRAINING")
    print(f"  {tag_b!r:24s}  →  {tag_a!r}")
    print(f"{'='*width}")
    for key, label in rows:
        if key == "mean_adapt_pipeline_composite" and "mean_adapt_pipeline_composite" not in before and "mean_adapt_pipeline_composite" not in after:
            continue
        bv = before.get(key, 0.0)
        av = after.get(key, 0.0)
        if isinstance(bv, dict):
            continue
        if isinstance(av, dict):
            continue
        delta = float(av) - float(bv)
        arrow = "↑" if delta > 0.005 else ("↓" if delta < -0.005 else "→")
        sign = "+" if delta >= 0 else ""
        print(f"  {label:20s}: {float(bv):6.3f}  →  {float(av):6.3f}  ({sign}{delta:.3f} {arrow})")
    print(f"{'='*width}")


# ── Local model client for in-process inference eval ──────────────────────────

class _LocalModelClient:
    """Duck-type OpenAI client wrapping a locally loaded Unsloth/PEFT model."""

    def __init__(self, model, tokenizer):
        self._model = model
        self._tokenizer = tokenizer
        if hasattr(self._model, "generation_config") and self._model.generation_config is not None:
            self._model.generation_config.max_length = None

    def _create(self, *, model=None, messages, temperature=0.3, max_tokens=MAX_NEW_TOKENS, **kw):
        import torch
        prompt = self._tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self._tokenizer(prompt, return_tensors="pt").to(self._model.device)
        use_cuda = torch.cuda.is_available() and str(self._model.device).startswith("cuda")
        cast_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        with torch.no_grad():
            if use_cuda:
                with torch.autocast(device_type="cuda", dtype=cast_dtype):
                    out = self._model.generate(
                        **inputs,
                        max_new_tokens=min(int(max_tokens), MAX_NEW_TOKENS),
                        temperature=max(float(temperature), 0.01),
                        do_sample=float(temperature) > 0.01,
                        pad_token_id=self._tokenizer.eos_token_id,
                    )
            else:
                out = self._model.generate(
                    **inputs,
                    max_new_tokens=min(int(max_tokens), MAX_NEW_TOKENS),
                    temperature=max(float(temperature), 0.01),
                    do_sample=float(temperature) > 0.01,
                    pad_token_id=self._tokenizer.eos_token_id,
                )
        text = self._tokenizer.decode(
            out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True
        )

        class _Msg:
            content = text
        class _Choice:
            message = _Msg()
        class _Resp:
            choices = [_Choice()]

        return _Resp()

    @property
    def chat(self):
        _self = self
        class _Comp:
            def create(self, **kw):
                return _self._create(**kw)
        class _Chat:
            completions = _Comp()
        return _Chat()


def _run_model_episodes(
    model,
    tokenizer,
    n_episodes: int = 3,
    tag: str = "MODEL",
    use_generator: bool = False,
) -> Dict[str, Any]:
    """Run multi-agent episodes using an in-process model client.

    use_generator=False keeps tasks fixed so base and trained models
    see identical scenarios — essential for a fair comparison.
    """
    from multi_agent.inference import run_episode as _run_ep

    client = _LocalModelClient(model, tokenizer)
    env = MultiAgentATCEnvironment(seed=77)
    gen = ChallengeGenerator(seed=77)
    sup = SupervisorAgent()

    # Two representative tasks: one easy, one hard
    eval_tasks = ["delhi_monsoon_recovery_easy", "bengaluru_irrops_hard"]

    composites, aman_rews, dman_rews, conflict_list, emg_list = [], [], [], [], []

    for ep in range(n_episodes):
        task_id = eval_tasks[ep % len(eval_tasks)]
        try:
            r = _run_ep(
                task_id=task_id,
                client=client,
                env=env,
                generator=gen if use_generator else None,
                supervisor=sup,
                episode_id=ep,
                use_generator=use_generator,
                model_name="local",
            )
            composites.append(float(r.get("composite", 0)))
            aman_rews.append(float(r.get("aman_reward", 0)))
            dman_rews.append(float(r.get("dman_reward", 0)))
            conflict_list.append(int(r.get("conflicts", 0)))
            emg_list.append(
                int(r.get("emg_arr_ok", 0)) + int(r.get("emg_dep_ok", 0))
            )
        except Exception as exc:
            print(f"  [WARN] model eval episode {ep} failed: {exc}")

    def _m(lst: list) -> float:
        return round(sum(lst) / max(1, len(lst)), 3) if lst else 0.0

    return {
        "tag":              tag,
        "n_episodes":       n_episodes,
        "mean_composite":   _m(composites),
        "mean_aman_reward": _m(aman_rews),
        "mean_dman_reward": _m(dman_rews),
        "mean_conflicts":   _m(conflict_list),
        "mean_emg_handled": _m(emg_list),
        "scores":           [round(s, 3) for s in composites],
    }


def _merge_adapt_eval(core: Dict[str, Any], adapt_block: Dict[str, Any]) -> Dict[str, Any]:
    """Attach ADAPT domain-pipeline metrics (end-to-end composite after domain→ATC)."""
    out = {**core}
    m = float(adapt_block.get("mean_composite", 0.0) or 0.0)
    out["mean_adapt_pipeline_composite"] = round(m, 3)
    out["adapt_eval_tag"] = adapt_block.get("tag", "")
    out["adapt_by_domain"] = adapt_block.get("by_domain", {})
    return out


def _run_adapt_domain_eval(
    model,
    tokenizer,
    n_episodes: int,
    tag: str,
) -> Dict[str, Any]:
    """Evaluate ADAPT on registered domain tasks (cycles tasks for coverage)."""
    from multi_agent.inference import run_domain_episode
    from domains import get_all_domain_tasks

    client = _LocalModelClient(model, tokenizer)
    env = MultiAgentATCEnvironment(seed=123)
    sup = SupervisorAgent()
    tasks = get_all_domain_tasks()
    if not tasks:
        return {
            "tag":              tag,
            "n_episodes":       0,
            "mean_composite":   0.0,
            "mean_aman_reward": 0.0,
            "mean_dman_reward": 0.0,
            "by_domain":        {},
            "scores":           [],
        }

    ids = sorted(tasks.keys())
    composites, aman_rews, dman_rews = [], [], []
    by_domain: Dict[str, float] = {}
    n = max(1, int(n_episodes))
    for ep in range(n):
        tid = ids[ep % len(ids)]
        try:
            r = run_domain_episode(
                tid, client, env, sup, ep, model_name="local",
            )
            c = float(r.get("composite", 0.0))
            a = float(r.get("aman_reward", 0.0))
            d = float(r.get("dman_reward", 0.0))
            composites.append(c)
            aman_rews.append(a)
            dman_rews.append(d)
            by_domain[tid] = c
        except Exception as exc:
            print(f"  [WARN] {tag} adapt eval ep={ep} task={tid}: {exc}")

    def _m(lst: list) -> float:
        return round(sum(lst) / max(1, len(lst)), 3) if lst else 0.0

    return {
        "tag":              tag,
        "n_episodes":       len(composites),
        "mean_composite":   _m(composites),
        "mean_aman_reward": _m(aman_rews),
        "mean_dman_reward": _m(dman_rews),
        "by_domain":        by_domain,
        "scores":           [round(s, 3) for s in composites],
    }


def _plot_payload_from_metrics(m: Dict[str, Any]) -> Dict[str, float]:
    """Normalize metrics for training/plot_rewards.py bar chart."""
    mc = float(m.get("mean_composite", 0.0) or 0.0)
    return {
        "mean_composite":    mc,
        "mean_aman":         float(m.get("mean_aman_reward", 0.0) or 0.0),
        "mean_dman":         float(m.get("mean_dman_reward", 0.0) or 0.0),
        "mean_coord":        0.0,
        "mean_adapt":        float(m.get("mean_adapt_pipeline_composite", 0.0) or 0.0),
        "success_rate":      round(sum(1 for s in m.get("scores", []) if float(s) >= 0.60) / max(1, len(m.get("scores", []))), 3),
    }


# ── Utilities ─────────────────────────────────────────────────────────────────

def _wandb_available() -> bool:
    try:
        import wandb
        return True
    except ImportError:
        return False


def _save_json(data: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def _check_reward_hacking(reward_log: Dict[str, List[float]]) -> None:
    """Warn when mean composite rises but role rewards collapse (gaming signal)."""
    comp = reward_log["composite"]
    if len(comp) < 100:
        return
    recent_50  = comp[-50:]
    earlier_50 = comp[-100:-50]
    mean_recent  = sum(recent_50)  / 50
    mean_earlier = sum(earlier_50) / 50
    if mean_recent > mean_earlier + 0.1:
        # Check if any role's recent std collapsed (< 0.05 = suspiciously uniform)
        for role in ("AMAN", "DMAN"):
            rs = reward_log.get(role, [])
            if len(rs) >= 20:
                recent = rs[-20:]
                mean_r = sum(recent) / len(recent)
                std_r = (sum((x - mean_r) ** 2 for x in recent) / len(recent)) ** 0.5
                if std_r < 0.05:
                    print(
                        f"[WARN] Possible reward hacking: {role} std={std_r:.4f} "
                        f"while composite reward is rising. Sample outputs and inspect."
                    )


def _print_final_stats(reward_log: Dict[str, List[float]]) -> None:
    print("\n=== TRAINING REWARD SUMMARY ===")
    for role, rewards in reward_log.items():
        if not rewards:
            continue
        n = len(rewards)
        first_q = rewards[:max(1, n // 4)]
        last_q  = rewards[max(0, 3 * n // 4):]
        mean_all   = sum(rewards) / n
        mean_first = sum(first_q) / len(first_q)
        mean_last  = sum(last_q)  / len(last_q)
        trend = "↑" if mean_last > mean_first + 0.05 else (
            "↓" if mean_last < mean_first - 0.05 else "→"
        )
        print(
            f"  {role:12s}: mean={mean_all:.3f} | "
            f"first_q={mean_first:.3f} -> last_q={mean_last:.3f} {trend}"
        )


# ── Evaluation loop ───────────────────────────────────────────────────────────

def evaluate(model_name_or_path: str, n_episodes: int = 20, seed: int = 99) -> Dict[str, Any]:
    """Run trained model on evaluation episodes."""
    torch, FastLanguageModel, _, _ = _require_training_deps()
    _configure_runtime_warnings()

    print(f"\nEvaluating {model_name_or_path} on {n_episodes} episodes...")
    model_source = _prefer_local_model_path(model_name_or_path)
    model, tokenizer = _load_model_with_fallback(
        FastLanguageModel,
        model_source,
        max_seq_length=MAX_SEQ_LEN,
    )
    if hasattr(model, "generation_config") and model.generation_config is not None:
        model.generation_config.max_length = None
    FastLanguageModel.for_inference(model)

    import random
    from training.dataset import AMAN_SYSTEM, DMAN_SYSTEM, SUPERVISOR_PROFILES

    env        = MultiAgentATCEnvironment(seed=seed)
    generator  = ChallengeGenerator(seed=seed)
    supervisor = SupervisorAgent()
    task_list  = list(ordered_tasks())
    rng        = random.Random(seed)

    results: Dict[str, List] = {
        "aman_rewards": [], "dman_rewards": [], "composite_scores": [],
        "conflict_counts": [], "coordination_scores": [],
        "generator_difficulty": [],
    }

    for ep in range(n_episodes):
        base_task = rng.choice(task_list)
        profile   = supervisor.sample_profile(ep)
        mutated, is_solvable = generator.mutate(base_task)

        aman_obs, dman_obs = env.reset(episode_id=ep, mutated_task=mutated)
        sup_desc = SUPERVISOR_PROFILES[profile]["description"]

        def _chat(system, user):
            msgs = [{"role": "system", "content": system}, {"role": "user", "content": user}]
            return tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)

        def _gen(prompt):
            import torch as _torch
            inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
            with _torch.no_grad():
                out = model.generate(
                    **inputs, max_new_tokens=MAX_NEW_TOKENS,
                    temperature=TEMPERATURE, do_sample=True,
                    pad_token_id=tokenizer.eos_token_id,
                )
            return tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)

        aman_comp = _gen(_chat(AMAN_SYSTEM + f"\n\nSUPERVISOR: {sup_desc}", aman_obs.to_prompt_text()))
        dman_comp = _gen(_chat(DMAN_SYSTEM + f"\n\nSUPERVISOR: {sup_desc}", dman_obs.to_prompt_text()))

        aman_action = parse_aman_action(aman_comp)
        dman_action = parse_dman_action(dman_comp)
        if not aman_action or not dman_action:
            continue

        aman_obs, dman_obs, _, done = env.step_bid(aman_action, dman_action)
        if not done:
            env.step_negotiate(aman_action, dman_action)

        result = env.finalize()
        generator.update(result.composite_score)

        results["aman_rewards"].append(result.aman_reward)
        results["dman_rewards"].append(result.dman_reward)
        results["composite_scores"].append(result.composite_score)
        results["conflict_counts"].append(result.per_role.cross_lane_conflicts)
        results["coordination_scores"].append(result.per_role.coordination_score)
        results["generator_difficulty"].append(generator.difficulty_level)

        print(
            f"  ep{ep:3d} | composite={result.composite_score:.3f} | "
            f"AMAN={result.aman_reward:.3f} | DMAN={result.dman_reward:.3f} | "
            f"coord={result.per_role.coordination_score:.3f} | "
            f"gen_lvl={generator.difficulty_level}"
        )

    def _mean(lst):
        return round(sum(lst) / max(1, len(lst)), 3)

    summary = {
        "mean_composite":    _mean(results["composite_scores"]),
        "mean_aman_reward":  _mean(results["aman_rewards"]),
        "mean_dman_reward":  _mean(results["dman_rewards"]),
        "mean_coordination": _mean(results["coordination_scores"]),
        "mean_conflicts":    _mean(results["conflict_counts"]),
        "final_gen_difficulty": results["generator_difficulty"][-1] if results["generator_difficulty"] else 1,
    }
    print("\n=== EVALUATION SUMMARY ===")
    for k, v in summary.items():
        print(f"  {k}: {v}")

    return {**results, "summary": summary}


# ── CLI ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="ATC Multi-Agent GRPO Training")
    parser.add_argument("--model",          default=DEFAULT_MODEL)
    parser.add_argument("--output_dir",     default=DEFAULT_OUTPUT)
    parser.add_argument("--episodes",       type=int, default=200)
    parser.add_argument("--lora_rank",      type=int, default=LORA_RANK)
    parser.add_argument("--n_generations",  type=int, default=None,
                        help="GRPO group size (default: N_GENERATIONS constant). "
                             "Use 2 on T4 Colab, 4 for best gradient quality.")
    parser.add_argument("--batch_size",     type=int, default=None)
    parser.add_argument("--grad_accum",     type=int, default=None)
    parser.add_argument("--max_new_tokens", type=int, default=None)
    parser.add_argument("--temperature",    type=float, default=None)
    parser.add_argument("--logging_steps",  type=int, default=None)
    parser.add_argument("--eval_episodes",  type=int, default=EVAL_EPISODES)
    parser.add_argument("--seed",           type=int, default=42)
    parser.add_argument("--no_eval",        action="store_true", help="Skip before/after eval")
    parser.add_argument("--eval_only",      action="store_true")
    parser.add_argument("--push_to_hub",    action="store_true")
    parser.add_argument("--hub_model_id",   default=None)
    parser.add_argument(
        "--domain_episode_ratio",
        type=float,
        default=0.30,
        help="Fraction of generated episodes that are ADAPT domain-transfer rows (0–1).",
    )
    parser.add_argument(
        "--adapt_focus",
        action="store_true",
        help="Include ADAPT rows in early curriculum stages and weight domain training.",
    )
    parser.add_argument(
        "--no_domain_stratify",
        action="store_true",
        help="Random domain task choice; default cycles all registered domain tasks.",
    )
    parser.add_argument(
        "--stage_epoch_scale",
        type=float,
        default=1.0,
        help="Multiply per-stage training epochs (use <1 for quick plots).",
    )
    parser.add_argument(
        "--adapt_eval_episodes",
        type=int,
        default=None,
        help="Episodes for ADAPT domain-pipeline eval (default: max(3, --eval_episodes)).",
    )
    args = parser.parse_args()

    # Allow CLI override of group size (useful for Colab memory tuning)
    if args.n_generations is not None:
        global N_GENERATIONS, BATCH_SIZE
        N_GENERATIONS = args.n_generations
        # Adjust batch size to stay divisible
        if BATCH_SIZE % N_GENERATIONS != 0:
            BATCH_SIZE = N_GENERATIONS

    global GRAD_ACCUM, MAX_NEW_TOKENS, LOGGING_STEPS, TEMPERATURE
    if args.batch_size is not None:
        BATCH_SIZE = max(1, args.batch_size)
    if args.grad_accum is not None:
        GRAD_ACCUM = max(1, args.grad_accum)
    if args.max_new_tokens is not None:
        MAX_NEW_TOKENS = max(32, args.max_new_tokens)
    if args.temperature is not None:
        TEMPERATURE = max(0.1, min(1.5, args.temperature))
    if args.logging_steps is not None:
        LOGGING_STEPS = max(1, args.logging_steps)

    if args.eval_only:
        evaluate(args.model, n_episodes=20, seed=args.seed)
    else:
        train(
            model_name=args.model,
            output_dir=args.output_dir,
            n_episodes=args.episodes,
            lora_rank=args.lora_rank,
            seed=args.seed,
            eval_episodes=max(3, args.eval_episodes),
            push_to_hub=args.push_to_hub,
            hub_model_id=args.hub_model_id,
            run_eval=not args.no_eval,
            domain_episode_ratio=float(args.domain_episode_ratio),
            adapt_focus=bool(args.adapt_focus),
            domain_stratify=not args.no_domain_stratify,
            stage_epoch_scale=float(args.stage_epoch_scale),
            adapt_eval_episodes=args.adapt_eval_episodes,
        )


if __name__ == "__main__":
    main()
