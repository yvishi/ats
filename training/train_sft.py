"""Supervised fine-tuning (SFT) for JSON-formatted AMAN / DMAN / ADAPT outputs.

Cold-starts the instruction model on strict JSON completions before GRPO so the
policy reliably parses tasks and matches ``parse_aman_action`` /
``parse_dman_action`` / ``parse_adapt_action``.

Prerequisites (same stack as ``train_grpo.py``):
  pip install torch unsloth unsloth-zoo trl transformers datasets accelerate peft bitsandbytes

1) Build data:
     python training/build_sft_dataset.py --out data/atc_sft.jsonl --episodes 500

2) Train LoRA SFT:
     python training/train_sft.py --dataset data/atc_sft.jsonl --output_dir outputs/sft-json

3) Point GRPO / inference at the saved adapter directory or merge to a HF model repo.

TRL versions differ: ``SFTTrainer`` may expect ``max_seq_length`` / ``dataset_text_field``
on the trainer, on ``SFTConfig``, or only under newer names (e.g. ``max_length``). This
script filters kwargs per installed signatures and falls back to ``transformers.Trainer``
if ``SFTTrainer`` still fails (see stderr / traceback).
"""

from __future__ import annotations

import argparse
import inspect
import json
import os
import sys
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _require_training_deps():
    if sys.version_info >= (3, 14):
        print("[ERROR] Python 3.14 not supported. Use 3.11 or 3.12.")
        sys.exit(1)
    try:
        import torch  # noqa: F401
    except ImportError as e:
        print(f"[ERROR] torch missing: {e}")
        sys.exit(1)
    try:
        import unsloth  # noqa: F401
        from unsloth import FastLanguageModel
    except Exception as e:
        print(f"[ERROR] unsloth import failed: {e}")
        print("Install: pip install unsloth unsloth-zoo")
        sys.exit(1)
    try:
        from trl import SFTConfig, SFTTrainer
    except ImportError as e:
        print(f"[ERROR] trl import failed: {e}")
        sys.exit(1)
    return FastLanguageModel, SFTConfig, SFTTrainer


def _sig_keys(cls) -> set:
    try:
        return set(inspect.signature(cls.__init__).parameters.keys()) - {"self"}
    except Exception:
        return set()


def _pick(d: Dict[str, Any], keys: set) -> Dict[str, Any]:
    return {k: v for k, v in d.items() if k in keys}


def _load_jsonl_rows(path: Path, agent_role: Optional[str]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if agent_role and row.get("agent_role") != agent_role:
                continue
            rows.append(row)
    if not rows:
        raise SystemExit(f"No rows loaded from {path} (filter={agent_role!r})")
    return rows


def _train_with_hf_trainer(
    model,
    tokenizer,
    train_text_ds,
    output_dir: Path,
    *,
    epochs: float,
    batch_size: int,
    grad_accum: int,
    lr: float,
    seed: int,
    max_seq_length: int,
    use_bf16: bool,
    use_fp16: bool,
) -> None:
    """HuggingFace ``Trainer`` + causal LM collator — stable across TRL versions (recommended for Colab)."""
    from transformers import (
        DataCollatorForLanguageModeling,
        Trainer,
        TrainerCallback,
        TrainingArguments,
    )

    class _EpochReportCallback(TrainerCallback):
        """Print a clear summary at the end of each training epoch (HF Trainer)."""

        def __init__(self, label: str = "SFT") -> None:
            self._label = label

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
                f"  [{self._label}] EPOCH END  epoch={ep:.2f}  global_step={gs}\n"
                f"  last_log: loss={loss_s}  lr={lr_s}\n"
                f"{'='*60}\n",
                flush=True,
            )

    if getattr(tokenizer, "pad_token", None) is None and getattr(tokenizer, "eos_token", None) is not None:
        tokenizer.pad_token = tokenizer.eos_token

    cfg = getattr(model, "config", None)
    if cfg is not None:
        cfg.use_cache = False
    if hasattr(model, "enable_input_require_grads"):
        try:
            model.enable_input_require_grads()
        except Exception:
            pass

    def tok(batch: Dict[str, List[str]]) -> Dict[str, Any]:
        return tokenizer(
            batch["text"],
            truncation=True,
            max_length=max_seq_length,
            padding=False,
        )

    tok_ds = train_text_ds.map(tok, batched=True, remove_columns=["text"])

    steps_per_epoch = max(1, len(tok_ds) // max(1, batch_size * grad_accum))
    save_steps = max(1, min(200, steps_per_epoch))

    ta_keys = _sig_keys(TrainingArguments)
    ta_kw = dict(
        output_dir=str(output_dir),
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=grad_accum,
        learning_rate=lr,
        logging_steps=max(1, min(10, steps_per_epoch)),
        save_steps=save_steps,
        save_total_limit=2,
        seed=seed,
        bf16=use_bf16,
        fp16=use_fp16,
        report_to="none",
        remove_unused_columns=False,
    )
    if "gradient_checkpointing" in ta_keys:
        ta_kw["gradient_checkpointing"] = True
    if "logging_strategy" in ta_keys:
        ta_kw["logging_strategy"] = "epoch"
    if "logging_first_step" in ta_keys:
        ta_kw["logging_first_step"] = True
    if "disable_tqdm" in ta_keys:
        ta_kw["disable_tqdm"] = False
    args = TrainingArguments(**_pick(ta_kw, ta_keys))
    collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=tok_ds,
        data_collator=collator,
        callbacks=[_EpochReportCallback("SFT")],
    )
    trainer.train()
    trainer.save_model(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))


def _try_sft_trainer(
    SFTConfig,
    SFTTrainer,
    *,
    model,
    tokenizer,
    train_ds,
    output_dir: Path,
    max_seq_length: int,
    epochs: float,
    batch_size: int,
    grad_accum: int,
    lr: float,
    seed: int,
    use_bf16: bool,
    use_fp16: bool,
) -> Tuple[bool, Optional[str]]:
    """Return (ok, error_message)."""
    cfg_keys = _sig_keys(SFTConfig)
    tr_keys = _sig_keys(SFTTrainer)

    n = len(train_ds)
    steps_per_epoch = max(1, n // max(1, batch_size * grad_accum))
    save_steps = max(1, min(200, steps_per_epoch))

    base_training = dict(
        output_dir=str(output_dir),
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=grad_accum,
        learning_rate=lr,
        logging_steps=max(1, min(10, steps_per_epoch)),
        save_steps=save_steps,
        save_total_limit=3,
        seed=seed,
        bf16=use_bf16,
        fp16=use_fp16,
        report_to="none",
    )
    if "remove_unused_columns" in cfg_keys:
        base_training["remove_unused_columns"] = False

    # TRL renamed max_seq_length -> max_length on some releases
    sft_specific: Dict[str, Any] = {}
    if "max_seq_length" in cfg_keys:
        sft_specific["max_seq_length"] = max_seq_length
    elif "max_length" in cfg_keys:
        sft_specific["max_length"] = max_seq_length

    if "dataset_text_field" in cfg_keys:
        sft_specific["dataset_text_field"] = "text"
    if "packing" in cfg_keys:
        sft_specific["packing"] = False

    merged = {**base_training, **sft_specific}
    cfg_kw = _pick(merged, cfg_keys)

    try:
        sft_args = SFTConfig(**cfg_kw)
    except TypeError as e:
        return False, f"SFTConfig construction failed: {e}\nKeys tried: {sorted(cfg_kw.keys())}"

    t_kw: Dict[str, Any] = dict(model=model, args=sft_args, train_dataset=train_ds)
    if "processing_class" in tr_keys:
        t_kw["processing_class"] = tokenizer
    elif "tokenizer" in tr_keys:
        t_kw["tokenizer"] = tokenizer
    else:
        return False, "SFTTrainer has neither processing_class nor tokenizer parameter"

    # Some TRL builds take these on the trainer, not the config
    if "max_seq_length" in tr_keys:
        t_kw["max_seq_length"] = max_seq_length
    elif "max_length" in tr_keys:
        t_kw["max_length"] = max_seq_length
    if "dataset_text_field" in tr_keys:
        t_kw["dataset_text_field"] = "text"
    if "packing" in tr_keys:
        t_kw["packing"] = False

    try:
        trainer = SFTTrainer(**_pick(t_kw, tr_keys))
        trainer.train()
        trainer.save_model(str(output_dir))
        tokenizer.save_pretrained(str(output_dir))
        return True, None
    except Exception as e:
        return False, f"{type(e).__name__}: {e}\n{traceback.format_exc()}"


def main() -> None:
    FastLanguageModel, SFTConfig, SFTTrainer = _require_training_deps()
    import torch
    from datasets import Dataset

    parser = argparse.ArgumentParser(description="SFT for ATC JSON agent outputs")
    parser.add_argument("--dataset", type=Path, required=True, help="JSONL from build_sft_dataset.py")
    parser.add_argument("--output_dir", type=Path, default=Path("outputs/atc-sft-json"))
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--max_seq_length", type=int, default=2048, help="Lower on T4 if OOM (e.g. 1024)")
    parser.add_argument("--epochs", type=float, default=1.0)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--grad_accum", type=int, default=4)
    parser.add_argument("--lora_rank", type=int, default=16)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--agent_role",
        type=str,
        default="",
        help="If set, only rows with this agent_role (AMAN, DMAN, or ADAPT)",
    )
    parser.add_argument(
        "--no_hf_fallback",
        action="store_true",
        help="Do not fall back to transformers.Trainer if SFTTrainer fails (only for --trainer auto)",
    )
    parser.add_argument(
        "--trainer",
        choices=("auto", "trl", "hf"),
        default="hf",
        help=(
            "hf = HuggingFace Trainer only (default; most reliable on Colab). "
            "trl = TRL SFTTrainer only. auto = try trl then hf on failure."
        ),
    )
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    role_filter: Optional[str] = args.agent_role.strip() or None
    if role_filter and role_filter not in ("AMAN", "DMAN", "ADAPT"):
        raise SystemExit("--agent_role must be AMAN, DMAN, ADAPT, or empty")

    raw_rows = _load_jsonl_rows(args.dataset, role_filter)
    model_source = args.model if os.path.isdir(args.model) else args.model

    print(f"[SFT] rows={len(raw_rows)}  model={model_source}  max_seq_length={args.max_seq_length}")

    try:
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=model_source,
            max_seq_length=args.max_seq_length,
            load_in_4bit=True,
            dtype=None,
        )

        LORA_ALPHA = args.lora_rank * 2
        LORA_TARGETS = ["q_proj", "v_proj", "k_proj", "o_proj"]
        model = FastLanguageModel.get_peft_model(
            model,
            r=args.lora_rank,
            lora_alpha=LORA_ALPHA,
            target_modules=LORA_TARGETS,
            lora_dropout=0.0,
            bias="none",
            use_gradient_checkpointing="unsloth",
            random_state=args.seed,
        )

        text_rows: List[Dict[str, str]] = []
        for row in raw_rows:
            if "messages" not in row:
                raise ValueError(f"Row missing 'messages': keys={list(row.keys())}")
            text = tokenizer.apply_chat_template(
                row["messages"],
                tokenize=False,
                add_generation_prompt=False,
            )
            text_rows.append({"text": text})

        train_ds = Dataset.from_list(text_rows)

        use_bf16 = torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8
        use_fp16 = torch.cuda.is_available() and not use_bf16

        ok_trl = False
        used_hf = False
        trainer_backend = "unknown"

        if args.trainer == "hf":
            print("[SFT] Using HuggingFace Trainer backend (--trainer hf).")
            _train_with_hf_trainer(
                model,
                tokenizer,
                train_ds,
                args.output_dir,
                epochs=args.epochs,
                batch_size=args.batch_size,
                grad_accum=args.grad_accum,
                lr=args.lr,
                seed=args.seed,
                max_seq_length=args.max_seq_length,
                use_bf16=use_bf16,
                use_fp16=use_fp16,
            )
            used_hf = True
            trainer_backend = "hf"
        elif args.trainer == "trl":
            ok_trl, err = _try_sft_trainer(
                SFTConfig,
                SFTTrainer,
                model=model,
                tokenizer=tokenizer,
                train_ds=train_ds,
                output_dir=args.output_dir,
                max_seq_length=args.max_seq_length,
                epochs=args.epochs,
                batch_size=args.batch_size,
                grad_accum=args.grad_accum,
                lr=args.lr,
                seed=args.seed,
                use_bf16=use_bf16,
                use_fp16=use_fp16,
            )
            if not ok_trl:
                print(err or "unknown TRL error", file=sys.stderr)
                raise SystemExit(1)
            trainer_backend = "trl"
        else:
            ok_trl, err = _try_sft_trainer(
                SFTConfig,
                SFTTrainer,
                model=model,
                tokenizer=tokenizer,
                train_ds=train_ds,
                output_dir=args.output_dir,
                max_seq_length=args.max_seq_length,
                epochs=args.epochs,
                batch_size=args.batch_size,
                grad_accum=args.grad_accum,
                lr=args.lr,
                seed=args.seed,
                use_bf16=use_bf16,
                use_fp16=use_fp16,
            )
            if ok_trl:
                trainer_backend = "trl"
            else:
                print("[WARN] SFTTrainer path failed:\n", err, file=sys.stderr)
                if args.no_hf_fallback:
                    raise SystemExit(1)
                print("[INFO] Falling back to transformers.Trainer …")
                _train_with_hf_trainer(
                    model,
                    tokenizer,
                    train_ds,
                    args.output_dir,
                    epochs=args.epochs,
                    batch_size=args.batch_size,
                    grad_accum=args.grad_accum,
                    lr=args.lr,
                    seed=args.seed,
                    max_seq_length=args.max_seq_length,
                    use_bf16=use_bf16,
                    use_fp16=use_fp16,
                )
                used_hf = True
                trainer_backend = "hf"

        meta = {
            "base_model": args.model,
            "dataset": str(args.dataset),
            "n_examples": len(train_ds),
            "agent_role_filter": role_filter,
            "lora_rank": args.lora_rank,
            "max_seq_length": args.max_seq_length,
            "trainer": args.trainer,
            "trainer_backend": trainer_backend,
            "used_sft_trainer": ok_trl,
            "used_hf_trainer_fallback": args.trainer == "auto" and used_hf and not ok_trl,
            "used_hf_trainer": args.trainer == "hf" or used_hf,
        }
        with open(args.output_dir / "sft_training_meta.json", "w", encoding="utf-8") as fh:
            json.dump(meta, fh, indent=2)
        print(f"[OK] SFT adapter saved to {args.output_dir}")

    except Exception:
        print("[FATAL] SFT run crashed:", file=sys.stderr)
        traceback.print_exc()
        raise


if __name__ == "__main__":
    main()
