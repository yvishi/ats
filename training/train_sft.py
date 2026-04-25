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
"""

from __future__ import annotations

import argparse
import inspect
import json
import os
import sys
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
        print(f"[ERROR] datasets/trl import failed: {e}")
        sys.exit(1)
    return FastLanguageModel, SFTConfig, SFTTrainer


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


def main() -> None:
    FastLanguageModel, SFTConfig, SFTTrainer = _require_training_deps()
    import torch
    from datasets import Dataset

    parser = argparse.ArgumentParser(description="SFT for ATC JSON agent outputs")
    parser.add_argument("--dataset", type=Path, required=True, help="JSONL from build_sft_dataset.py")
    parser.add_argument("--output_dir", type=Path, default=Path("outputs/atc-sft-json"))
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--max_seq_length", type=int, default=4096)
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
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    role_filter: Optional[str] = args.agent_role.strip() or None
    if role_filter and role_filter not in ("AMAN", "DMAN", "ADAPT"):
        raise SystemExit("--agent_role must be AMAN, DMAN, ADAPT, or empty")

    raw_rows = _load_jsonl_rows(args.dataset, role_filter)
    model_source = args.model if os.path.isdir(args.model) else args.model

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
        text = tokenizer.apply_chat_template(
            row["messages"],
            tokenize=False,
            add_generation_prompt=False,
        )
        text_rows.append({"text": text})

    train_ds = Dataset.from_list(text_rows)

    use_bf16 = torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8
    use_fp16 = torch.cuda.is_available() and not use_bf16

    cfg_kwargs: Dict[str, Any] = dict(
        output_dir=str(args.output_dir),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        logging_steps=10,
        save_steps=max(50, len(train_ds) // (args.batch_size * args.grad_accum * 2)),
        save_total_limit=3,
        seed=args.seed,
        bf16=use_bf16,
        fp16=use_fp16,
        report_to="none",
    )
    if _config_supports("max_seq_length", SFTConfig):
        cfg_kwargs["max_seq_length"] = args.max_seq_length
    if _config_supports("dataset_text_field", SFTConfig):
        cfg_kwargs["dataset_text_field"] = "text"
    if _config_supports("packing", SFTConfig):
        cfg_kwargs["packing"] = False

    sft_args = SFTConfig(**cfg_kwargs)

    trainer_kw: Dict[str, Any] = dict(
        model=model,
        args=sft_args,
        train_dataset=train_ds,
    )
    if _trainer_supports("processing_class", SFTTrainer):
        trainer_kw["processing_class"] = tokenizer
    elif _trainer_supports("tokenizer", SFTTrainer):
        trainer_kw["tokenizer"] = tokenizer
    else:
        raise SystemExit("TRL SFTTrainer API mismatch: no processing_class/tokenizer parameter")

    trainer = SFTTrainer(**trainer_kw)
    trainer.train()
    trainer.save_model(str(args.output_dir))
    tokenizer.save_pretrained(str(args.output_dir))

    meta = {
        "base_model": args.model,
        "dataset": str(args.dataset),
        "n_examples": len(train_ds),
        "agent_role_filter": role_filter,
        "lora_rank": args.lora_rank,
    }
    with open(args.output_dir / "sft_training_meta.json", "w", encoding="utf-8") as fh:
        json.dump(meta, fh, indent=2)
    print(f"[OK] SFT adapter saved to {args.output_dir}")


if __name__ == "__main__":
    main()
