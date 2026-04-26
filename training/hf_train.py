"""
Hugging Face Spaces (Docker) — GPU training launcher for ATC GRPO (+ optional SFT).

Creates or updates a private HF Space that clones this project, runs training on GPU,
then uploads the output folder (LoRA adapter, curves, metrics) to a Hub model repo.

Local usage:
  pip install "huggingface_hub>=0.20"
  python training/hf_train.py \\
      --hf_token hf_xxxx \\
      --hf_user your_username \\
      --model Qwen/Qwen2.5-7B-Instruct \\
      --episodes 80

Space URL:  https://huggingface.co/spaces/<user>/<space_name>
Output:     https://huggingface.co/<user>/<adapter_name>

See training/HF_TRAINING.md for step-by-step setup and HF Jobs (CLI) notes.
"""

from __future__ import annotations

import argparse
import shlex
import subprocess
import sys
import textwrap
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


PUSH_SCRIPT = textwrap.dedent("""\
    \"\"\"Push training outputs to HF Hub model repo.\"\"\"
    import argparse
    import os
    from pathlib import Path

    from huggingface_hub import HfApi

    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--repo_id", required=True)
    args = parser.parse_args()

    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    api = HfApi(token=token)

    api.create_repo(repo_id=args.repo_id, repo_type="model", exist_ok=True, private=True)
    api.upload_folder(
        folder_path=args.output_dir,
        repo_id=args.repo_id,
        repo_type="model",
        ignore_patterns=["*.bin.tmp", "*.safetensors.tmp", "__pycache__/*"],
    )
    print(f"Pushed outputs to https://huggingface.co/{args.repo_id}")
""")


def _dockerfile() -> str:
    return textwrap.dedent("""\
        FROM nvidia/cuda:12.1.0-cudnn8-runtime-ubuntu22.04

        ENV DEBIAN_FRONTEND=noninteractive
        ENV PYTHONUNBUFFERED=1
        ENV TORCH_COMPILE_DISABLE=1
        ENV TOKENIZERS_PARALLELISM=false
        ENV HF_HUB_ENABLE_HF_TRANSFER=1

        RUN apt-get update && apt-get install -y \\
            git curl python3.11 python3-pip \\
            && ln -sf /usr/bin/python3.11 /usr/bin/python \\
            && ln -sf /usr/bin/python3.11 /usr/bin/python3 \\
            && rm -rf /var/lib/apt/lists/*

        # Unsloth first (pulls compatible transformers stack).
        RUN pip install --no-cache-dir "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"
        RUN pip install --no-cache-dir \\
            "trl>=0.9.6" "datasets>=2.20.0" "accelerate>=0.32.0" "peft>=0.12.0" "bitsandbytes>=0.43.0" \\
            "huggingface_hub>=0.26" "hf_transfer" "matplotlib>=3.9.0" "numpy>=1.26.0"
        RUN pip install --no-cache-dir \\
            "openenv-core[core]>=0.2.3" "fastapi>=0.128.0" "openai>=2.30.0" "pydantic>=2.12.0" "uvicorn>=0.41.0"

        WORKDIR /app
        COPY hf_push_outputs.py /app/hf_push_outputs.py
        COPY entrypoint.sh /app/entrypoint.sh
        RUN chmod +x /app/entrypoint.sh
        CMD ["/app/entrypoint.sh"]
    """)


def _entrypoint(args: argparse.Namespace) -> str:
    """Shell script run inside the Space container (runtime clone + train + push)."""
    q = shlex.quote
    # github.com/org/repo.git (no scheme) for https://TOKEN@host/... clones
    clone_host = args.repo_url.split("://", 1)[-1].rstrip("/")

    lines = [
        "#!/usr/bin/env bash",
        "set -euo pipefail",
        'echo "=== Clone repository ==="',
        "if [ -n \"${GITHUB_TOKEN:-}\" ]; then",
        f'  CLONE_URL="https://${{GITHUB_TOKEN}}@{clone_host}"',
        "else",
        f"  CLONE_URL={q(args.repo_url)}",
        "fi",
        f'git clone --branch {q(args.repo_branch)} --depth 1 "$CLONE_URL" /app/repo',
        "cd /app/repo",
    ]

    if args.run_sft:
        lines += [
            'echo "=== SFT: build dataset ==="',
            "python training/build_sft_dataset.py \\",
            f"  --out /app/sft_data.jsonl \\",
            f"  --episodes {int(args.sft_episodes)} \\",
            f"  --seed {int(args.seed)}",
            'echo "=== SFT: train LoRA ==="',
            "python training/train_sft.py \\",
            "  --dataset /app/sft_data.jsonl \\",
            "  --output_dir /app/sft_adapter \\",
            f"  --model {q(args.model)} \\",
            f"  --epochs {float(args.sft_epochs)} \\",
            f"  --max_seq_length {int(args.sft_max_seq_length)} \\",
            f"  --batch_size {int(args.sft_batch_size)} \\",
            f"  --grad_accum {int(args.sft_grad_accum)} \\",
            f"  --lora_rank {int(args.lora_rank)} \\",
            f"  --seed {int(args.seed)} \\",
            "  --trainer hf",
        ]

    grpo_parts = [
        "python training/train_grpo.py",
        f"  --model {q(args.model)}",
        "  --output_dir /app/outputs",
        f"  --episodes {int(args.episodes)}",
        f"  --lora_rank {int(args.lora_rank)}",
        f"  --seed {int(args.seed)}",
        f"  --eval_episodes {int(args.eval_episodes)}",
        f"  --batch_size {int(args.batch_size)}",
        f"  --grad_accum {int(args.grad_accum)}",
        f"  --n_generations {int(args.n_generations)}",
        f"  --max_new_tokens {int(args.max_new_tokens)}",
        f"  --logging_steps {int(args.logging_steps)}",
        f"  --domain_episode_ratio {float(args.domain_episode_ratio)}",
        f"  --stage_epoch_scale {float(args.stage_epoch_scale)}",
        f"  --adapt_eval_episodes {int(args.adapt_eval_episodes)}",
    ]
    if args.adapt_focus:
        grpo_parts.append("  --adapt_focus")
    if args.no_eval:
        grpo_parts.append("  --no_eval")
    if args.run_sft:
        grpo_parts.append("  --sft_adapter /app/sft_adapter")

    lines += ['echo "=== GRPO ==="']
    for i, part in enumerate(grpo_parts):
        if i < len(grpo_parts) - 1:
            lines.append(part + " \\")
        else:
            lines.append(part)

    lines += [
        'echo "=== Push to Hub ==="',
        "python /app/hf_push_outputs.py \\",
        "  --output_dir /app/outputs \\",
        f"  --repo_id {q(args.hf_output_repo)}",
        'echo "=== Done ==="',
    ]
    return "\n".join(lines) + "\n"


def _run(cmd: list[str], **kw) -> None:
    print(f"$ {' '.join(cmd)}")
    subprocess.run(cmd, check=True, **kw)


def _hf_api(token: str):
    try:
        from huggingface_hub import HfApi
    except ImportError:
        print("[ERROR] pip install huggingface_hub")
        sys.exit(1)
    return HfApi(token=token)


def _ensure_repo(api, repo_id: str, repo_type: str) -> str:
    kwargs: dict = dict(repo_id=repo_id, repo_type=repo_type, exist_ok=True, private=True)
    if repo_type == "space":
        kwargs["space_sdk"] = "docker"
    url = api.create_repo(**kwargs)
    print(f"Repo ready: {url}")
    return url


def main() -> None:
    parser = argparse.ArgumentParser(description="Deploy ATC training to Hugging Face Spaces (Docker GPU)")
    parser.add_argument("--hf_token", required=True, help="HF token with write access (hub + spaces)")
    parser.add_argument("--hf_user", required=True, help="HF username or org")
    parser.add_argument("--repo_url", default="https://github.com/yvishi/ats.git")
    parser.add_argument("--repo_branch", default="cleaned")
    parser.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--episodes", type=int, default=80)
    parser.add_argument("--eval_episodes", type=int, default=4)
    parser.add_argument("--n_generations", type=int, default=4)
    parser.add_argument("--lora_rank", type=int, default=16)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--grad_accum", type=int, default=2)
    parser.add_argument("--max_new_tokens", type=int, default=256)
    parser.add_argument("--logging_steps", type=int, default=1)
    parser.add_argument("--domain_episode_ratio", type=float, default=0.45)
    parser.add_argument("--stage_epoch_scale", type=float, default=0.35)
    parser.add_argument("--adapt_eval_episodes", type=int, default=4)
    parser.add_argument("--adapt_focus", action="store_true")
    parser.add_argument("--no_eval", action="store_true", help="Skip GRPO before/after eval (faster)")
    parser.add_argument("--run_sft", action="store_true", help="Run SFT then GRPO with --sft_adapter")
    parser.add_argument("--sft_episodes", type=int, default=300)
    parser.add_argument("--sft_epochs", type=float, default=1.0)
    parser.add_argument("--sft_max_seq_length", type=int, default=2048)
    parser.add_argument("--sft_batch_size", type=int, default=1)
    parser.add_argument("--sft_grad_accum", type=int, default=8)
    parser.add_argument(
        "--hardware",
        default="t4-small",
        help="Space hardware: t4-small, a10g-small, a100-large, … (see HF Space settings)",
    )
    parser.add_argument("--space_name", default="atc-grpo-runner")
    parser.add_argument("--adapter_name", default="atc-grpo-adapter")
    parser.add_argument("--wandb_key", default="")
    parser.add_argument("--github_token", default="", help="GitHub PAT if the source repo is private")
    args = parser.parse_args()

    _hu = (args.hf_user or "").strip()
    _bad = {
        "your_hf_username",
        "yourusername",
        "your_username",
        "username",
        "hf_user",
        "myusername",
    }
    if not _hu or "/" in _hu or " " in _hu:
        print(
            "[ERROR] --hf_user must be your Hugging Face username or org id "
            "(from https://huggingface.co/<this-part>), with no spaces or slashes.",
            file=sys.stderr,
        )
        sys.exit(2)
    if _hu.upper() == "YOUR_HF_USERNAME" or _hu.lower() in _bad:
        print(
            "[ERROR] Replace the docs placeholder: pass your real HF username, e.g. "
            "--hf_user jdoe   (not YOUR_HF_USERNAME).",
            file=sys.stderr,
        )
        sys.exit(2)
    args.hf_user = _hu

    space_repo_id = f"{args.hf_user}/{args.space_name}"
    adapter_repo_id = f"{args.hf_user}/{args.adapter_name}"
    args.hf_output_repo = adapter_repo_id

    print(f"\n{'='*60}")
    print("  ATC training → Hugging Face Space (Docker)")
    print(f"  Space:   https://huggingface.co/spaces/{space_repo_id}")
    print(f"  Output:  https://huggingface.co/{adapter_repo_id}")
    print(f"  GPU:     {args.hardware}")
    print(f"  Model:   {args.model}")
    print(f"  GRPO ep: {args.episodes}  SFT: {'yes' if args.run_sft else 'no'}")
    print(f"{'='*60}\n")

    api = _hf_api(args.hf_token)

    print("[1/4] Ensure Space repo exists...")
    _ensure_repo(api, space_repo_id, "space")
    _ensure_repo(api, adapter_repo_id, "model")

    print("[2/4] Upload Dockerfile, entrypoint, push helper...")
    with tempfile.TemporaryDirectory() as tmp:
        tmp_p = Path(tmp)
        (tmp_p / "Dockerfile").write_text(_dockerfile(), encoding="utf-8")
        (tmp_p / "entrypoint.sh").write_text(_entrypoint(args), encoding="utf-8")
        (tmp_p / "hf_push_outputs.py").write_text(PUSH_SCRIPT, encoding="utf-8")
        readme = textwrap.dedent(f"""\
            ---
            title: ATC GRPO Runner
            sdk: docker
            hardware: {args.hardware}
            startup_duration_timeout: 6h
            ---

            # ATC training on GPU

            Clones **`{args.repo_url}`** (`{args.repo_branch}`), runs **{"SFT + " if args.run_sft else ""}GRPO**,
            uploads the LoRA folder to **`{adapter_repo_id}`**.

            **Secrets:** `HF_TOKEN` (required). `GITHUB_TOKEN` only if the Git repo is private.

            Logs: Space **Logs** tab. Long runs: hardware tab may need a larger GPU tier.
        """)
        (tmp_p / "README.md").write_text(readme, encoding="utf-8")

        api.upload_folder(folder_path=str(tmp_p), repo_id=space_repo_id, repo_type="space")

    print(f"[3/4] Request hardware {args.hardware}...")
    try:
        api.request_space_hardware(repo_id=space_repo_id, hardware=args.hardware)
        print(f"  Hardware set to {args.hardware}")
    except Exception as e:
        print(f"  [WARN] Could not set hardware via API: {e}")
        print(f"  Set manually: https://huggingface.co/spaces/{space_repo_id}/settings")

    print("[4/4] Space secrets...")
    try:
        api.add_space_secret(repo_id=space_repo_id, key="HF_TOKEN", value=args.hf_token)
        secrets = ["HF_TOKEN"]
        if args.github_token:
            api.add_space_secret(repo_id=space_repo_id, key="GITHUB_TOKEN", value=args.github_token)
            secrets.append("GITHUB_TOKEN")
        if args.wandb_key:
            api.add_space_secret(repo_id=space_repo_id, key="WANDB_API_KEY", value=args.wandb_key)
            secrets.append("WANDB_API_KEY")
        print(f"  Secrets set: {', '.join(secrets)}")
    except Exception as e:
        print(f"  [WARN] Could not set secrets via API: {e}")
        print("  Add HF_TOKEN (and GITHUB_TOKEN if needed) under Space Settings → Variables and secrets.")

    print("\nDone. Open your Space (there is no /logs URL — use the Space page + Logs UI).")
    print(f"  Space home: https://huggingface.co/spaces/{space_repo_id}")
    print("  In the browser: open that page → **Logs** / **Build** (or the status indicator).")
    print("  From terminal (HF CLI):  hf spaces logs " + space_repo_id + " --build")
    print("                           hf spaces logs " + space_repo_id)
    print("\nDownload adapter after run:")
    print(f"  huggingface-cli download {adapter_repo_id} --local-dir ./hf-adapter")


if __name__ == "__main__":
    main()
