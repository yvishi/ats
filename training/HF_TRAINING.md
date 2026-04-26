# Train on Hugging Face (Spaces GPU)

This repo can run **SFT (optional) + GRPO** on a **private Hugging Face Space** with a GPU, then upload the LoRA folder to a **Hub model repo**. That keeps your laptop off and billing goes through HF credits.

## Prerequisites

1. **Hugging Face account** with **write token** (`Settings → Access tokens`, role *write*).
2. **Billing / credits** for the GPU tier you pick (Space **Settings → Hardware**). A **T4** is the cheapest; **7B + Unsloth** may need **A10G** or larger — if the job OOMs, raise hardware in the launcher (`--hardware`) or shrink `--model` / batch flags.
3. **Local Python** with `huggingface_hub` (to run the launcher only):

   ```bash
   pip install "huggingface_hub>=0.26"
   ```

## One-command deploy (recommended)

From your clone of this repository:

```bash
python training/hf_train.py \
  --hf_token "$HF_TOKEN" \
  --hf_user jdoe \
  --repo_url https://github.com/yvishi/ats.git \
  --repo_branch cleaned \
  --model Qwen/Qwen2.5-7B-Instruct \
  --hardware a10g-small \
  --episodes 80 \
  --run_sft \
  --adapt_focus
```

- **`--run_sft`**: builds jsonl, runs `train_sft.py`, then `train_grpo.py --sft_adapter /app/sft_adapter`.
- Omit **`--run_sft`** for GRPO-only (faster, less JSON stability).
- **`--no_eval`**: skip GRPO before/after eval on the Space (faster).
- **`--github_token`**: only if you use a **private** GitHub repo for `--repo_url`.

The script:

1. Creates/updates Space **`<your-username>/atc-grpo-runner`** (Docker SDK).
2. Creates/updates model repo **`<your-username>/atc-grpo-adapter`**.
3. Sets Space secret **`HF_TOKEN`** (and optional **`GITHUB_TOKEN`**).
4. Requests the GPU **`--hardware`** (if the API allows; otherwise set it in the Space UI).

## Watch the run

**There is no working page at** `.../spaces/user/name/logs` **(that path 404s).** Use one of these:

1. **Browser:** open the Space **home** only:  
   `https://huggingface.co/spaces/<your-username>/atc-grpo-runner`  
   Then use the **Logs** / **Build** UI on that page (wording varies; look for runtime/build output or the status badge).

2. **CLI** (install [HF CLI](https://huggingface.co/docs/huggingface_hub/guides/cli), `huggingface-cli login`):

   ```bash
   hf spaces logs <your-username>/atc-grpo-runner --build
   hf spaces logs <your-username>/atc-grpo-runner
   hf spaces logs <your-username>/atc-grpo-runner -f
   ```

### GPU usage shows 0%

Common causes:

- **Docker build still failing** — GPU is not used until the container **runs**. Check **Build** logs first.
- **Space asleep** — open the Space URL in a browser to wake it; **Factory reboot** after fixing errors.
- **Hardware not actually enabled** — Space **Settings → Hardware** must show a **GPU** tier (not CPU Basic). The launcher may fail to set hardware via API; set it manually.
- **Run finished or crashed** — container exits; GPU goes idle. Read **run** logs for tracebacks.

## Download the adapter

After training finishes:

```bash
huggingface-cli download <your-username>/atc-grpo-adapter --local-dir ./hf-adapter
```

Use that folder as `--trained` in `training/eval.py` or point inference at the Hub repo.

## Change code or hyperparameters

Re-run **`python training/hf_train.py ...`** with new flags; it **re-uploads** the Dockerfile and entrypoint. Then **restart** the Space (Factory reboot or push empty commit) so the new image runs.

## Hugging Face Jobs (CLI, alternative)

HF also offers **`hf jobs`** to run a command on a rented GPU (prepaid credits). See the official quickstart: [Hugging Face Jobs](https://huggingface.co/docs/hub/jobs-quickstart). Pattern:

1. Install the [HF CLI](https://huggingface.co/docs/huggingface_hub/guides/cli).
2. Put the same commands as in `entrypoint.sh` into your job script (clone repo → `train_grpo.py` / `train_sft.py`).
3. Pass **`HF_TOKEN`** as a job secret.

This repo’s **Spaces** path is the most integrated “one script” experience; **Jobs** are better if you want a single CLI `run` without maintaining a Space.

## Troubleshooting

| Issue | What to try |
|-------|-------------|
| **403** creating Space/model | Use your **real** `--hf_user` (profile URL slug), not the literal text `YOUR_HF_USERNAME`. Token must be **write** (classic) or fine-grained with **create repos** + **Spaces** for that account/org. |
| **README / “Startup duration too large”** | The launcher only sets `title` + `sdk: docker` (no `startup_duration_timeout` — avoids HF’s `6h` cap errors). Re-run the launcher after updating the repo, or on the Space click **Files and versions** → **README.md** and remove any `startup_duration_timeout: 8h` left from an old run. |
| **Docker build failed** on `unsloth` git / `colab-new` | Use the **latest** `hf_train.py`: image is `pytorch/pytorch:2.4.0-cuda12.1-…` + **unsloth from PyPI** (no git, no `colab-new`). Re-run the launcher. If the build log shows a **pip** error, paste the last 30 lines. |
| **`torch==2.10.0` vs `torchvision==0.19.0` / unsloth import error** | The launcher Dockerfile upgrades **torchvision** and **torchaudio** from `download.pytorch.org/whl/cu121` after unsloth so they match the newer **torch** unsloth pulls in. Re-run the launcher to refresh the Space Dockerfile. |
| **`/usr/bin/env: 'bash\r': No such file or directory` (exit 127)** | The Space’s `entrypoint.sh` had **Windows (CRLF)** line endings. The launcher now writes **LF** on all OSes. **Re-run** `python training/hf_train.py ...` to re-upload, then restart the Space. If you hand-edit `entrypoint.sh` on Windows, use **LF** in the editor. |
| Build timeout / killed | Unsloth + deps in one `RUN` can take 10–20+ minutes; if HF still kills the build, we can move installs to `entrypoint.sh` (slower per run, faster image build). |
| CUDA OOM | `--hardware a10g-small` or larger; lower `--batch_size`, `--max_new_tokens`, or use a smaller `--model`. |
| Clone fails | Public repo: leave `GITHUB_TOKEN` unset. Private: pass `--github_token` and ensure Space secret is set. |
| Empty adapter repo | Check **Logs** for tracebacks; confirm `HF_TOKEN` has **write** and `hf_push_outputs.py` ran. |
