# Branch Policy

This repository uses a two-track branch model:

- `release/hackathon-standard`: only hackathon-compliant code and docs. This is the only branch allowed to flow toward `main`.
- `testing/groq-experiments`: experimentation branch for Groq-only tests/docs/scripts.

## Merge Rules

1. Never merge `testing/groq-experiments` directly into `main`.
2. Only cherry-pick non-Groq-safe commits from testing into release.
3. Release branch must pass `scripts/check_release_hygiene.sh` before submission.
4. Release branch must pass:
   - `python -m openenv.cli validate .`
   - `pytest -q`
   - `python scripts/run_graders.py`

## Guardrail

`scripts/check_release_hygiene.sh` enforces that release-only files contain no `GROQ_` or `groq` markers and that Groq-only artifacts are absent.
