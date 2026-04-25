"""Build supervised fine-tuning (SFT) JSONL for AMAN / DMAN / ADAPT JSON I/O.

Teacher labels come from the same deterministic heuristics used in inference
(``multi_agent.inference``), so completions match ``parse_*`` expectations.

Usage:
  python training/build_sft_dataset.py --out data/atc_sft.jsonl --episodes 400

Then run SFT:
  python training/train_sft.py --dataset data/atc_sft.jsonl --output_dir outputs/sft-lora
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from multi_agent.environment import MultiAgentATCEnvironment
from multi_agent.generator import ChallengeGenerator
from multi_agent.models import AgentRole, SupervisorProfileName, SUPERVISOR_PROFILES
from multi_agent.supervisor import SupervisorAgent
from tasks import MAX_TASK_TIER, task_catalog, tasks_up_to_tier

from training.dataset import (
    ADAPT_SYSTEM,
    AMAN_SYSTEM,
    DMAN_SYSTEM,
    _heuristic_aman_action,
    _heuristic_dman_action,
)
from training.sft_schema import adapt_action_to_json_str, aman_action_to_json_str, dman_action_to_json_str


def _chat_messages(system: str, user: str, assistant_json: str) -> List[Dict[str, str]]:
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
        {"role": "assistant", "content": assistant_json},
    ]


def build_sft_rows(
    n_episodes: int,
    seed: int,
    *,
    include_negotiate: bool = True,
    include_adapt: bool = True,
    domain_episode_ratio: float = 0.12,
) -> List[Dict[str, Any]]:
    """Return SFT rows with ``messages`` + metadata (each row = one training example)."""
    from multi_agent.inference import _build_aman_heuristic, _build_dman_heuristic

    rng = random.Random(seed)
    supervisor = SupervisorAgent()
    env = MultiAgentATCEnvironment(seed=seed)
    generator = ChallengeGenerator(seed=seed)
    rows: List[Dict[str, Any]] = []

    domain_tasks: Dict[str, Any] = {}
    build_adapt_observation = None
    _build_adapt_heuristic = None
    if include_adapt:
        try:
            from domains import get_all_domain_tasks
            from multi_agent.adapt import _build_adapt_heuristic as _bah
            from multi_agent.adapt import build_adapt_observation as _bao

            domain_tasks = get_all_domain_tasks()
            build_adapt_observation = _bao
            _build_adapt_heuristic = _bah
        except Exception:
            domain_tasks = {}

    for ep_id in range(n_episodes):
        if (
            include_adapt
            and domain_tasks
            and build_adapt_observation is not None
            and _build_adapt_heuristic is not None
            and rng.random() < domain_episode_ratio
        ):
            tid = rng.choice(list(domain_tasks.keys()))
            dtask = domain_tasks[tid]
            profile = supervisor.sample_profile(ep_id)
            sup_desc = SUPERVISOR_PROFILES[profile]["description"]
            obs = build_adapt_observation(dtask, profile)
            teacher = _build_adapt_heuristic(obs, dtask)
            assistant = adapt_action_to_json_str(teacher)
            system = ADAPT_SYSTEM + f"\n\nSUPERVISOR CONTEXT: {sup_desc}"
            user = obs.to_prompt_text()
            rows.append({
                "messages": _chat_messages(system, user, assistant),
                "task_id": tid,
                "agent_role": AgentRole.ADAPT.value,
                "episode_id": ep_id,
                "round": "adapt",
            })
            continue

        progress = ep_id / max(1, n_episodes - 1)
        max_tier = max(0, min(MAX_TASK_TIER, int(math.sqrt(progress) * MAX_TASK_TIER)))
        if generator.is_in_rescue_mode():
            max_tier = 0
        tier_pool = tasks_up_to_tier(max_tier)
        base_task = rng.choice(tier_pool)
        generator._difficulty_level = max(0, max_tier)
        profile = supervisor.sample_profile(ep_id)
        sup_desc = SUPERVISOR_PROFILES[profile]["description"]
        mutated_task, _ = generator.mutate(base_task)

        aman_obs, dman_obs = env.reset(
            episode_id=ep_id,
            supervisor_profile=profile,
            mutated_task=mutated_task,
        )
        atfm = env._state.atfm_deadlines

        aman_teacher = _build_aman_heuristic(aman_obs)
        dman_teacher = _build_dman_heuristic(dman_obs, atfm)

        aman_sys = AMAN_SYSTEM + f"\n\nSUPERVISOR TODAY: {sup_desc}"
        dman_sys = DMAN_SYSTEM + f"\n\nSUPERVISOR TODAY: {sup_desc}"

        rows.append({
            "messages": _chat_messages(
                aman_sys,
                aman_obs.to_prompt_text(),
                aman_action_to_json_str(aman_teacher),
            ),
            "task_id": mutated_task.task_id,
            "agent_role": AgentRole.AMAN.value,
            "episode_id": ep_id,
            "round": "bid",
            "curriculum_tier": max_tier,
        })
        rows.append({
            "messages": _chat_messages(
                dman_sys,
                dman_obs.to_prompt_text(),
                dman_action_to_json_str(dman_teacher),
            ),
            "task_id": mutated_task.task_id,
            "agent_role": AgentRole.DMAN.value,
            "episode_id": ep_id,
            "round": "bid",
            "curriculum_tier": max_tier,
        })

        if include_negotiate:
            try:
                h_aman = _heuristic_aman_action(mutated_task)
                h_dman = _heuristic_dman_action(mutated_task)
                neg_aman_obs, neg_dman_obs, _, bid_done = env.step_bid(h_aman, h_dman)
                if not bid_done:
                    neg_aman = _build_aman_heuristic(neg_aman_obs)
                    neg_dman = _build_dman_heuristic(neg_dman_obs, atfm)
                    rows.append({
                        "messages": _chat_messages(
                            aman_sys,
                            neg_aman_obs.to_prompt_text(),
                            aman_action_to_json_str(neg_aman),
                        ),
                        "task_id": mutated_task.task_id,
                        "agent_role": AgentRole.AMAN.value,
                        "episode_id": ep_id,
                        "round": "negotiate",
                        "curriculum_tier": max_tier,
                    })
                    rows.append({
                        "messages": _chat_messages(
                            dman_sys,
                            neg_dman_obs.to_prompt_text(),
                            dman_action_to_json_str(neg_dman),
                        ),
                        "task_id": mutated_task.task_id,
                        "agent_role": AgentRole.DMAN.value,
                        "episode_id": ep_id,
                        "round": "negotiate",
                        "curriculum_tier": max_tier,
                    })
            except Exception:
                pass

    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Build JSONL for JSON-format SFT")
    parser.add_argument("--out", type=Path, required=True, help="Output .jsonl path")
    parser.add_argument("--episodes", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no_negotiate", action="store_true", help="Omit negotiate-round rows")
    parser.add_argument("--no_adapt", action="store_true", help="Omit ADAPT domain rows")
    parser.add_argument("--domain_ratio", type=float, default=0.12, help="Fraction of ADAPT episodes")
    args = parser.parse_args()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    rows = build_sft_rows(
        args.episodes,
        args.seed,
        include_negotiate=not args.no_negotiate,
        include_adapt=not args.no_adapt,
        domain_episode_ratio=args.domain_ratio,
    )
    with open(args.out, "w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"[OK] Wrote {len(rows)} SFT rows to {args.out}")


if __name__ == "__main__":
    main()
