"""Per-agent reward functions for GRPO training.

Design principles:
  1. Potential-based reward shaping (Ng et al. 1999) — policy-gradient safe,
     no convergence interference, dense signal from sparse environment.
  2. Group-relative advantage (GRPO) — advantage computed within each
     generation group, not against a learned value baseline (saves VRAM).
  3. Role decomposition — AMAN and DMAN rewards are independent signals
     logged separately so training curves show cooperation emergence.
  4. Theory-of-mind bonus — rewards pre-emptive coordination even when
     the environment did not force it (trains proactive behaviour).
  5. Supervisor alignment shaping — profile-weighted reward component
     trains the model to follow changing expert preferences (Snorkel AI).

Reward function signatures match TRL GRPOTrainer expectation:
    fn(completions: List[str], **kwargs) -> List[float]
where kwargs carries per-sample metadata packed into the dataset.
"""

from __future__ import annotations

import json
import os
import re
from typing import Any, Dict, List, Optional

try:
    from ..engine import simulate_plan
    from ..models import OperationType, PriorityClass, SlotAssignment, TaskDefinition
    from ..tasks import task_catalog
    from .dataset import parse_aman_action, parse_dman_action, parse_generator_action
    from ..multi_agent.environment import MultiAgentATCEnvironment
    from ..multi_agent.generator import ChallengeGenerator
    from ..multi_agent.models import (
        AgentRole,
        PerRoleMetrics,
        SupervisorProfileName,
        SUPERVISOR_PROFILES,
    )
    from ..multi_agent.supervisor import SupervisorAgent
except ImportError:
    import sys, os
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from engine import simulate_plan
    from models import OperationType, PriorityClass, SlotAssignment, TaskDefinition
    from tasks import task_catalog
    from training.dataset import parse_aman_action, parse_dman_action, parse_generator_action
    from multi_agent.environment import MultiAgentATCEnvironment
    from multi_agent.generator import ChallengeGenerator
    from multi_agent.models import (
        AgentRole,
        PerRoleMetrics,
        SupervisorProfileName,
        SUPERVISOR_PROFILES,
    )
    from multi_agent.supervisor import SupervisorAgent

_CATALOG = None
_SUPERVISOR = SupervisorAgent()
_TRACE_REWARDS = os.getenv("ATC_REWARD_TRACE", "").strip().lower() in {"1", "true", "yes", "on"}


def _debug_reward_trace(role: str, components: Dict[str, float]) -> None:
    if not _TRACE_REWARDS:
        return
    print(f"\n[REWARD TRACE] role={role}")
    for key, value in components.items():
        print(f"  {key}: {value:.4f}")


def _get_catalog() -> Dict[str, TaskDefinition]:
    global _CATALOG
    if _CATALOG is None:
        _CATALOG = task_catalog()
    return _CATALOG


def _normalized_cross_conflict_penalty(
    our_slots: List[SlotAssignment],
    other_slots: List[SlotAssignment],
    conflict_count: int,
    weight: float = 0.35,
) -> float:
    if conflict_count <= 0 or not our_slots or not other_slots:
        return 0.0

    our_by_runway: Dict[str, int] = {}
    other_by_runway: Dict[str, int] = {}

    for slot in our_slots:
        our_by_runway[slot.runway] = our_by_runway.get(slot.runway, 0) + 1
    for slot in other_slots:
        other_by_runway[slot.runway] = other_by_runway.get(slot.runway, 0) + 1

    max_cross_conflicts = 0
    for runway in set(our_by_runway) & set(other_by_runway):
        max_cross_conflicts += our_by_runway[runway] * other_by_runway[runway]

    if max_cross_conflicts <= 0:
        return 0.0

    normalized = min(1.0, conflict_count / max_cross_conflicts)
    return weight * normalized


# ── AMAN reward function ──────────────────────────────────────────────────────

def aman_reward_fn(
    completions: List[str],
    task_id: List[str],
    supervisor_profile: List[str],
    dman_slots_json: List[str],   # DMAN's slots from same episode (for cross-agent eval)
    atfm_deadlines_json: List[str],
    **kwargs: Any,
) -> List[float]:
    """GRPO reward function for AMAN (Arrival Manager) role.

    Called with a batch of K completions per prompt.
    Returns per-completion reward in [-1, 1].
    """
    catalog = _get_catalog()
    rewards: List[float] = []

    for completion, tid, profile_str, dman_json, _atfm_json in zip(
        completions, task_id, supervisor_profile, dman_slots_json, atfm_deadlines_json
    ):
        task = catalog.get(tid)
        if task is None:
            rewards.append(-1.0)
            continue

        aman_action = parse_aman_action(completion)
        if aman_action is None:
            rewards.append(-0.8)  # malformed JSON penalty
            continue

        dman_slots = _parse_slots_json(dman_json)
        profile = SupervisorProfileName(profile_str)

        # Merge AMAN + DMAN slots for full simulation
        merged = aman_action.arrival_slots + dman_slots
        outcome = simulate_plan(task, merged)

        # Per-role metrics (arrivals only)
        arrivals = [f for f in task.flights if f.operation == OperationType.ARRIVAL]
        arr_map  = {s.flight_id: s for s in aman_action.arrival_slots}

        delay_total = 0
        missing = 0
        emg_ok = emg_miss = 0

        for f in arrivals:
            slot = arr_map.get(f.flight_id)
            if slot:
                delay = abs(slot.assigned_minute - f.scheduled_minute)
                delay_total += delay
                is_emg = f.priority in (PriorityClass.EMERGENCY, PriorityClass.MEDICAL)
                if is_emg:
                    (emg_ok if delay <= 5 else emg_miss)
                    if delay <= 5:
                        emg_ok += 1
                    else:
                        emg_miss += 1
            else:
                missing += 1

        arr_count  = max(1, len(arrivals))
        budget     = task.delay_budget / 2.0
        delay_eff  = max(0.0, 1.0 - delay_total / max(1, budget))
        coverage   = 1.0 - missing / arr_count
        emg_score  = emg_ok / max(1, emg_ok + emg_miss) if (emg_ok + emg_miss) else 1.0

        # Cross-lane conflict penalty
        aman_runways = {s.runway for s in aman_action.arrival_slots}
        dman_runways = {s.runway for s in dman_slots}
        cross_penalty = 0.0
        if aman_runways & dman_runways:
            cross_outcome = simulate_plan(task, merged)
            cross_penalty = _normalized_cross_conflict_penalty(
                aman_action.arrival_slots,
                dman_slots,
                cross_outcome.metrics.conflict_count,
            )

        # Theory-of-mind: did AMAN yield for DMAN emergencies without being forced?
        tom_bonus = _compute_tom_bonus_aman(aman_action, dman_slots, task)

        # Supervisor alignment
        sup_align = _supervisor_alignment(outcome, task, profile)

        reward = (
            0.35 * delay_eff
            + 0.25 * emg_score
            + 0.20 * coverage
            + 0.10 * tom_bonus
            + 0.05 * sup_align
            - cross_penalty
        )
        reward = round(max(-1.0, min(1.0, reward)), 4)
        _debug_reward_trace(
            role="AMAN",
            components={
                "delay_eff": delay_eff,
                "emg_score": emg_score,
                "coverage": coverage,
                "tom_bonus": tom_bonus,
                "sup_align": sup_align,
                "cross_penalty": -cross_penalty,
                "final_reward": reward,
            },
        )
        rewards.append(reward)

    return rewards


# ── DMAN reward function ──────────────────────────────────────────────────────

def dman_reward_fn(
    completions: List[str],
    task_id: List[str],
    supervisor_profile: List[str],
    aman_slots_json: List[str],
    atfm_deadlines_json: List[str],
    **kwargs: Any,
) -> List[float]:
    """GRPO reward function for DMAN (Departure Manager) role."""
    catalog = _get_catalog()
    rewards: List[float] = []

    for completion, tid, profile_str, aman_json, atfm_json in zip(
        completions, task_id, supervisor_profile, aman_slots_json, atfm_deadlines_json
    ):
        task = catalog.get(tid)
        if task is None:
            rewards.append(-1.0)
            continue

        dman_action = parse_dman_action(completion)
        if dman_action is None:
            rewards.append(-0.8)
            continue

        aman_slots = _parse_slots_json(aman_json)
        atfm = json.loads(atfm_json) if atfm_json else {}
        profile = SupervisorProfileName(profile_str)

        merged  = aman_slots + dman_action.departure_slots
        outcome = simulate_plan(task, merged)

        departures = [f for f in task.flights if f.operation == OperationType.DEPARTURE]
        dep_map    = {s.flight_id: s for s in dman_action.departure_slots}

        delay_total = 0
        missing = 0
        emg_ok = emg_miss = 0
        atfm_ok = atfm_viol = 0

        for f in departures:
            slot = dep_map.get(f.flight_id)
            if slot:
                delay = abs(slot.assigned_minute - f.scheduled_minute)
                delay_total += delay
                is_emg = f.priority in (PriorityClass.EMERGENCY, PriorityClass.MEDICAL)
                if is_emg:
                    if delay <= 5:
                        emg_ok += 1
                    else:
                        emg_miss += 1
                # ATFM
                deadline = atfm.get(f.flight_id)
                if deadline is not None:
                    if slot.assigned_minute <= deadline:
                        atfm_ok += 1
                    else:
                        atfm_viol += 1
            else:
                missing += 1

        dep_count  = max(1, len(departures))
        budget     = task.delay_budget / 2.0
        delay_eff  = max(0.0, 1.0 - delay_total / max(1, budget))
        coverage   = 1.0 - missing / dep_count
        emg_score  = emg_ok / max(1, emg_ok + emg_miss) if (emg_ok + emg_miss) else 1.0
        atfm_score = atfm_ok / max(1, atfm_ok + atfm_viol) if (atfm_ok + atfm_viol) else 1.0

        cross_penalty = 0.0
        dman_runways = {s.runway for s in dman_action.departure_slots}
        aman_runways = {s.runway for s in aman_slots}
        if dman_runways & aman_runways:
            cross_outcome = simulate_plan(task, merged)
            cross_penalty = _normalized_cross_conflict_penalty(
                dman_action.departure_slots,
                aman_slots,
                cross_outcome.metrics.conflict_count,
            )

        tom_bonus = _compute_tom_bonus_dman(dman_action, aman_slots, task)
        sup_align = _supervisor_alignment(outcome, task, profile)

        reward = (
            0.30 * delay_eff
            + 0.20 * atfm_score
            + 0.20 * emg_score
            + 0.15 * coverage
            + 0.10 * tom_bonus
            + 0.05 * sup_align
            - cross_penalty
        )
        reward = round(max(-1.0, min(1.0, reward)), 4)
        _debug_reward_trace(
            role="DMAN",
            components={
                "delay_eff": delay_eff,
                "atfm_score": atfm_score,
                "emg_score": emg_score,
                "coverage": coverage,
                "tom_bonus": tom_bonus,
                "sup_align": sup_align,
                "cross_penalty": -cross_penalty,
                "final_reward": reward,
            },
        )
        rewards.append(reward)

    return rewards


# ── Generator reward function ─────────────────────────────────────────────────

def generator_reward_fn(
    completions: List[str],
    task_id: List[str],
    controller_scores: List[float],
    **kwargs: Any,
) -> List[float]:
    """GRPO reward function for GENERATOR role.

    Adversarial: generator wins when controllers fail.
    Penalised for producing unsolvable scenarios (reward hacking guard).
    """
    catalog = _get_catalog()
    rewards: List[float] = []

    for completion, tid, ctrl_score in zip(completions, task_id, controller_scores):
        task = catalog.get(tid)
        if task is None:
            rewards.append(-1.0)
            continue

        gen_action = parse_generator_action(completion)
        if gen_action is None:
            rewards.append(-0.5)
            continue

        generator = ChallengeGenerator()
        mutated_task, is_solvable = generator.mutate(task, gen_action)

        reward = generator.compute_reward(ctrl_score, is_solvable)
        rewards.append(reward)

    return rewards


# ── Supervisor reward function ────────────────────────────────────────────────

def supervisor_reward_fn(
    completions: List[str],
    task_id: List[str],
    supervisor_profile: List[str],
    merged_plan_json: List[str],
    **kwargs: Any,
) -> List[float]:
    """GRPO reward for SUPERVISOR role — measures preference alignment accuracy."""
    catalog = _get_catalog()
    rewards: List[float] = []

    for completion, tid, profile_str, plan_json in zip(
        completions, task_id, supervisor_profile, merged_plan_json
    ):
        task = catalog.get(tid)
        if task is None:
            rewards.append(-1.0)
            continue

        try:
            plan_data = json.loads(plan_json)
            slots = [SlotAssignment(**s) for s in plan_data]
        except Exception:
            rewards.append(-0.5)
            continue

        outcome = simulate_plan(task, slots)
        profile = SupervisorProfileName(profile_str)
        score   = _SUPERVISOR.score_plan(outcome, task, profile)

        # Parse supervisor's own score from completion
        supervisor_claimed = _extract_supervisor_score(completion)
        calibration_bonus  = 0.0
        if supervisor_claimed is not None:
            # Reward accurate self-assessment (calibration)
            calibration_bonus = max(0.0, 0.2 - abs(supervisor_claimed - score))

        rewards.append(round(min(1.0, score + calibration_bonus), 4))

    return rewards


# ── Helper utilities ──────────────────────────────────────────────────────────

def _parse_slots_json(json_str: str) -> List[SlotAssignment]:
    try:
        data = json.loads(json_str)
        return [SlotAssignment(**item) for item in data]
    except Exception:
        return []


def _compute_tom_bonus_aman(aman_action, dman_slots, task) -> float:
    """Theory-of-mind bonus: AMAN pre-emptively left gap for DMAN emergency."""
    if not dman_slots:
        return 0.0
    # Find DMAN emergency departures
    flights_by_id = {f.flight_id: f for f in task.flights}
    emg_deps = [
        s for s in dman_slots
        if flights_by_id.get(s.flight_id, None) and
        flights_by_id[s.flight_id].priority in (PriorityClass.EMERGENCY, PriorityClass.MEDICAL)
    ]
    if not emg_deps:
        return 0.0
    aman_runway_times = [(s.runway, s.assigned_minute) for s in aman_action.arrival_slots]
    for dep in emg_deps:
        # Check AMAN left ≥3 min gap around this dep slot on same runway
        gap_ok = all(
            abs(m - dep.assigned_minute) >= 3
            for rwy, m in aman_runway_times
            if rwy == dep.runway
        )
        if gap_ok:
            return 1.0
    return 0.1  # emergency existed but AMAN did not yield


def _compute_tom_bonus_dman(dman_action, aman_slots, task) -> float:
    """Theory-of-mind bonus: DMAN broadcast emergency AND cleared gap."""
    if not aman_slots:
        return 0.0
    flights_by_id = {f.flight_id: f for f in task.flights}
    emg_arrs = [
        s for s in aman_slots
        if flights_by_id.get(s.flight_id, None) and
        flights_by_id[s.flight_id].priority in (PriorityClass.EMERGENCY, PriorityClass.MEDICAL)
    ]
    if not emg_arrs:
        return 0.0
    dman_runway_times = [(s.runway, s.assigned_minute) for s in dman_action.departure_slots]
    for arr in emg_arrs:
        gap_ok = all(
            abs(m - arr.assigned_minute) >= 3
            for rwy, m in dman_runway_times
            if rwy == arr.runway
        )
        if gap_ok:
            return 1.0
    return 0.1


def _supervisor_alignment(outcome, task, profile: SupervisorProfileName) -> float:
    return _SUPERVISOR.score_plan(outcome, task, profile)


def _extract_supervisor_score(completion: str) -> Optional[float]:
    """Parse supervisor's claimed score from JSON completion."""
    try:
        data = json.loads(completion)
        return float(data.get("score", None))
    except Exception:
        match = re.search(r'"score"\s*:\s*([0-9.]+)', completion)
        if match:
            return float(match.group(1))
    return None
