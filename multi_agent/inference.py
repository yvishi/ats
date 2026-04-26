"""Multi-agent ATC inference runner.

Runs AMAN + DMAN agents against the ATC environment, either with:
  - heuristic-baseline: deterministic planner (no LLM required)
  - any HF/OpenAI-compatible model: LLM-driven AMAN and DMAN agents

Output format mirrors inference.py structured logs:
  [START] task=... env=... model=...
  [STEP]  step=... role=AMAN|DMAN reward=... conflicts=... done=...
    (conflicts = wake/separation issue count, not raw diagnostic list length)
  [NEG]   round=... conflicts_resolved=... messages=...
  [END]   task=... success=... composite=... aman=... dman=... coord=...

Usage:
  python multi_agent/inference.py                           # heuristic demo
  python multi_agent/inference.py --model Qwen/Qwen2.5-7B  # LLM agents
  python multi_agent/inference.py --task bengaluru_irrops_hard --episodes 3
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from engine import simulate_plan
from graders import grade_multi_agent
from models import (
    FlightRecord,
    OperationType,
    PriorityClass,
    SlotAssignment,
    TaskDefinition,
)
from planner import build_heuristic_plan, build_refined_plan, _flight_sort_key
from tasks import task_catalog, ordered_tasks
from multi_agent.environment import (
    DOMAIN_MAX_NEGOTIATE_ROUNDS,
    MAX_NEGOTIATE_ROUNDS,
    MultiAgentATCEnvironment,
    count_separation_issues,
)
from multi_agent.generator import ChallengeGenerator
from multi_agent.models import (
    AMANAction,
    AgentRole,
    DMANAction,
    NegotiationMessage,
    MessageType,
    MultiAgentObservation,
    SupervisorProfileName,
    SUPERVISOR_PROFILES,
)
from multi_agent.supervisor import SupervisorAgent
from multi_agent.visual_events import (
    VisualSink,
    emit,
    serialize_action_layout,
    serialize_task_snapshot,
    terminal_event,
)
from training.dataset import (
    AMAN_SYSTEM,
    DMAN_SYSTEM,
    parse_aman_action,
    parse_dman_action,
)


# ── Config ────────────────────────────────────────────────────────────────────

API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1").rstrip("/")
MODEL_NAME   = os.getenv("MODEL_NAME", "heuristic-baseline")
HF_TOKEN     = os.getenv("HF_TOKEN", "").strip()
BENCHMARK    = "atc_multiagent_openenv"
MAX_TOKENS   = 1024
TEMPERATURE  = 0.3
SUCCESS_THRESHOLD = 0.60

# N appended to AMAN/DMAN system prompts during ADAPT → ATC runs (bid + 1st negotiate LLM pass).
# Focuses the model on clearing separation conflicts the heuristic pipeline often leaves on
# tight remapped windows (e.g. ICU/ADAPT), without naming the source domain.
_DOMAIN_TRANSFER_LLM_HINT = (
    "STRUCTURAL-TRANSFER SCHEDULE: windows may be tight after cross-domain remapping. "
    "In NEGOTIATE, use the CONFLICT LOG: shift arrival/departure minutes on shared runways to "
    "satisfy minimum separation, staggering arrivals vs departures when the log says spacing "
    "or wake conflicts. Prefer the smallest time moves that clear diagnostics; do not miss "
    "ATFM/departure network deadlines (DMAN)."
)


# ── Structured logging ────────────────────────────────────────────────────────

def _p(msg: str) -> None:
    try:
        print(msg, flush=True)
    except (BrokenPipeError, OSError):
        pass


def log_start(task: str, model: str) -> None:
    _p(f"[START] task={task} env={BENCHMARK} model={model}")


def log_step(step: int, role: str, reward: float, conflicts: int, done: bool) -> None:
    _p(f"[STEP]  step={step} role={role} reward={reward:.3f} conflicts={conflicts} done={str(done).lower()}")


def log_neg(rnd: int, resolved: int, messages: int) -> None:
    _p(f"[NEG]   round={rnd} conflicts_resolved={resolved} messages={messages}")


def log_end(
    task: str, success: bool, composite: float,
    aman: float, dman: float, coord: float, gen_diff: int,
) -> None:
    _p(
        f"[END]   task={task} success={str(success).lower()} "
        f"composite={composite:.3f} aman={aman:.3f} dman={dman:.3f} "
        f"coord={coord:.3f} gen_difficulty={gen_diff}"
    )


# ── Heuristic AMAN planner ────────────────────────────────────────────────────

def _build_aman_heuristic(
    obs: MultiAgentObservation,
    separation_repair: int = 0,
) -> AMANAction:
    """Deterministic arrival sequencer — priority-sorted, wake-turbulence aware.

    ``separation_repair`` (0 = off): in later negotiate passes, add N minutes to
    required wake/ spacing gaps and widen DMAN emergency buffers to clear residual
    cross-runway conflicts (stronger for larger N; typical 1–3).
    """
    repair = max(0, int(separation_repair))
    arrivals = sorted(obs.my_flights, key=_flight_sort_key)
    runway_last: Dict[str, Tuple[int, str]] = {
        r.runway_id: (0, "M") for r in obs.all_runways
    }
    from constants import SEPARATION_BY_WAKE
    slots: List[SlotAssignment] = []
    emergency_yields: List[str] = []

    # Check for DMAN emergency broadcasts — yield slots around those minutes
    dman_emg_slots: List[Tuple[str, int]] = [
        (m.runway_id, m.requested_minute)
        for m in obs.incoming_messages
        if m.is_emergency and m.from_role == AgentRole.DMAN
    ]

    for flight in arrivals:
        best_rwy, best_min = None, None
        best_gap = -1

        for rwy_id in flight.allowed_runways:
            rwy = next((r for r in obs.all_runways if r.runway_id == rwy_id), None)
            if not rwy:
                continue
            if OperationType.ARRIVAL not in rwy.allowed_operations:
                continue

            last_min, last_wake = runway_last.get(rwy_id, (0, "M"))
            from engine import _capacity_spacing
            cap_gap = _capacity_spacing(rwy)
            wake_gap = SEPARATION_BY_WAKE.get(
                (last_wake, flight.wake_class.value), 3
            )
            min_gap = max(cap_gap, wake_gap) + repair
            earliest = max(flight.earliest_minute, last_min + min_gap)

            if earliest > flight.latest_minute:
                continue

            em_buf = 3 + max(1, repair)
            bump = 3 + max(1, repair)
            # Avoid DMAN emergency slots (±em_buf min)
            blocked = any(
                rwy_id == er and abs(earliest - et) < em_buf
                for er, et in dman_emg_slots
            )
            if blocked:
                earliest = min(flight.latest_minute, earliest + bump)
                if earliest > flight.latest_minute:
                    continue
                emergency_yields.append(flight.flight_id)

            gap_to_scheduled = abs(earliest - flight.scheduled_minute)
            if best_rwy is None or gap_to_scheduled < best_gap:
                best_rwy = rwy_id
                best_min = earliest
                best_gap = gap_to_scheduled

        if best_rwy and best_min is not None:
            slots.append(SlotAssignment(
                flight_id=flight.flight_id,
                runway=best_rwy,
                assigned_minute=best_min,
                hold_minutes=max(0, best_min - flight.scheduled_minute),
            ))
            runway_last[best_rwy] = (best_min, flight.wake_class.value)

    # Build outgoing messages — claim slots and broadcast emergencies
    messages: List[NegotiationMessage] = []
    for s in slots:
        flight = next((f for f in arrivals if f.flight_id == s.flight_id), None)
        if flight and flight.priority in (PriorityClass.EMERGENCY, PriorityClass.MEDICAL):
            messages.append(NegotiationMessage(
                from_role=AgentRole.AMAN,
                message_type=MessageType.EMERGENCY_BCAST,
                flight_id=s.flight_id,
                requested_minute=s.assigned_minute,
                runway_id=s.runway,
                priority=flight.priority,
                reason=f"Emergency arrival — runway {s.runway} must be clear at T+{s.assigned_minute}",
                is_emergency=True,
            ))
        else:
            messages.append(NegotiationMessage(
                from_role=AgentRole.AMAN,
                message_type=MessageType.RUNWAY_CLAIM,
                flight_id=s.flight_id if flight else "?",
                requested_minute=s.assigned_minute,
                runway_id=s.runway,
                priority=flight.priority if flight else PriorityClass.NORMAL,
                reason="Arrival slot claim",
                is_emergency=False,
            ))

    return AMANAction(
        arrival_slots=slots,
        rationale=f"Heuristic AMAN: {len(slots)}/{len(arrivals)} arrivals sequenced, "
                  f"priority-sorted, wake-turbulence aware"
                  f"{f', separation-repair (level {repair})' if repair else ''}.",
        emergency_yields=emergency_yields,
        outgoing_messages=messages,
        commit=False,
    )


# ── Heuristic DMAN planner ────────────────────────────────────────────────────

def _build_dman_heuristic(
    obs: MultiAgentObservation,
    atfm_deadlines: Dict[str, int],
    separation_repair: int = 0,
) -> DMANAction:
    """Deterministic departure sequencer — ATFM-aware, emergency priority.

    ``separation_repair`` (0 = off): add N minutes to wake gaps and expand buffers
    around AMAN-claimed times to break residual cross-lane spacing (larger N = more slack).
    """
    repair = max(0, int(separation_repair))
    from constants import SEPARATION_BY_WAKE
    from engine import _capacity_spacing
    from planner import PRIORITY_RANK

    def _departure_sort_key(flight: FlightRecord) -> Tuple[int, int, int, int, float, int]:
        deadline = atfm_deadlines.get(flight.flight_id)
        # Keep emergency traffic first, then pull ATFM-constrained flights ahead of
        # unconstrained traffic so hard network slot windows are not accidentally lost.
        deadline_bucket = 0 if flight.priority in (PriorityClass.EMERGENCY, PriorityClass.MEDICAL) else (
            1 if deadline is not None else 2
        )
        return (
            deadline_bucket,
            deadline if deadline is not None else flight.scheduled_minute,
            PRIORITY_RANK[flight.priority],
            flight.scheduled_minute,
            -flight.connection_risk,
            -flight.passengers,
        )

    departures = sorted(obs.my_flights, key=_departure_sort_key)
    runway_last: Dict[str, Tuple[int, str]] = {
        r.runway_id: (0, "M") for r in obs.all_runways
    }

    # Reserve slots claimed by AMAN
    aman_claims: Dict[str, List[int]] = {}
    for m in obs.incoming_messages:
        if m.from_role == AgentRole.AMAN:
            aman_claims.setdefault(m.runway_id, []).append(m.requested_minute)

    slots: List[SlotAssignment] = []
    atfm_compliance: Dict[str, int] = {}
    emergency_broadcasts: List[str] = []
    messages: List[NegotiationMessage] = []

    for flight in departures:
        is_emg = flight.priority in (PriorityClass.EMERGENCY, PriorityClass.MEDICAL)
        deadline = atfm_deadlines.get(flight.flight_id)
        best_rwy, best_min = None, None
        best_score = float("inf")

        for rwy_id in flight.allowed_runways:
            rwy = next((r for r in obs.all_runways if r.runway_id == rwy_id), None)
            if not rwy:
                continue
            if OperationType.DEPARTURE not in rwy.allowed_operations:
                continue

            last_min, last_wake = runway_last.get(rwy_id, (0, "M"))
            cap_gap  = _capacity_spacing(rwy)
            wake_gap = SEPARATION_BY_WAKE.get((last_wake, flight.wake_class.value), 3)
            min_gap  = max(cap_gap, wake_gap) + repair
            earliest = max(flight.earliest_minute, last_min + min_gap)

            if earliest > flight.latest_minute:
                continue

            claim_buf = 3 + max(1, repair)
            # Avoid AMAN-claimed slots (±claim_buf min) unless this is an emergency
            if not is_emg:
                aman_mins = aman_claims.get(rwy_id, [])
                while any(abs(earliest - am) < claim_buf for am in aman_mins):
                    earliest += 1
                if earliest > flight.latest_minute:
                    continue

            # ATFM deadline constraint
            if deadline is not None and earliest > deadline:
                continue

            score = abs(earliest - flight.scheduled_minute)
            if score < best_score:
                best_rwy  = rwy_id
                best_min  = earliest
                best_score = score

        if best_rwy and best_min is not None:
            slots.append(SlotAssignment(
                flight_id=flight.flight_id,
                runway=best_rwy,
                assigned_minute=best_min,
                hold_minutes=max(0, best_min - flight.scheduled_minute),
            ))
            runway_last[best_rwy] = (best_min, flight.wake_class.value)

            if deadline is not None:
                atfm_compliance[flight.flight_id] = deadline

            msg_type = MessageType.EMERGENCY_BCAST if is_emg else MessageType.RUNWAY_CLAIM
            if is_emg:
                emergency_broadcasts.append(flight.flight_id)
            messages.append(NegotiationMessage(
                from_role=AgentRole.DMAN,
                message_type=msg_type,
                flight_id=flight.flight_id,
                requested_minute=best_min,
                runway_id=best_rwy,
                priority=flight.priority,
                reason=(
                    f"Emergency departure — yield runway {best_rwy} at T+{best_min}"
                    if is_emg else "Departure slot claim"
                ),
                is_emergency=is_emg,
            ))

    return DMANAction(
        departure_slots=slots,
        rationale=f"Heuristic DMAN: {len(slots)}/{len(departures)} departures sequenced, "
                  f"ATFM-compliant, emergency-priority"
                  f"{f', separation-repair (level {repair})' if repair else ''}.",
        atfm_compliance=atfm_compliance,
        emergency_broadcasts=emergency_broadcasts,
        outgoing_messages=messages,
        commit=False,
    )


# ── LLM agent ─────────────────────────────────────────────────────────────────

def _llm_action(
    client,
    model_name: str,
    system: str,
    obs: MultiAgentObservation,
    sup_desc: str,
    role: AgentRole,
    temperature: Optional[float] = None,
    system_extra: str = "",
) -> Optional[AMANAction | DMANAction]:
    if client is None:
        return None
    temp = TEMPERATURE if temperature is None else temperature
    system_content = system + f"\n\nSUPERVISOR TODAY: {sup_desc}"
    if (system_extra or "").strip():
        system_content = system_content + "\n\n" + system_extra.strip()
    try:
        resp = client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": system_content},
                {"role": "user",   "content": obs.to_prompt_text()},
            ],
            temperature=temp,
            max_tokens=MAX_TOKENS,
        )
        text = (resp.choices[0].message.content or "").strip()
        if role == AgentRole.AMAN:
            return parse_aman_action(text)
        return parse_dman_action(text)
    except Exception as exc:
        _p(f"[WARN] LLM call failed ({role.value}): {exc} — falling back to heuristic")
        return None


def _negotiate_multi_pass(
    client,
    selected_model: str,
    sup_desc: str,
    env: MultiAgentATCEnvironment,
    aman_obs_start: MultiAgentObservation,
    dman_obs_start: MultiAgentObservation,
    atfm: Dict[str, int],
    rounds: List[Dict[str, Any]],
    visual_sink: VisualSink = None,
    max_passes: Optional[int] = None,
    llm_system_extra: str = "",
) -> Tuple[MultiAgentObservation, MultiAgentObservation]:
    """Revision passes on separation conflicts until resolved or cap reached.

    Pass 1 uses the LLM when ``client`` is set (lower temperature for stability);
    later passes use tiered separation-repair heuristics (0 = off; higher passes
    add more inter-slot slack).  Native ATC uses 2 / 1 passes (with/without client);
    ADAPT domain transfer uses DOMAIN_MAX_NEGOTIATE_ROUNDS for more repair attempts.
    """
    aman_cur, dman_cur = aman_obs_start, dman_obs_start
    if max_passes is not None:
        max_p = int(max_passes)
    elif client is not None:
        max_p = MAX_NEGOTIATE_ROUNDS
    else:
        max_p = 1

    for neg_r in range(1, max_p + 1):
        sep_before = count_separation_issues(env._state.conflict_log)
        if sep_before == 0:
            break

        use_llm = client is not None and neg_r == 1
        if use_llm:
            aman_action = (
                _llm_action(
                    client,
                    selected_model,
                    AMAN_SYSTEM,
                    aman_cur,
                    sup_desc,
                    AgentRole.AMAN,
                    temperature=0.2,
                    system_extra=llm_system_extra,
                )
                or _build_aman_heuristic(aman_cur)
            )
            dman_action = (
                _llm_action(
                    client,
                    selected_model,
                    DMAN_SYSTEM,
                    dman_cur,
                    sup_desc,
                    AgentRole.DMAN,
                    temperature=0.2,
                    system_extra=llm_system_extra,
                )
                or _build_dman_heuristic(dman_cur, atfm)
            )
        else:
            repair = 0 if neg_r == 1 else min(neg_r - 1, 3)
            aman_action = _build_aman_heuristic(aman_cur, separation_repair=repair)
            dman_action = _build_dman_heuristic(dman_cur, atfm, separation_repair=repair)

        aman_cur, dman_cur, partial_r, _ = env.step_negotiate(aman_action, dman_action)
        sep_after = count_separation_issues(env._state.conflict_log)
        resolved = max(0, sep_before - sep_after)
        total_msgs = len(aman_action.outgoing_messages) + len(dman_action.outgoing_messages)
        log_neg(neg_r, resolved, total_msgs)
        log_step(1 + neg_r, "NEGOTIATE", partial_r, sep_after, True)

        emit(
            visual_sink,
            {
                "type": "negotiation_tick",
                "pass": neg_r,
                "partial_reward": partial_r,
                "separation_issues_after": sep_after,
                "used_llm": use_llm,
                "layout": serialize_action_layout(aman_action, dman_action),
            },
        )

        rounds.append({
            "round": "NEGOTIATE",
            "negotiate_pass": neg_r,
            "partial_reward": partial_r,
            "separation_issues_before": sep_before,
            "separation_issues_after": sep_after,
            "diagnostics_count": len(env._state.conflict_log),
            "conflicts_resolved": resolved,
            "total_messages": total_msgs,
            "aman_alternatives_offered": sum(
                len(m.proposed_alternatives) for m in aman_action.outgoing_messages
            ),
            "dman_alternatives_offered": sum(
                len(m.proposed_alternatives) for m in dman_action.outgoing_messages
            ),
            "used_llm": use_llm,
        })

        if sep_after == 0:
            break

    return aman_cur, dman_cur


# ── Episode runner ─────────────────────────────────────────────────────────────

def _save_transcript(
    transcript_dir: Path,
    episode_id: int,
    data: Dict[str, Any],
) -> None:
    """Write one episode transcript to <transcript_dir>/episode_NNNNN.json."""
    transcript_dir.mkdir(parents=True, exist_ok=True)
    path = transcript_dir / f"episode_{episode_id:05d}.json"
    try:
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(data, fh, indent=2, default=str)
    except OSError as exc:
        _p(f"[WARN] Could not save transcript for episode {episode_id}: {exc}")


def run_episode(
    task_id: str,
    client,
    env: MultiAgentATCEnvironment,
    generator: Optional[ChallengeGenerator],
    supervisor: SupervisorAgent,
    episode_id: int,
    use_generator: bool = True,
    model_name: Optional[str] = None,
    transcript_dir: Optional[Path] = None,
    visual_sink: VisualSink = None,
    visual_profile: str = "atc",
) -> Dict:
    """Run one full AMAN/DMAN episode. Returns result dict for logging.

    Args:
        transcript_dir: If set, writes a JSON transcript of the full episode
            (actions, messages, rewards, metadata) to this directory.
        visual_sink: Optional callback receiving visual event dicts (SSE / UI).
        visual_profile: ``atc`` | ``icu`` — UI theme only (same simulation).
    """
    catalog  = task_catalog()
    base_task = catalog.get(task_id, list(catalog.values())[0])
    profile   = supervisor.sample_profile(episode_id)
    sup_desc  = SUPERVISOR_PROFILES[profile]["description"]

    mutations_applied: List[str] = []
    if use_generator and generator is not None:
        mutated_task, solvable = generator.mutate(base_task)
        gen_difficulty = generator.difficulty_level
        mutations_applied = [
            m["type"]
            for m in getattr(generator, "_mutation_history", [])[-3:]
        ]

    else:
        mutated_task, solvable = base_task, True
        gen_difficulty = 1

    # Enable domain randomisation when the generator is inactive — replaces the
    # structural diversity that generator mutations would otherwise provide.
    aman_obs, dman_obs = env.reset(
        episode_id=episode_id,
        supervisor_profile=profile,
        mutated_task=mutated_task,
        randomize=not use_generator,
    )
    selected_model = model_name or MODEL_NAME
    atfm = env._state.atfm_deadlines

    # Accumulate round data for the transcript
    rounds: List[Dict[str, Any]] = []
    visual_log: List[Dict[str, Any]] = []

    def _v(ev: Dict[str, Any]) -> None:
        visual_log.append(dict(ev))
        emit(visual_sink, ev)

    assert env._state.task is not None
    _v(
        {
            "type": "scene_reset",
            "episode_id": episode_id,
            "task_id": task_id,
            "resolved_task_id": env._state.task.task_id,
            "visual_profile": visual_profile,
            "supervisor_profile": profile.value,
            "mutations_applied": mutations_applied,
            "task": serialize_task_snapshot(env._state.task),
        },
    )

    # ── Round 0: BID ──────────────────────────────────────────────────────────
    _v({"type": "llm_started", "role": "AMAN", "model": selected_model})
    aman_action = (
        _llm_action(client, selected_model, AMAN_SYSTEM, aman_obs, sup_desc, AgentRole.AMAN)
        or _build_aman_heuristic(aman_obs)
    )
    _v({"type": "llm_finished", "role": "AMAN", "used_llm": client is not None})

    _v({"type": "llm_started", "role": "DMAN", "model": selected_model})
    dman_action = (
        _llm_action(client, selected_model, DMAN_SYSTEM, dman_obs, sup_desc, AgentRole.DMAN)
        or _build_dman_heuristic(dman_obs, atfm)
    )
    _v({"type": "llm_finished", "role": "DMAN", "used_llm": client is not None})

    _v({"type": "action_layout", "phase": "BID", "layout": serialize_action_layout(aman_action, dman_action)})

    aman_obs2, dman_obs2, partial_r, done = env.step_bid(aman_action, dman_action)
    n_sep_after_bid = count_separation_issues(env._state.conflict_log)
    n_diag_after_bid = len(env._state.conflict_log)
    log_step(1, "BID", partial_r, n_sep_after_bid, done)

    rounds.append({
        "round": "BID",
        "partial_reward": partial_r,
        "separation_issues_detected": n_sep_after_bid,
        "diagnostics_count": n_diag_after_bid,
        "aman_slots": len(aman_action.arrival_slots),
        "dman_slots": len(dman_action.departure_slots),
        "aman_messages": len(aman_action.outgoing_messages),
        "dman_messages": len(dman_action.outgoing_messages),
        "done": done,
    })

    # ── Negotiate: LLM first (if available), then heuristic-only repair passes ─
    if not done:
        _negotiate_multi_pass(
            client,
            selected_model,
            sup_desc,
            env,
            aman_obs2,
            dman_obs2,
            atfm,
            rounds,
            visual_sink=lambda ev: _v(ev),
        )

    # ── Finalize ──────────────────────────────────────────────────────────────
    result = env.finalize()
    if generator is not None:
        generator.update(result.composite_score)
        gen_difficulty = generator.difficulty_level

    episode_result = {
        "composite":      result.composite_score,
        "aman_reward":    result.aman_reward,
        "dman_reward":    result.dman_reward,
        "coord_score":    result.per_role.coordination_score,
        "conflicts":      result.per_role.cross_lane_conflicts,
        "atfm_viol":      result.per_role.atfm_violations,
        "emg_arr_ok":     result.per_role.emergency_arrivals_ok,
        "emg_dep_ok":     result.per_role.emergency_departures_ok,
        "neg_rounds":     result.negotiation_rounds,
        "gen_difficulty": gen_difficulty,
        "supervisor":     profile.value,
        "solvable":       solvable,
    }

    _v(
        {
            "type": "score_update",
            "composite": result.composite_score,
            "aman_reward": result.aman_reward,
            "dman_reward": result.dman_reward,
        },
    )
    _v(
        terminal_event(
            composite=result.composite_score,
            aman_reward=result.aman_reward,
            dman_reward=result.dman_reward,
            coordination=result.per_role.coordination_score,
            cross_lane_conflicts=result.per_role.cross_lane_conflicts,
            atfm_violations=result.per_role.atfm_violations,
            negotiation_rounds=result.negotiation_rounds,
        )
    )

    if transcript_dir is not None:
        _save_transcript(
            transcript_dir,
            episode_id,
            {
                "episode_id":        episode_id,
                "task_id":           task_id,
                "supervisor_profile": profile.value,
                "gen_difficulty":    gen_difficulty,
                "mutations_applied": mutations_applied,
                "solvable":          solvable,
                "visual_log":        visual_log,
                "rounds":            rounds,
                "final": {
                    "composite_score":    result.composite_score,
                    "aman_reward":        result.aman_reward,
                    "dman_reward":        result.dman_reward,
                    "generator_reward":   result.generator_reward,
                    "supervisor_score":   result.supervisor_score,
                    "coordination_score": result.per_role.coordination_score,
                    "cross_lane_conflicts": result.per_role.cross_lane_conflicts,
                    "negotiation_rounds": result.negotiation_rounds,
                    "atfm_violations":    result.per_role.atfm_violations,
                    "emergency_arrivals_ok":   result.per_role.emergency_arrivals_ok,
                    "emergency_departures_ok": result.per_role.emergency_departures_ok,
                },
            },
        )

    return episode_result

def run_domain_episode(
    domain_task_id: str,
    client,
    env: MultiAgentATCEnvironment,
    supervisor: SupervisorAgent,
    episode_id: int,
    model_name: Optional[str] = None,
    transcript_dir: Optional[Path] = None,
    visual_sink: VisualSink = None,
    visual_profile: str = "icu",
) -> Dict:
    """Run one ADAPT domain-transfer episode.

    ADAPT maps the domain task → ATC-parameterised task.
    AMAN and DMAN then solve it with no code changes.

    Args:
        domain_task_id: Key from the domains registry, e.g. 'icu_mass_casualty'.
    """
    from domains import get_all_domain_tasks
    from multi_agent.adapt import (
        build_adapt_observation,
        _build_adapt_heuristic,
        apply_adapt_mapping,
        parse_adapt_action,
    )
    from training.dataset import ADAPT_SYSTEM

    all_domain_tasks = get_all_domain_tasks()
    if domain_task_id not in all_domain_tasks:
        available = ", ".join(sorted(all_domain_tasks.keys()))
        raise ValueError(
            f"Domain task {domain_task_id!r} not found. Available: {available}"
        )

    domain_task = all_domain_tasks[domain_task_id]
    profile     = supervisor.sample_profile(episode_id)
    sup_desc    = SUPERVISOR_PROFILES[profile]["description"]
    rounds: List[Dict[str, Any]] = []
    visual_log: List[Dict[str, Any]] = []

    def _v(ev: Dict[str, Any]) -> None:
        visual_log.append(dict(ev))
        emit(visual_sink, ev)

    # ── ADAPT step ────────────────────────────────────────────────────────────
    adapt_obs = build_adapt_observation(task=domain_task, profile=profile)

    adapt_action: Optional[object] = None
    adapt_rationale = "(heuristic)"

    _v(
        {
            "type": "adapt_scene",
            "domain_task_id": domain_task_id,
            "visual_profile": visual_profile,
            "entity_types": list(adapt_obs.entity_types),
        },
    )

    if client is not None:
        try:
            resp = client.chat.completions.create(
                model=model_name or MODEL_NAME,
                messages=[
                    {"role": "system", "content": ADAPT_SYSTEM},
                    {"role": "user",   "content": adapt_obs.to_prompt_text()},
                ],
                temperature=TEMPERATURE,
                max_tokens=MAX_TOKENS,
            )
            text = (resp.choices[0].message.content or "").strip()
            adapt_action   = parse_adapt_action(text)
            adapt_rationale = (adapt_action.rationale if adapt_action else "(LLM parse failed)")
        except Exception as exc:
            _p(f"[WARN] ADAPT LLM call failed: {exc} — falling back to structural heuristic")

    if adapt_action is None:
        adapt_action   = _build_adapt_heuristic(adapt_obs, domain_task)
        adapt_rationale = adapt_action.rationale

    _v(
        {
            "type": "adapt_mapping",
            "wake_map": dict(adapt_action.entity_wake_map),
            "priority_map": dict(adapt_action.entity_priority_map),
            "rationale_preview": (adapt_rationale or "")[:200],
        },
    )

    _p(f"[ADAPT] domain={domain_task_id} entities={adapt_obs.entity_types}")
    _p(f"[ADAPT] wake_map={adapt_action.entity_wake_map}")
    _p(f"[ADAPT] priority_map={adapt_action.entity_priority_map}")
    _p(f"[ADAPT] rationale={adapt_rationale[:120].encode('ascii', errors='replace').decode()}")


    # ── Apply ADAPT mapping and run AMAN/DMAN ────────────────────────────────
    mapped_task = apply_adapt_mapping(domain_task, adapt_action)

    aman_obs, dman_obs = env.reset(
        episode_id=episode_id,
        supervisor_profile=profile,
        mutated_task=mapped_task,
    )
    atfm = env._state.atfm_deadlines
    selected_model = model_name or MODEL_NAME

    assert env._state.task is not None
    _v(
        {
            "type": "scene_reset",
            "episode_id": episode_id,
            "task_id": domain_task_id,
            "resolved_task_id": env._state.task.task_id,
            "visual_profile": visual_profile,
            "supervisor_profile": profile.value,
            "mutations_applied": [],
            "task": serialize_task_snapshot(env._state.task),
        },
    )

    _v({"type": "llm_started", "role": "AMAN", "model": selected_model})
    aman_action = _build_aman_heuristic(aman_obs)
    dman_action = _build_dman_heuristic(dman_obs, atfm)
    if client is not None:
        aman_action = (
            _llm_action(
                client,
                selected_model,
                AMAN_SYSTEM,
                aman_obs,
                sup_desc,
                AgentRole.AMAN,
                system_extra=_DOMAIN_TRANSFER_LLM_HINT,
            )
            or aman_action
        )
    _v({"type": "llm_finished", "role": "AMAN", "used_llm": client is not None})

    _v({"type": "llm_started", "role": "DMAN", "model": selected_model})
    if client is not None:
        dman_action = (
            _llm_action(
                client,
                selected_model,
                DMAN_SYSTEM,
                dman_obs,
                sup_desc,
                AgentRole.DMAN,
                system_extra=_DOMAIN_TRANSFER_LLM_HINT,
            )
            or dman_action
        )
    _v({"type": "llm_finished", "role": "DMAN", "used_llm": client is not None})

    _v({"type": "action_layout", "phase": "BID", "layout": serialize_action_layout(aman_action, dman_action)})

    aman_obs2, dman_obs2, partial_r, done = env.step_bid(aman_action, dman_action)
    n_sep_after_bid = count_separation_issues(env._state.conflict_log)
    n_diag_after_bid = len(env._state.conflict_log)
    log_step(1, "BID", partial_r, n_sep_after_bid, done)

    rounds.append({
        "round": "BID",
        "partial_reward": partial_r,
        "separation_issues_detected": n_sep_after_bid,
        "diagnostics_count": n_diag_after_bid,
        "aman_slots": len(aman_action.arrival_slots),
        "dman_slots": len(dman_action.departure_slots),
        "aman_messages": len(aman_action.outgoing_messages),
        "dman_messages": len(dman_action.outgoing_messages),
        "done": done,
    })

    if not done:
        _negotiate_multi_pass(
            client,
            selected_model,
            sup_desc,
            env,
            aman_obs2,
            dman_obs2,
            atfm,
            rounds,
            visual_sink=lambda ev: _v(ev),
            max_passes=DOMAIN_MAX_NEGOTIATE_ROUNDS,
            llm_system_extra=_DOMAIN_TRANSFER_LLM_HINT if client is not None else "",
        )

    result = env.finalize()

    _v(
        {
            "type": "score_update",
            "composite": result.composite_score,
            "aman_reward": result.aman_reward,
            "dman_reward": result.dman_reward,
        },
    )
    _v(
        terminal_event(
            composite=result.composite_score,
            aman_reward=result.aman_reward,
            dman_reward=result.dman_reward,
            coordination=result.per_role.coordination_score,
            cross_lane_conflicts=result.per_role.cross_lane_conflicts,
            atfm_violations=result.per_role.atfm_violations,
            negotiation_rounds=result.negotiation_rounds,
        )
    )

    episode_result: Dict[str, Any] = {
        "domain_task_id":  domain_task_id,
        "composite":       result.composite_score,
        "aman_reward":     result.aman_reward,
        "dman_reward":     result.dman_reward,
        "coord_score":     result.per_role.coordination_score,
        "conflicts":       result.per_role.cross_lane_conflicts,
        "neg_rounds":      result.negotiation_rounds,
        "supervisor":      profile.value,
        "adapt_wake_map":  adapt_action.entity_wake_map,
        "adapt_priority_map": adapt_action.entity_priority_map,
    }

    if transcript_dir is not None:
        _save_transcript(
            transcript_dir,
            episode_id,
            {
                "episode_id":         episode_id,
                "domain_task_id":     domain_task_id,
                "mode":               "adapt_domain_transfer",
                "supervisor_profile": profile.value,
                "adapt_wake_map":     adapt_action.entity_wake_map,
                "adapt_priority_map": adapt_action.entity_priority_map,
                "adapt_rationale":    adapt_rationale,
                "visual_log":         visual_log,
                "rounds":             rounds,
                "final": {
                    "composite_score":    result.composite_score,
                    "aman_reward":        result.aman_reward,
                    "dman_reward":        result.dman_reward,
                    "coordination_score": result.per_role.coordination_score,
                    "cross_lane_conflicts": result.per_role.cross_lane_conflicts,
                    "negotiation_rounds": result.negotiation_rounds,
                    "atfm_violations":    result.per_role.atfm_violations,
                },
            },
        )

    _p(
        f"[END]   domain={domain_task_id} success={str(result.composite_score >= SUCCESS_THRESHOLD).lower()} "
        f"composite={result.composite_score:.3f} aman={result.aman_reward:.3f} dman={result.dman_reward:.3f}"
    )
    return episode_result


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Multi-Agent ATC Inference Runner")
    parser.add_argument("--task",     default="bengaluru_irrops_hard",
                        help="Task ID (default: bengaluru_irrops_hard)")
    parser.add_argument("--episodes", type=int, default=1,
                        help="Number of episodes to run")
    parser.add_argument("--model",    default=MODEL_NAME,
                        help="Model name or 'heuristic-baseline'")
    parser.add_argument("--no_generator", action="store_true",
                        help="Disable self-play generator (use base tasks)")
    parser.add_argument("--all_tasks", action="store_true",
                        help="Run all ATC tasks in sequence")
    parser.add_argument("--domain", default=None,
                        help=(
                            "Run ADAPT domain-transfer mode on a domain task ID "
                            "(e.g. icu_mass_casualty, icu_flu_surge, icu_normal_day). "
                            "Use --domain list to show all available domain tasks."
                        ))
    parser.add_argument("--seed",     type=int, default=42)
    parser.add_argument("--transcript_dir", default=None,
                        help="Directory to write per-episode JSON transcripts "
                             "(default: no transcripts saved)")
    args = parser.parse_args()

    model = args.model or MODEL_NAME
    use_llm = model != "heuristic-baseline" and HF_TOKEN

    client = None
    if use_llm:
        try:
            from openai import OpenAI
            client = OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN)
            _p(f"[INFO] LLM client initialised: {model}")
        except ImportError:
            _p("[WARN] openai package missing — using heuristic baseline")

    env       = MultiAgentATCEnvironment(seed=args.seed)
    generator = ChallengeGenerator(seed=args.seed)
    supervisor = SupervisorAgent()

    transcript_dir = Path(args.transcript_dir) if args.transcript_dir else None

    # ── ADAPT domain-transfer mode ────────────────────────────────────────────
    if args.domain is not None:
        if args.domain == "list":
            from domains import get_all_domain_tasks
            all_dt = get_all_domain_tasks()
            _p("Available domain tasks:")
            for tid in sorted(all_dt.keys()):
                t = all_dt[tid]
                _p(f"  {tid:<35} ({t.difficulty.value}, {len(t.flights)} entities, {len(t.runways)} resources)")
            return

        _p(f"[INFO] ADAPT domain-transfer mode: {args.domain}")
        all_results: List[Dict] = []
        for ep in range(args.episodes):
            result = run_domain_episode(
                domain_task_id=args.domain,
                client=client,
                env=env,
                supervisor=supervisor,
                episode_id=ep,
                model_name=model,
                transcript_dir=transcript_dir,
            )
            all_results.append(result)

        if len(all_results) > 1:
            _p("\n=== DOMAIN TRANSFER SUMMARY ===")
            _p(f"Domain:          {args.domain}")
            _p(f"Episodes:        {len(all_results)}")
            _p(f"Mean composite:  {sum(r['composite'] for r in all_results)/len(all_results):.3f}")
            _p(f"Success rate:    {sum(1 for r in all_results if r['composite'] >= SUCCESS_THRESHOLD)/len(all_results):.1%}")
        return

    # ── Standard ATC mode ─────────────────────────────────────────────────────
    task_ids = (
        [t.task_id for t in ordered_tasks()]
        if args.all_tasks
        else [args.task]
    )

    all_results: List[Dict] = []

    for task_id in task_ids:
        log_start(task_id, model)
        for ep in range(args.episodes):
            result = run_episode(
                task_id=task_id,
                client=client,
                model_name=model,
                env=env,
                generator=generator,
                supervisor=supervisor,
                episode_id=ep,
                use_generator=not args.no_generator,
                transcript_dir=transcript_dir,
            )
            success = result["composite"] >= SUCCESS_THRESHOLD
            log_end(
                task=task_id,
                success=success,
                composite=result["composite"],
                aman=result["aman_reward"],
                dman=result["dman_reward"],
                coord=result["coord_score"],
                gen_diff=result["gen_difficulty"],
            )
            all_results.append({"task_id": task_id, "episode": ep, **result})

    # Summary
    if len(all_results) > 1:
        _p("\n=== MULTI-EPISODE SUMMARY ===")
        _p(f"Episodes:          {len(all_results)}")
        _p(f"Mean composite:    {sum(r['composite'] for r in all_results)/len(all_results):.3f}")
        _p(f"Mean AMAN reward:  {sum(r['aman_reward'] for r in all_results)/len(all_results):.3f}")
        _p(f"Mean DMAN reward:  {sum(r['dman_reward'] for r in all_results)/len(all_results):.3f}")
        _p(f"Mean coord score:  {sum(r['coord_score'] for r in all_results)/len(all_results):.3f}")
        _p(f"Total conflicts:   {sum(r['conflicts'] for r in all_results)}")
        _p(f"Success rate:      {sum(1 for r in all_results if r['composite'] >= SUCCESS_THRESHOLD)/len(all_results):.1%}")
        _p(f"Final gen level:   {all_results[-1]['gen_difficulty']}/6")


if __name__ == "__main__":
    main()

