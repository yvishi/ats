"""Normalized visual events for HF Space pixel-art UI (SSE / transcripts).

Keep payloads bounded: no prompts, only layout + scores + phase markers.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Literal, Optional

from models import FlightRecord, TaskDefinition
from multi_agent.models import AMANAction, DMANAction

VisualSink = Optional[Callable[[Dict[str, Any]], None]]

TerminalSeverity = Literal["success", "degraded", "catastrophic"]


def emit(sink: VisualSink, event: Dict[str, Any]) -> None:
    if sink is not None:
        sink(dict(event))


def _slot_row(s) -> Dict[str, Any]:
    return {
        "flight_id": s.flight_id,
        "runway": s.runway,
        "assigned_minute": int(s.assigned_minute),
        "hold_minutes": int(s.hold_minutes),
    }


def serialize_action_layout(aman: AMANAction, dman: DMANAction) -> Dict[str, Any]:
    return {
        "aman_arrivals": [_slot_row(s) for s in aman.arrival_slots],
        "dman_departures": [_slot_row(s) for s in dman.departure_slots],
    }


def _flight_minimal(f: FlightRecord) -> Dict[str, Any]:
    return {
        "flight_id": f.flight_id,
        "operation": f.operation.value if hasattr(f.operation, "value") else str(f.operation),
        "wake": f.wake_class.value if hasattr(f.wake_class, "value") else str(f.wake_class),
        "scheduled": int(f.scheduled_minute),
        "earliest": int(f.earliest_minute),
        "latest": int(f.latest_minute),
        "runways": list(f.allowed_runways),
        "priority": f.priority.value if hasattr(f.priority, "value") else str(f.priority),
    }


def serialize_task_snapshot(task: TaskDefinition) -> Dict[str, Any]:
    return {
        "task_id": task.task_id,
        "airport": getattr(task, "airport", None) or task.task_id,
        "runways": [
            {
                "runway_id": r.runway_id,
                "ops": [o.value for o in r.allowed_operations],
            }
            for r in task.runways
        ],
        "flights": [_flight_minimal(f) for f in task.flights],
    }


def classify_terminal(
    composite: float,
    cross_lane_conflicts: int,
    atfm_violations: int,
) -> TerminalSeverity:
    """Single policy for catastrophic vs degraded (shared UI animation)."""
    if composite < 0.12 or (composite < 0.22 and cross_lane_conflicts >= 6):
        return "catastrophic"
    if composite < 0.45 or cross_lane_conflicts >= 3 or atfm_violations >= 2:
        return "degraded"
    return "success"


def terminal_event(
    *,
    composite: float,
    aman_reward: float,
    dman_reward: float,
    coordination: float,
    cross_lane_conflicts: int,
    atfm_violations: int,
    negotiation_rounds: int,
) -> Dict[str, Any]:
    sev = classify_terminal(composite, cross_lane_conflicts, atfm_violations)
    return {
        "type": "terminal",
        "severity": sev,
        "composite": round(float(composite), 4),
        "aman_reward": round(float(aman_reward), 4),
        "dman_reward": round(float(dman_reward), 4),
        "coordination": round(float(coordination), 4),
        "cross_lane_conflicts": int(cross_lane_conflicts),
        "atfm_violations": int(atfm_violations),
        "negotiation_rounds": int(negotiation_rounds),
    }
