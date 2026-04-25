"""Serialize AMAN/DMAN/ADAPT actions to strict JSON for SFT labels.

Every string produced here is intended to parse cleanly with
``parse_aman_action``, ``parse_dman_action``, and ``parse_adapt_action``.
"""

from __future__ import annotations

import json
from typing import Any, Dict, List

from multi_agent.models import AMANAction, ADAPTAction, DMANAction, NegotiationMessage


def _msg_to_dict(m: NegotiationMessage) -> Dict[str, Any]:
    pr = m.priority
    priority_val = pr.value if hasattr(pr, "value") else str(pr)
    return {
        "from_role": m.from_role.value,
        "message_type": m.message_type.value,
        "flight_id": m.flight_id,
        "requested_minute": int(m.requested_minute),
        "runway_id": m.runway_id,
        "priority": priority_val,
        "reason": m.reason,
        "is_emergency": bool(m.is_emergency),
    }


def aman_action_to_json_str(action: AMANAction) -> str:
    """Strict JSON for AMAN completions (no markdown fences)."""
    slots = [
        {
            "flight_id": s.flight_id,
            "runway": s.runway,
            "assigned_minute": int(s.assigned_minute),
            "hold_minutes": int(s.hold_minutes),
        }
        for s in action.arrival_slots
    ]
    obj: Dict[str, Any] = {
        "arrival_slots": slots,
        "rationale": action.rationale or "SFT teacher: heuristic arrival plan.",
        "emergency_yields": list(action.emergency_yields),
        "outgoing_messages": [_msg_to_dict(m) for m in action.outgoing_messages],
        "commit": bool(action.commit),
    }
    return json.dumps(obj, ensure_ascii=False, separators=(",", ":"))


def dman_action_to_json_str(action: DMANAction) -> str:
    """Strict JSON for DMAN completions (no markdown fences)."""
    slots = [
        {
            "flight_id": s.flight_id,
            "runway": s.runway,
            "assigned_minute": int(s.assigned_minute),
            "hold_minutes": int(s.hold_minutes),
        }
        for s in action.departure_slots
    ]
    atfm: Dict[str, int] = {str(k): int(v) for k, v in action.atfm_compliance.items()}
    obj: Dict[str, Any] = {
        "departure_slots": slots,
        "rationale": action.rationale or "SFT teacher: heuristic departure plan.",
        "atfm_compliance": atfm,
        "emergency_broadcasts": list(action.emergency_broadcasts),
        "outgoing_messages": [_msg_to_dict(m) for m in action.outgoing_messages],
        "commit": bool(action.commit),
    }
    return json.dumps(obj, ensure_ascii=False, separators=(",", ":"))


def adapt_action_to_json_str(action: ADAPTAction) -> str:
    """Strict JSON for ADAPT completions."""
    obj = {
        "entity_wake_map": dict(action.entity_wake_map),
        "entity_priority_map": dict(action.entity_priority_map),
        "rationale": action.rationale or "SFT teacher: structural mapping.",
    }
    return json.dumps(obj, ensure_ascii=False, separators=(",", ":"))


def verify_aman_parseable(action: AMANAction, parse_fn) -> bool:
    p = parse_fn(aman_action_to_json_str(action))
    return p is not None and len(p.arrival_slots) == len(action.arrival_slots)


def verify_dman_parseable(action: DMANAction, parse_fn) -> bool:
    p = parse_fn(dman_action_to_json_str(action))
    return p is not None and len(p.departure_slots) == len(action.departure_slots)


def verify_adapt_parseable(action: ADAPTAction, parse_fn) -> bool:
    p = parse_fn(adapt_action_to_json_str(action))
    return p is not None and bool(p.entity_wake_map) and bool(p.entity_priority_map)
