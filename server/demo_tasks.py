"""Curated demo tasks for the Space UI (picker + ``visual_scene_key`` for scenes)."""

from __future__ import annotations

from typing import Any, Dict, List

from domains import get_all_domain_tasks
from multi_agent.visual_events import serialize_task_snapshot
from tasks import task_catalog

# Keep in sync with ``space_frontend/src/tasks/registry.ts`` scene keys.
_RAW_VISUAL_DEMO_TASKS: List[Dict[str, Any]] = [
    {
        "mode": "atc",
        "task_id": "bhopal_solo_dep_t0",
        "tier": 0,
        "visual_scene_key": "bhopal",
        "label": "Bhopal Solo Departure",
        "description": "Single runway, departures only. The simplest scheduling challenge—learn DMAN slot assignment basics.",
        "difficulty": "Introductory",
        "flight_count": 4,
        "runway_count": 1,
    },
    {
        "mode": "atc",
        "task_id": "vadodara_mixed_pair_t0",
        "tier": 0,
        "visual_scene_key": "vadodara",
        "label": "Vadodara Mixed Pair",
        "description": "Arrivals and departures share one runway. AMAN and DMAN must negotiate runway access.",
        "difficulty": "Introductory",
        "flight_count": 6,
        "runway_count": 1,
    },
    {
        "mode": "atc",
        "task_id": "pune_wake_intro_t1",
        "tier": 1,
        "visual_scene_key": "pune",
        "label": "Pune Wake Turbulence",
        "description": "Heavy aircraft require enforced ICAO wake separation. Violate it and scores collapse.",
        "difficulty": "Easy",
        "flight_count": 8,
        "runway_count": 1,
    },
    {
        "mode": "atc",
        "task_id": "nagpur_dual_runway_t1",
        "tier": 1,
        "visual_scene_key": "nagpur",
        "label": "Nagpur Dual Runway",
        "description": "Two independent runways with crossing taxiway constraints. Parallel scheduling unlocks throughput.",
        "difficulty": "Easy",
        "flight_count": 8,
        "runway_count": 2,
    },
    {
        "mode": "atc",
        "task_id": "delhi_monsoon_recovery_easy",
        "tier": 2,
        "visual_scene_key": "delhi",
        "label": "Delhi Monsoon Recovery",
        "description": "Monsoon weather disrupts the original plan. Recover sequence under ATFM constraints.",
        "difficulty": "Medium",
        "flight_count": 10,
        "runway_count": 2,
    },
    {
        "mode": "atc",
        "task_id": "mumbai_bank_balance_medium",
        "tier": 3,
        "visual_scene_key": "mumbai",
        "label": "Mumbai Bank Balance",
        "description": "India's busiest hub: banked arrivals over departures. Balance throughput or face cascade delays.",
        "difficulty": "Hard",
        "flight_count": 12,
        "runway_count": 2,
    },
    {
        "mode": "atc",
        "task_id": "hyderabad_cargo_crunch_medium_hard",
        "tier": 2,
        "visual_scene_key": "hyderabad",
        "label": "Hyderabad Cargo Crunch",
        "description": "Cargo freighters hold strict ATFM slots; wake turbulence from heavies threatens pax aircraft.",
        "difficulty": "Medium",
        "flight_count": 10,
        "runway_count": 2,
    },
    {
        "mode": "atc",
        "task_id": "bengaluru_irrops_hard",
        "tier": 4,
        "visual_scene_key": "bengaluru",
        "label": "Bengaluru IRROPS",
        "description": "Technical holds, diversion inbounds, and runway incursion risk. Expert-level coordination required.",
        "difficulty": "Expert",
        "flight_count": 14,
        "runway_count": 3,
    },
    {
        "mode": "domain",
        "task_id": "icu_normal_day",
        "tier": 0,
        "visual_scene_key": "icu",
        "label": "ICU Normal Day",
        "description": "ADAPT maps ATC reasoning to ICU bed management. Routine patient scheduling on a standard shift.",
        "difficulty": "Introductory",
        "flight_count": 8,
        "runway_count": 4,
    },
    {
        "mode": "domain",
        "task_id": "icu_flu_surge",
        "tier": 1,
        "visual_scene_key": "icu",
        "label": "ICU Flu Surge",
        "description": "Flu season surge doubles patient volume. Critical care beds overflow—prioritize triage.",
        "difficulty": "Medium",
        "flight_count": 12,
        "runway_count": 4,
    },
    {
        "mode": "domain",
        "task_id": "icu_mass_casualty",
        "tier": 2,
        "visual_scene_key": "icu",
        "label": "ICU Mass Casualty",
        "description": "Mass casualty incident. Agents must prevent catastrophic failure by coordinating across all beds.",
        "difficulty": "Hard",
        "flight_count": 16,
        "runway_count": 4,
    },
]


def _attach_task_previews(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Attach real runway/bed + flight/patient rows for idle canvas preview."""
    catalog = task_catalog()
    domain_map = get_all_domain_tasks()
    out: List[Dict[str, Any]] = []
    for row in rows:
        tid = row["task_id"]
        tdef = domain_map.get(tid) if row.get("mode") == "domain" else catalog.get(tid)
        preview = serialize_task_snapshot(tdef) if tdef is not None else None
        if preview is not None:
            row = {
                **row,
                "flight_count": len(tdef.flights),
                "runway_count": len(tdef.runways),
                "task_preview": preview,
            }
        out.append(row)
    return out


VISUAL_DEMO_TASKS: List[Dict[str, Any]] = _attach_task_previews(_RAW_VISUAL_DEMO_TASKS)
