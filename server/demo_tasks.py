"""Curated demo tasks for the Space UI (picker + ``visual_scene_key`` for scenes)."""

from __future__ import annotations

from typing import Any, Dict, List

# Keep in sync with ``space_frontend/src/tasks/registry.ts`` scene keys.
VISUAL_DEMO_TASKS: List[Dict[str, Any]] = [
    {"mode": "atc", "task_id": "bhopal_solo_dep_t0", "tier": 0, "visual_scene_key": "bhopal", "label": "Bhopal solo (T0)"},
    {"mode": "atc", "task_id": "vadodara_mixed_pair_t0", "tier": 0, "visual_scene_key": "vadodara", "label": "Vadodara mixed (T0)"},
    {"mode": "atc", "task_id": "pune_wake_intro_t1", "tier": 1, "visual_scene_key": "pune", "label": "Pune wake (T1)"},
    {"mode": "atc", "task_id": "nagpur_dual_runway_t1", "tier": 1, "visual_scene_key": "nagpur", "label": "Nagpur dual (T1)"},
    {"mode": "atc", "task_id": "delhi_monsoon_recovery_easy", "tier": 2, "visual_scene_key": "delhi", "label": "Delhi monsoon (easy)"},
    {"mode": "atc", "task_id": "mumbai_bank_balance_medium", "tier": 3, "visual_scene_key": "mumbai", "label": "Mumbai bank (med)"},
    {"mode": "atc", "task_id": "hyderabad_cargo_crunch_medium_hard", "tier": 2, "visual_scene_key": "hyderabad", "label": "Hyderabad cargo"},
    {"mode": "atc", "task_id": "bengaluru_irrops_hard", "tier": 4, "visual_scene_key": "bengaluru", "label": "Bengaluru IRROPS (hard)"},
    {"mode": "domain", "task_id": "icu_normal_day", "tier": 0, "visual_scene_key": "icu", "label": "ICU normal day"},
    {"mode": "domain", "task_id": "icu_flu_surge", "tier": 1, "visual_scene_key": "icu", "label": "ICU flu surge"},
    {"mode": "domain", "task_id": "icu_mass_casualty", "tier": 2, "visual_scene_key": "icu", "label": "ICU mass casualty"},
]
