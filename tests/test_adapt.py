"""Smoke tests for the ADAPT meta-agent (domain transfer).

Covers:
  - build_adapt_observation
  - _build_adapt_heuristic
  - apply_adapt_mapping
  - parse_adapt_action
  - adapt_reward_fn
  - _make_adapt_sample
  - ICU domain catalog
  - ADAPT entry in REWARD_FN_DISPATCH
"""

from __future__ import annotations

import json

import pytest

from domains.icu import icu_task_catalog, ICU_DOMAIN_DESCRIPTION
from models import PriorityClass, WakeClass
from multi_agent.adapt import (
    _build_adapt_heuristic,
    apply_adapt_mapping,
    build_adapt_observation,
    parse_adapt_action,
)
from multi_agent.models import (
    ADAPTAction,
    ADAPTObservation,
    AgentRole,
    SupervisorProfileName,
)


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def icu_catalog():
    return icu_task_catalog()


@pytest.fixture(scope="module")
def normal_task(icu_catalog):
    return icu_catalog["icu_normal_day"]


@pytest.fixture(scope="module")
def mass_casualty_task(icu_catalog):
    return icu_catalog["icu_mass_casualty"]


@pytest.fixture(scope="module")
def flu_surge_task(icu_catalog):
    return icu_catalog["icu_flu_surge"]


@pytest.fixture(scope="module")
def normal_obs(normal_task):
    return build_adapt_observation(
        task=normal_task,
        profile=SupervisorProfileName.SAFETY_STRICT,
        domain_name="Hospital ICU Surge Management",
        domain_description=ICU_DOMAIN_DESCRIPTION,
    )


@pytest.fixture(scope="module")
def mass_casualty_obs(mass_casualty_task):
    return build_adapt_observation(
        task=mass_casualty_task,
        profile=SupervisorProfileName.EMERGENCY_FOCUS,
        domain_name="Hospital ICU Surge Management",
        domain_description=ICU_DOMAIN_DESCRIPTION,
    )


# ── ICU domain catalog ────────────────────────────────────────────────────────

def test_icu_catalog_has_three_tasks(icu_catalog):
    assert len(icu_catalog) == 3
    assert "icu_normal_day" in icu_catalog
    assert "icu_flu_surge" in icu_catalog
    assert "icu_mass_casualty" in icu_catalog


def test_icu_tasks_have_beds_as_runways(icu_catalog):
    for task in icu_catalog.values():
        runway_ids = {r.runway_id for r in task.runways}
        assert runway_ids == {"BED_A", "BED_B", "BED_C", "BED_D"}


def test_mass_casualty_has_trauma_arrivals(mass_casualty_task):
    trauma = [f for f in mass_casualty_task.flights if f.airline == "TRAUMA"]
    assert len(trauma) == 3, "mass_casualty must have exactly 3 TRAUMA arrivals"


def test_flu_surge_has_cardiac_arrivals(flu_surge_task):
    cardiac = [f for f in flu_surge_task.flights if f.airline == "CARDIAC"]
    assert len(cardiac) == 3, "flu_surge must have exactly 3 CARDIAC arrivals"


# ── build_adapt_observation ───────────────────────────────────────────────────

def test_adapt_observation_type(normal_obs):
    assert isinstance(normal_obs, ADAPTObservation)
    assert normal_obs.role == AgentRole.ADAPT


def test_adapt_observation_entity_types_complete(normal_obs, normal_task):
    expected = sorted({f.airline for f in normal_task.flights if f.airline})
    assert normal_obs.entity_types == expected


def test_adapt_observation_resource_names(normal_obs, normal_task):
    expected = [r.runway_id for r in normal_task.runways]
    assert normal_obs.resource_names == expected


def test_adapt_observation_entity_count(normal_obs, normal_task):
    assert normal_obs.entity_count == len(normal_task.flights)


def test_mass_casualty_obs_has_emergencies(mass_casualty_obs):
    assert mass_casualty_obs.has_emergencies is True


def test_mass_casualty_obs_has_hard_deadlines(mass_casualty_obs):
    assert mass_casualty_obs.has_hard_deadlines is True


def test_normal_obs_prompt_text_contains_domain(normal_obs):
    text = normal_obs.to_prompt_text()
    assert "ADAPT OBSERVATION" in text
    assert normal_obs.domain_name in text
    assert "ENTITY TYPES" in text


def test_adapt_observation_supervisor_description_nonempty(normal_obs):
    assert len(normal_obs.supervisor_description) > 10


# ── _build_adapt_heuristic ────────────────────────────────────────────────────

def test_heuristic_covers_all_entity_types(normal_obs, normal_task):
    action = _build_adapt_heuristic(normal_obs, normal_task)
    for et in normal_obs.entity_types:
        assert et in action.entity_wake_map, f"{et!r} missing from wake map"
        assert et in action.entity_priority_map, f"{et!r} missing from priority map"


def test_heuristic_trauma_maps_to_heavy_emergency(mass_casualty_obs, mass_casualty_task):
    action = _build_adapt_heuristic(mass_casualty_obs, mass_casualty_task)
    assert action.entity_wake_map.get("TRAUMA") == WakeClass.HEAVY.value
    assert action.entity_priority_map.get("TRAUMA") == PriorityClass.EMERGENCY.value


def test_heuristic_routine_maps_to_light(normal_obs, normal_task):
    action = _build_adapt_heuristic(normal_obs, normal_task)
    assert action.entity_wake_map.get("ROUTINE") == WakeClass.LIGHT.value
    assert action.entity_priority_map.get("ROUTINE") == PriorityClass.NORMAL.value


def test_heuristic_cardiac_maps_to_medium_medical(normal_obs, normal_task):
    action = _build_adapt_heuristic(normal_obs, normal_task)
    assert action.entity_wake_map.get("CARDIAC") == WakeClass.MEDIUM.value
    assert action.entity_priority_map.get("CARDIAC") == PriorityClass.MEDICAL.value


def test_heuristic_has_rationale(normal_obs, normal_task):
    action = _build_adapt_heuristic(normal_obs, normal_task)
    assert len(action.rationale.strip()) > 0


def test_heuristic_returns_adapt_action_type(normal_obs, normal_task):
    action = _build_adapt_heuristic(normal_obs, normal_task)
    assert isinstance(action, ADAPTAction)


# ── apply_adapt_mapping ───────────────────────────────────────────────────────

def test_apply_mapping_overrides_wake_class(mass_casualty_task):
    action = ADAPTAction(
        entity_wake_map={"TRAUMA": "H", "CARDIAC": "M", "ROUTINE": "L", "POST_OP": "M"},
        entity_priority_map={"TRAUMA": "emergency", "CARDIAC": "medical", "ROUTINE": "normal", "POST_OP": "connection"},
        rationale="test mapping",
    )
    mapped = apply_adapt_mapping(mass_casualty_task, action)
    for f in mapped.flights:
        if f.airline == "TRAUMA":
            assert f.wake_class == WakeClass.HEAVY
            assert f.priority == PriorityClass.EMERGENCY
        elif f.airline == "CARDIAC":
            assert f.wake_class == WakeClass.MEDIUM
            assert f.priority == PriorityClass.MEDICAL
        elif f.airline == "ROUTINE":
            assert f.wake_class == WakeClass.LIGHT
            assert f.priority == PriorityClass.NORMAL


def test_apply_mapping_preserves_unmatched_flights(normal_task):
    """Flights with unknown entity type should be passed through unchanged."""
    action = ADAPTAction(
        entity_wake_map={"UNKNOWN_TYPE": "H"},
        entity_priority_map={"UNKNOWN_TYPE": "emergency"},
        rationale="test",
    )
    mapped = apply_adapt_mapping(normal_task, action)
    # All flights should still be present
    assert len(mapped.flights) == len(normal_task.flights)


def test_apply_mapping_preserves_task_metadata(normal_task):
    action = ADAPTAction(
        entity_wake_map={"ROUTINE": "L"},
        entity_priority_map={"ROUTINE": "normal"},
        rationale="test",
    )
    mapped = apply_adapt_mapping(normal_task, action)
    assert mapped.task_id == normal_task.task_id
    assert mapped.delay_budget == normal_task.delay_budget
    assert len(mapped.runways) == len(normal_task.runways)


def test_apply_mapping_ignores_invalid_wake_value(normal_task):
    """Invalid wake class should be skipped — flight stays with original value."""
    original_wakes = {f.flight_id: f.wake_class for f in normal_task.flights}
    action = ADAPTAction(
        entity_wake_map={"ROUTINE": "INVALID_WAKE"},
        entity_priority_map={},
        rationale="test",
    )
    mapped = apply_adapt_mapping(normal_task, action)
    for f in mapped.flights:
        if f.airline == "ROUTINE":
            assert f.wake_class == original_wakes[f.flight_id]


# ── parse_adapt_action ────────────────────────────────────────────────────────

def test_parse_valid_json_completion():
    completion = json.dumps({
        "entity_wake_map": {"TRAUMA": "H", "CARDIAC": "M"},
        "entity_priority_map": {"TRAUMA": "emergency", "CARDIAC": "medical"},
        "rationale": "Trauma is most critical, mapped to H.",
    })
    action = parse_adapt_action(completion)
    assert action is not None
    assert action.entity_wake_map["TRAUMA"] == "H"
    assert action.entity_priority_map["CARDIAC"] == "medical"


def test_parse_json_in_markdown_fences():
    completion = "```json\n" + json.dumps({
        "entity_wake_map": {"ROUTINE": "L"},
        "entity_priority_map": {"ROUTINE": "normal"},
        "rationale": "Routine is standard.",
    }) + "\n```"
    action = parse_adapt_action(completion)
    assert action is not None
    assert action.entity_wake_map["ROUTINE"] == "L"


def test_parse_returns_none_for_non_json():
    assert parse_adapt_action("I cannot complete this task.") is None
    assert parse_adapt_action("") is None
    assert parse_adapt_action(None) is None


def test_parse_returns_none_for_malformed_json():
    assert parse_adapt_action("{entity_wake_map: broken}") is None


def test_parse_handles_dict_completion():
    """TRL sometimes passes completions as dicts."""
    completion = {
        "content": json.dumps({
            "entity_wake_map": {"TRAUMA": "H"},
            "entity_priority_map": {"TRAUMA": "emergency"},
            "rationale": "dict input test",
        })
    }
    action = parse_adapt_action(completion)
    assert action is not None
    assert action.entity_wake_map["TRAUMA"] == "H"


def test_parse_missing_rationale_defaults_to_empty():
    completion = json.dumps({
        "entity_wake_map": {"ROUTINE": "L"},
        "entity_priority_map": {"ROUTINE": "normal"},
    })
    action = parse_adapt_action(completion)
    assert action is not None
    assert action.rationale == ""


# ── adapt_reward_fn ───────────────────────────────────────────────────────────

def test_adapt_reward_returns_list_of_floats(mass_casualty_task):
    from training.reward_functions import adapt_reward_fn

    completion = json.dumps({
        "entity_wake_map": {"TRAUMA": "H", "CARDIAC": "M", "ROUTINE": "L", "POST_OP": "M"},
        "entity_priority_map": {"TRAUMA": "emergency", "CARDIAC": "medical", "ROUTINE": "normal", "POST_OP": "connection"},
        "rationale": "TRAUMA must be mapped to H/emergency for immediate ICU admission.",
    })
    rewards = adapt_reward_fn(
        completions=[completion],
        task_id=[mass_casualty_task.task_id],
        domain_task_json=[mass_casualty_task.model_dump_json()],
        supervisor_profile=[SupervisorProfileName.EMERGENCY_FOCUS.value],
    )
    assert isinstance(rewards, list)
    assert len(rewards) == 1
    assert isinstance(rewards[0], float)


def test_adapt_reward_bounded(mass_casualty_task):
    from training.reward_functions import adapt_reward_fn

    completion = json.dumps({
        "entity_wake_map": {"TRAUMA": "H", "CARDIAC": "M", "ROUTINE": "L", "POST_OP": "M"},
        "entity_priority_map": {"TRAUMA": "emergency", "CARDIAC": "medical", "ROUTINE": "normal", "POST_OP": "connection"},
        "rationale": "Critical mapping for mass casualty.",
    })
    rewards = adapt_reward_fn(
        completions=[completion],
        task_id=[mass_casualty_task.task_id],
        domain_task_json=[mass_casualty_task.model_dump_json()],
        supervisor_profile=[SupervisorProfileName.EMERGENCY_FOCUS.value],
    )
    assert -1.0 <= rewards[0] <= 1.0


def test_adapt_reward_bad_json_penalised(mass_casualty_task):
    from training.reward_functions import adapt_reward_fn

    rewards = adapt_reward_fn(
        completions=["not json at all"],
        task_id=[mass_casualty_task.task_id],
        domain_task_json=[mass_casualty_task.model_dump_json()],
        supervisor_profile=[SupervisorProfileName.SAFETY_STRICT.value],
    )
    assert rewards[0] <= -0.4


def test_adapt_reward_missing_domain_task_penalised():
    from training.reward_functions import adapt_reward_fn

    rewards = adapt_reward_fn(
        completions=["{}"],
        task_id=["some_task"],
        domain_task_json=[None],
        supervisor_profile=[SupervisorProfileName.SAFETY_STRICT.value],
    )
    assert rewards[0] <= -0.5


def test_adapt_reward_batch_length_matches(mass_casualty_task):
    """Reward list length must match completions list length."""
    from training.reward_functions import adapt_reward_fn

    dtjson = mass_casualty_task.model_dump_json()
    completions = [
        json.dumps({"entity_wake_map": {"TRAUMA": "H"}, "entity_priority_map": {"TRAUMA": "emergency"}, "rationale": "test"}),
        "bad json",
    ]
    rewards = adapt_reward_fn(
        completions=completions,
        task_id=[mass_casualty_task.task_id] * 2,
        domain_task_json=[dtjson] * 2,
        supervisor_profile=[SupervisorProfileName.SAFETY_STRICT.value] * 2,
    )
    assert len(rewards) == 2


# ── _make_adapt_sample ────────────────────────────────────────────────────────

def test_make_adapt_sample_structure(normal_task, normal_obs):
    from training.dataset import _make_adapt_sample

    sample = _make_adapt_sample(ep_id=0, obs=normal_obs, domain_task=normal_task)
    assert "prompt" in sample
    assert len(sample["prompt"]) == 2
    assert sample["prompt"][0]["role"] == "system"
    assert sample["prompt"][1]["role"] == "user"
    assert sample["agent_role"] == AgentRole.ADAPT.value
    assert sample["round"] == "adapt"
    assert "domain_task_json" in sample


def test_make_adapt_sample_prompt_contains_observation(normal_task, normal_obs):
    from training.dataset import _make_adapt_sample

    sample = _make_adapt_sample(ep_id=5, obs=normal_obs, domain_task=normal_task)
    user_content = sample["prompt"][1]["content"]
    assert "ADAPT OBSERVATION" in user_content
    assert normal_obs.domain_name in user_content


def test_make_adapt_sample_domain_task_json_valid(normal_task, normal_obs):
    from training.dataset import _make_adapt_sample

    sample = _make_adapt_sample(ep_id=0, obs=normal_obs, domain_task=normal_task)
    from models import TaskDefinition
    restored = TaskDefinition.model_validate_json(sample["domain_task_json"])
    assert restored.task_id == normal_task.task_id


def test_make_adapt_sample_system_contains_adapt_rules(normal_task, normal_obs):
    from training.dataset import ADAPT_SYSTEM, _make_adapt_sample

    sample = _make_adapt_sample(ep_id=0, obs=normal_obs, domain_task=normal_task)
    sys_content = sample["prompt"][0]["content"]
    assert "entity_wake_map" in sys_content
    assert "entity_priority_map" in sys_content


# ── Dispatch table ────────────────────────────────────────────────────────────

def test_adapt_in_dispatch_table():
    from training.train_grpo import REWARD_FN_DISPATCH
    from training.reward_functions import adapt_reward_fn

    assert AgentRole.ADAPT.value in REWARD_FN_DISPATCH
    assert REWARD_FN_DISPATCH[AgentRole.ADAPT.value] is adapt_reward_fn


def test_dispatch_table_has_all_roles():
    from training.train_grpo import REWARD_FN_DISPATCH

    for role in AgentRole:
        assert role.value in REWARD_FN_DISPATCH, f"{role.value} missing from dispatch table"


# ── build_episode_dataset integration ────────────────────────────────────────

def test_dataset_includes_adapt_samples():
    """~30% of episodes should produce ADAPT samples when include_adapt=True."""
    from training.dataset import build_episode_dataset

    samples = build_episode_dataset(n_episodes=20, seed=0, include_adapt=True, domain_episode_ratio=0.5)
    adapt_samples = [s for s in samples if s["agent_role"] == AgentRole.ADAPT.value]
    assert len(adapt_samples) >= 1, "Expected at least one ADAPT sample in 20 episodes at 50% ratio"


def test_dataset_adapt_samples_have_required_fields():
    from training.dataset import build_episode_dataset

    samples = build_episode_dataset(n_episodes=20, seed=7, include_adapt=True, domain_episode_ratio=0.8)
    adapt_samples = [s for s in samples if s["agent_role"] == AgentRole.ADAPT.value]
    for s in adapt_samples:
        assert "prompt" in s
        assert "domain_task_json" in s
        assert s["round"] == "adapt"
        assert s["supervisor_profile"] in {p.value for p in SupervisorProfileName}


def test_dataset_no_adapt_when_disabled():
    from training.dataset import build_episode_dataset

    samples = build_episode_dataset(n_episodes=20, seed=0, include_adapt=False)
    adapt_samples = [s for s in samples if s["agent_role"] == AgentRole.ADAPT.value]
    assert len(adapt_samples) == 0
