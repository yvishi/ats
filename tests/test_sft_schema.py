"""SFT label serialization must parse with the same parsers as rollout."""

from __future__ import annotations

import pytest

from multi_agent.inference import _build_aman_heuristic, _build_dman_heuristic
from multi_agent.environment import MultiAgentATCEnvironment
from multi_agent.adapt import parse_adapt_action, _build_adapt_heuristic, build_adapt_observation
from domains.icu import icu_task_catalog
from training.dataset import parse_aman_action, parse_dman_action
from training.sft_schema import (
    adapt_action_to_json_str,
    aman_action_to_json_str,
    dman_action_to_json_str,
    verify_adapt_parseable,
    verify_aman_parseable,
    verify_dman_parseable,
)
from multi_agent.models import SupervisorProfileName


@pytest.fixture
def env():
    return MultiAgentATCEnvironment(seed=1)


def test_aman_sft_json_parseable(env):
    aman_obs, _ = env.reset(task_id="vadodara_mixed_pair_t0", episode_id=0)
    action = _build_aman_heuristic(aman_obs)
    assert verify_aman_parseable(action, parse_aman_action)
    parsed = parse_aman_action(aman_action_to_json_str(action))
    assert parsed is not None
    assert len(parsed.arrival_slots) == len(action.arrival_slots)


def test_dman_sft_json_parseable(env):
    _, dman_obs = env.reset(task_id="vadodara_mixed_pair_t0", episode_id=0)
    action = _build_dman_heuristic(dman_obs, env._state.atfm_deadlines)
    assert verify_dman_parseable(action, parse_dman_action)
    parsed = parse_dman_action(dman_action_to_json_str(action))
    assert parsed is not None
    assert len(parsed.departure_slots) == len(action.departure_slots)


def test_adapt_sft_json_parseable():
    task = icu_task_catalog()["icu_normal_day"]
    profile = SupervisorProfileName.SAFETY_STRICT
    obs = build_adapt_observation(task, profile)
    teacher = _build_adapt_heuristic(obs, task)
    assert verify_adapt_parseable(teacher, parse_adapt_action)
    p = parse_adapt_action(adapt_action_to_json_str(teacher))
    assert p is not None
    assert p.entity_wake_map == teacher.entity_wake_map
    assert p.entity_priority_map == teacher.entity_priority_map
