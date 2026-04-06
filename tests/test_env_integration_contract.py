"""Integration tests for environment state transitions and shared constants."""

from __future__ import annotations

import constants
import engine
import planner
from models import ATCOptimizationAction
from planner import build_heuristic_plan
from server.atc_environment import ATCOptimizationEnvironment


def test_reset_step_state_integration_flow() -> None:
    env = ATCOptimizationEnvironment()
    obs = env.reset(task_id="mumbai_bank_balance_medium")
    assert obs.steps_remaining > 0
    assert env.state.step_count == 0

    proposal = build_heuristic_plan(obs)
    next_obs = env.step(
        ATCOptimizationAction(
            proposal=proposal,
            rationale="Integration test plan",
            commit=False,
        )
    )

    assert env.state.step_count == 1
    assert len(env.state.history) == 1
    assert len(env.state.grader_history) >= 1
    assert next_obs.steps_remaining == max(0, env.state.max_steps - env.state.step_count)


def test_separation_matrix_is_centralized() -> None:
    assert planner.SEPARATION_BY_WAKE is constants.SEPARATION_BY_WAKE
    assert engine.SEPARATION_BY_WAKE is constants.SEPARATION_BY_WAKE
