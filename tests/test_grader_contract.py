"""Grader contract tests for strict score bounds and deterministic scoring policy."""

from __future__ import annotations

from engine import simulate_plan
from graders import GatedCompositeGrader, LLMSupervisorGrader, grade_task
from models import ATCOptimizationAction
from planner import build_heuristic_plan
from server.atc_environment import ATCOptimizationEnvironment
from tasks import task_catalog


def _task_and_outcome(task_id: str = "mumbai_bank_balance_medium"):
    env = ATCOptimizationEnvironment()
    obs = env.reset(task_id=task_id)
    proposal = build_heuristic_plan(obs)
    task = task_catalog()[task_id]
    outcome = simulate_plan(task, proposal)
    return task, outcome, proposal


def test_llm_supervisor_fallback_score_is_strictly_bounded() -> None:
    task, outcome, proposal = _task_and_outcome("bengaluru_irrops_hard")
    grader = LLMSupervisorGrader()
    grade = grader.grade(task, outcome, proposal, rationale="short")
    assert 0.0 < grade.score < 1.0


def test_composite_uses_gated_deterministic_formula() -> None:
    task, outcome, proposal = _task_and_outcome()
    composite = GatedCompositeGrader().grade(task, outcome, proposal, rationale="Deterministic plan")
    # The composite score must be strictly bounded
    assert 0.0 < composite.score < 1.0
    # Sub-scores must be present
    assert "gate_ceiling" in composite.sub_scores
    assert "priority_score" in composite.sub_scores
    assert "efficiency_score" in composite.sub_scores


def test_all_graders_return_strictly_bounded_scores_for_all_tasks() -> None:
    for task_id in task_catalog():
        task, outcome, proposal = _task_and_outcome(task_id)
        grades = grade_task(task, outcome, proposal, "Deterministic contract test.")
        for grade in grades:
            assert 0.0 < grade.score < 1.0, (
                f"{grade.grader_name} out of strict range for {task_id}: {grade.score}"
            )


def test_step_rejects_duplicate_assignment_with_strictly_bounded_score() -> None:
    env = ATCOptimizationEnvironment()
    obs = env.reset(task_id="delhi_monsoon_recovery_easy")
    proposal = build_heuristic_plan(obs)
    duplicated = proposal + [proposal[0]]
    result = env.step(
        ATCOptimizationAction(
            proposal=duplicated,
            rationale="Intentional duplicate for robustness grading.",
            commit=True,
        )
    )
    assert result.current_metrics.invalid_assignments >= 1
    assert 0.0 < result.current_metrics.overall_score < 1.0
