"""Inference contract tests for logging, validation, and step semantics."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

import inference
from server.atc_environment import ATCOptimizationEnvironment


def _make_observation(task_id: str = "delhi_monsoon_recovery_easy"):
    env = ATCOptimizationEnvironment()
    return env.reset(task_id=task_id)


def test_log_format_matches_required_field_order(capsys) -> None:
    inference.log_start("task_a", "atc_optimization_openenv", "heuristic-baseline")
    inference.log_step(
        step=2,
        action="submit_plan(count=8,commit=false)",
        reward=0.125,
        done=False,
        error=None,
    )
    inference.log_end(
        task="task_a",
        success=True,
        steps=2,
        score=0.91,
        rewards=[0.5, 0.125],
    )

    lines = [line for line in capsys.readouterr().out.strip().splitlines() if line]
    assert lines[0] == "[START] task=task_a env=atc_optimization_openenv model=heuristic-baseline"
    assert (
        lines[1]
        == "[STEP] step=2 action=submit_plan(count=8,commit=false) reward=0.12 done=false error=null"
    )
    assert (
        lines[2]
        == "[END] task=task_a success=true steps=2 score=0.91 rewards=0.50,0.12"
    )


def test_extract_json_object_rejects_non_object_payload() -> None:
    with pytest.raises(ValueError, match="response payload is not an object"):
        inference._extract_json_object("[1,2,3]")


class _FakeCompletions:
    def __init__(self, content: str):
        self._content = content

    def create(self, **_: object):
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=self._content))]
        )


class _FakeClient:
    def __init__(self, content: str):
        self.chat = SimpleNamespace(completions=_FakeCompletions(content))


def test_get_model_action_falls_back_on_invalid_model_json(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(inference, "HF_TOKEN", "fake-token")
    observation = _make_observation()
    fake_client = _FakeClient("not-json")

    action = inference.get_model_action(
        client=fake_client,
        observation=observation,
        task_id=observation.task_id,
        step=1,
    )

    assert len(action.proposal) == len(observation.flights)
    assert "Fell back" in action.rationale


def test_step_budget_and_commit_semantics() -> None:
    assert inference._step_budget(0) == 1
    assert inference._step_budget(1) == 1
    assert inference._step_budget(10) <= inference.MAX_STEPS_CAP

    observation = _make_observation()
    first = inference.get_model_action(
        client=None,
        observation=observation,
        task_id=observation.task_id,
        step=1,
    )
    final = inference.get_model_action(
        client=None,
        observation=observation,
        task_id=observation.task_id,
        step=inference._step_budget(observation.steps_remaining),
    )

    assert first.commit is False
    assert final.commit is True
