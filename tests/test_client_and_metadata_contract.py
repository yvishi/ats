"""Client/state and metadata contract tests."""

from __future__ import annotations

from client import ATCOptimizationEnv
from models import ATCOptimizationState, default_allowed_operations
from server.atc_environment import ATCOptimizationEnvironment


def test_client_parse_state_regression() -> None:
    client = ATCOptimizationEnv(base_url="http://127.0.0.1:65535")
    payload = {
        "task_id": "demo",
        "task_title": "Demo Task",
        "seed": 7,
        "max_steps": 4,
        "active_task_ids": ["delhi_monsoon_recovery_easy", "mumbai_bank_balance_medium"],
    }
    parsed = client._parse_state(payload)
    assert isinstance(parsed, ATCOptimizationState)
    assert parsed.seed == 7
    assert "mumbai_bank_balance_medium" in parsed.active_task_ids


def test_models_default_factory_for_allowed_operations() -> None:
    defaults = default_allowed_operations()
    assert len(defaults) == 2
    assert defaults[0].value == "arrival"
    assert defaults[1].value == "departure"


def test_environment_metadata_and_task_enumeration_contract() -> None:
    env = ATCOptimizationEnvironment()
    metadata = env.get_metadata()
    assert metadata.author == "ATC Optimization OpenEnv Contributors"

    obs = env.reset(task_id="delhi_monsoon_recovery_easy")
    assert obs.task_id == "delhi_monsoon_recovery_easy"
    assert "bengaluru_irrops_hard" in env.state.active_task_ids
    assert len(env.state.active_task_ids) >= 3
