from __future__ import annotations

import pytest

from gage_eval.reporting.contracts import validate_metric_scope


@pytest.mark.fast
@pytest.mark.parametrize(
    "metric",
    [
        {"metric_id": "reward_mean", "scope": "run", "value": 0.5},
        {"metric_id": "task_reward", "scope": "task", "task_id": "task-1", "value": 0.5},
        {"metric_id": "section_reward", "scope": "section", "section_id": "gen/overview", "value": 0.5},
    ],
)
def test_valid_metric_scope(metric: dict) -> None:
    assert validate_metric_scope(metric) == []


@pytest.mark.fast
@pytest.mark.parametrize(
    "metric,missing",
    [
        ({"metric_id": "x", "value": 1}, "scope"),
        ({"metric_id": "x", "scope": "task", "value": 1}, "task_id"),
        ({"metric_id": "x", "scope": "section", "value": 1}, "section_id"),
    ],
)
def test_invalid_metric_scope(metric: dict, missing: str) -> None:
    errors = validate_metric_scope(metric)

    assert any(missing in error for error in errors)
