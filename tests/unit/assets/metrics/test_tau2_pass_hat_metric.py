from __future__ import annotations

import pytest

from gage_eval.config.pipeline_config import MetricSpec
from gage_eval.metrics import MetricContext, MetricRegistry
from gage_eval.metrics.builtin.tau2 import Tau2PassHatMetric


def _make_context(sample_id: str, task_id: str, reward: float, mock_trace) -> MetricContext:
    return MetricContext(
        sample_id=sample_id,
        sample={"metadata": {"tau2": {"task_id": task_id}}},
        model_output={},
        judge_output={"tau2": {"reward": reward}},
        args={},
        trace=mock_trace,
    )


def test_tau2_pass_hat_metric_metadata(mock_trace) -> None:
    spec = MetricSpec(
        metric_id="tau2_pass_hat_k",
        implementation="tau2_pass_hat_k",
        aggregation="tau2_pass_hat",
        params={},
    )
    metric = Tau2PassHatMetric(spec)
    context = _make_context("sample-1", "task-1", 1.0, mock_trace)

    result = metric.compute(context)

    assert result.values["pass"] == 1.0
    assert result.metadata["task_id"] == "task-1"
    assert result.metadata["missing_reward"] is False


def test_tau2_pass_hat_aggregator(mock_trace) -> None:
    registry = MetricRegistry()
    spec = MetricSpec(
        metric_id="tau2_pass_hat_k",
        implementation="tau2_pass_hat_k",
        aggregation="tau2_pass_hat",
        params={},
    )
    instance = registry.build_metric(spec)

    contexts = [
        _make_context("sample-a-0", "task-a", 1.0, mock_trace),
        _make_context("sample-a-1", "task-a", 1.0, mock_trace),
        _make_context("sample-b-0", "task-b", 1.0, mock_trace),
        _make_context("sample-b-1", "task-b", 0.0, mock_trace),
    ]

    for context in contexts:
        instance.evaluate(context)

    aggregated = instance.finalize()

    assert aggregated["aggregation"] == "tau2_pass_hat"
    values = aggregated["values"]
    assert values["pass_hat@1"] == pytest.approx(0.75)
    assert values["pass_hat@2"] == pytest.approx(0.5)
