from __future__ import annotations

from gage_eval.config.pipeline_config import MetricSpec
from gage_eval.metrics.base import MetricContext
from gage_eval.metrics.builtin.terminal_bench import (
    TerminalBenchFailureReasonMetric,
    TerminalBenchResolveRateMetric,
)


def test_terminal_bench_resolve_rate_reads_runtime_eval_result(mock_trace) -> None:
    spec = MetricSpec(
        metric_id="terminal_bench_resolve_rate",
        implementation="terminal_bench_resolve_rate",
        params={},
    )
    metric = TerminalBenchResolveRateMetric(spec)
    context = MetricContext(
        sample_id="tb2__1",
        sample={},
        model_output={},
        judge_output={
            "status": "passed",
            "score": 1.0,
            "summary": "Terminal benchmark requirements satisfied.",
            "raw_output": {"resolved": True, "failure_reason": None},
        },
        args=spec.params,
        trace=mock_trace,
    )

    result = metric.compute(context)

    assert result.values["resolve_rate"] == 1.0
    assert result.metadata["resolved"] is True
    assert result.metadata["failure_reason"] is None


def test_terminal_bench_failure_reason_reads_runtime_eval_result(mock_trace) -> None:
    spec = MetricSpec(
        metric_id="terminal_bench_failure_reason",
        implementation="terminal_bench_failure_reason",
        aggregation="categorical_count",
        params={},
    )
    metric = TerminalBenchFailureReasonMetric(spec)
    context = MetricContext(
        sample_id="tb2__2",
        sample={},
        model_output={},
        judge_output={
            "status": "failed",
            "score": 0.0,
            "summary": "Missing required surfaces: terminal",
            "raw_output": {"resolved": False, "failure_reason": "missing_required_surfaces"},
        },
        args=spec.params,
        trace=mock_trace,
    )

    result = metric.compute(context)

    assert result.values["count"] == 1.0
    assert result.metadata["resolved"] is False
    assert result.metadata["failure_reason"] == "missing_required_surfaces"
