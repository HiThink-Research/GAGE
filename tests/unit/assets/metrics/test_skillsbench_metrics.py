from __future__ import annotations

from gage_eval.config.pipeline_config import MetricSpec
from gage_eval.metrics.base import MetricContext
from gage_eval.metrics.builtin.skillsbench import (
    SkillsBenchFailureReasonMetric,
    SkillsBenchResolveRateMetric,
)


def test_skillsbench_resolve_rate_reads_runtime_eval_result(mock_trace) -> None:
    spec = MetricSpec(metric_id="skillsbench_resolve_rate", implementation="skillsbench_resolve_rate", params={})
    metric = SkillsBenchResolveRateMetric(spec)
    context = MetricContext(
        sample_id="skillsbench__1",
        sample={},
        model_output={},
        judge_output={
            "status": "pass",
            "score": 1.0,
            "summary": "resolved",
            "raw_output": {"resolved": True, "failure_reason": None},
        },
        args=spec.params,
        trace=mock_trace,
    )

    result = metric.compute(context)

    assert result.values["resolve_rate"] == 1.0
    assert result.metadata["resolved"] is True
    assert result.metadata["failure_reason"] is None


def test_skillsbench_failure_reason_reads_runtime_eval_result(mock_trace) -> None:
    spec = MetricSpec(
        metric_id="skillsbench_failure_reason",
        implementation="skillsbench_failure_reason",
        aggregation="categorical_count",
        params={},
    )
    metric = SkillsBenchFailureReasonMetric(spec)
    context = MetricContext(
        sample_id="skillsbench__2",
        sample={},
        model_output={},
        judge_output={
            "status": "fail",
            "score": 0.0,
            "summary": "missing_agent_workspace",
            "raw_output": {"resolved": False, "failure_reason": "missing_agent_workspace"},
        },
        args=spec.params,
        trace=mock_trace,
    )

    result = metric.compute(context)

    assert result.values["count"] == 1.0
    assert result.metadata["resolved"] is False
    assert result.metadata["failure_reason"] == "missing_agent_workspace"
