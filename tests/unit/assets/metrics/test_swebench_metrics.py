from __future__ import annotations

from gage_eval.metrics import MetricRegistry, MetricContext
from gage_eval.config.pipeline_config import MetricSpec
from gage_eval.observability.trace import ObservabilityTrace


def _ctx(judge_output):
    return MetricContext(
        sample_id="s1",
        sample={},
        model_output={},
        judge_output=judge_output,
        args={},
        trace=ObservabilityTrace(),
    )


def test_swebench_resolve_rate_metric():
    registry = MetricRegistry()
    instance = registry.build_metric(
        MetricSpec(metric_id="swebench_resolve_rate", implementation="swebench_resolve_rate")
    )

    res_ok = instance.evaluate(_ctx({"resolved": True}))
    res_fail = instance.evaluate(_ctx({"resolved": False, "failure_reason": "assertion_error"}))

    assert res_ok.values["resolve_rate"] == 1.0
    assert res_fail.values["resolve_rate"] == 0.0


def test_swebench_failure_reason_aggregation():
    registry = MetricRegistry()
    instance = registry.build_metric(
        MetricSpec(
            metric_id="swebench_failure_reason",
            implementation="swebench_failure_reason",
            aggregation="categorical_count",
        )
    )

    instance.evaluate(_ctx({"resolved": False, "failure_reason": "assertion_error"}))
    instance.evaluate(_ctx({"resolved": False}))
    instance.evaluate(_ctx({"resolved": True}))

    aggregated = instance.finalize()
    values = aggregated["values"]
    assert values["assertion_error"] == 1
    assert values["unknown"] == 1
    assert aggregated["count"] == 2
