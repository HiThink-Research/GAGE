from gage_eval.config.pipeline_config import MetricSpec
from gage_eval.metrics import MetricRegistry, MetricContext
from gage_eval.observability.trace import ObservabilityTrace


def test_comparison_metric_metadata_includes_reference():
    spec = MetricSpec(metric_id="numeric_match", implementation="numeric_match", aggregation="mean")
    registry = MetricRegistry()
    metric = registry.build_metric(spec)

    sample = {"id": "s1", "label": 3}
    model_output = {"answer": 3}
    trace = ObservabilityTrace()
    ctx = MetricContext(
        sample_id="s1",
        sample=sample,
        model_output=model_output,
        judge_output={},
        args=metric.spec.params,
        trace=trace,
    )
    res = metric.evaluate(ctx)
    assert res.metadata["prediction"] == 3
    assert res.metadata["reference"] == 3
    aggregated = metric.finalize()
    assert aggregated["values"]["score"] == 1.0
    assert aggregated["count"] == 1
