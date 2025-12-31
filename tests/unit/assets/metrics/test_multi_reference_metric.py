from gage_eval.metrics.base import MetricContext, MultiReferenceTextMetric
from gage_eval.config.pipeline_config import MetricSpec
from gage_eval.observability.trace import ObservabilityTrace


class DummyMultiRefMetric(MultiReferenceTextMetric):
    """Test-only multi-reference metric using default equality matching."""
    pass


def make_ctx(sample_refs, answer, params=None):
    spec = MetricSpec(metric_id="mref", implementation="mref_impl", params=params or {})
    metric = DummyMultiRefMetric(spec)
    ctx = MetricContext(
        sample_id="s1",
        sample={"references": sample_refs},
        model_output={"answer": answer},
        judge_output={},
        args=spec.params,
        trace=ObservabilityTrace(),
    )
    return metric, ctx


def test_multi_reference_best_score_and_metadata():
    metric, ctx = make_ctx(["Paris", "London"], "paris")
    result = metric.compute(ctx)
    assert result.values["score"] == 1.0
    assert result.metadata["best_reference"] == "paris"
    assert result.metadata["references"] == ["paris", "london"]


def test_multi_reference_separator_split():
    metric, ctx = make_ctx("A;B", "b", params={"separator": ";", "case_sensitive": False})
    result = metric.compute(ctx)
    assert result.values["score"] == 1.0
    assert result.metadata["best_reference"] == "b"
    assert result.metadata["references"] == ["a", "b"]


def test_multi_reference_empty_refs_warns_and_zero():
    metric, ctx = make_ctx([], "anything")
    result = metric.compute(ctx)
    assert result.values["score"] == 0.0
    assert result.metadata["warning"] == "empty_refs_or_pred"
