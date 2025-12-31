from gage_eval.metrics.base import MetricContext, SequenceDistanceMetric
from gage_eval.config.pipeline_config import MetricSpec
from gage_eval.observability.trace import ObservabilityTrace


class DummySequenceDistanceMetric(SequenceDistanceMetric):
    """Test-only metric that reuses the base compare implementation."""
    pass


def make_ctx(label: str, answer: str, params: dict | None = None) -> tuple[DummySequenceDistanceMetric, MetricContext]:
    spec = MetricSpec(metric_id="seq", implementation="seq_impl", params=params or {})
    metric = DummySequenceDistanceMetric(spec)
    ctx = MetricContext(
        sample_id="s1",
        sample={"label": label},
        model_output={"answer": answer},
        judge_output={},
        args=spec.params,
        trace=ObservabilityTrace(),
    )
    return metric, ctx


def test_sequence_distance_anls_exact_match():
    metric, ctx = make_ctx(label="hello", answer="hello")
    result = metric.compute(ctx)
    assert result.values["score"] == 1.0
    assert result.metadata["distance"] == 0
    assert result.metadata["strategy"] == "anls"


def test_sequence_distance_wer_clamped():
    metric, ctx = make_ctx(label="a", answer="abc", params={"normalize": "wer"})
    result = metric.compute(ctx)
    assert result.values["score"] == 0.0
    assert result.metadata["distance"] == 2
    assert result.metadata["strategy"] == "wer"


def test_sequence_distance_raw_returns_distance():
    metric, ctx = make_ctx(label="kitten", answer="sitting", params={"normalize": "raw"})
    result = metric.compute(ctx)
    assert result.values["score"] == 3.0
    assert result.metadata["distance"] == 3
    assert result.metadata["strategy"] == "raw"
