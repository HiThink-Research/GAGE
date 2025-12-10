from gage_eval.metrics.base import ComparisonMetric, MetricContext
from gage_eval.config.pipeline_config import MetricSpec
from gage_eval.observability.trace import ObservabilityTrace


class DummyComparisonMetric(ComparisonMetric):
    value_key = "acc"

    def compare(self, prediction, reference):
        score = 1.0 if prediction == reference else 0.0
        return score, {"note": "checked"}


def test_comparison_metric_compute_includes_metadata():
    spec = MetricSpec(metric_id="cmp", implementation="cmp_impl")
    metric = DummyComparisonMetric(spec)
    ctx = MetricContext(
        sample_id="s3",
        sample={"label": "A"},
        model_output={"answer": "A"},
        judge_output={},
        args={},
        trace=ObservabilityTrace(),
    )
    result = metric.compute(ctx)
    assert result.values["acc"] == 1.0
    assert result.metadata["prediction"] == "A"
    assert result.metadata["reference"] == "A"
    assert result.metadata["note"] == "checked"


def test_comparison_metric_supports_field_overrides():
    spec = MetricSpec(
        metric_id="cmp2",
        implementation="cmp_impl",
        params={"prediction_field": "model_output.pred", "label_field": "sample.gold"},
    )
    metric = DummyComparisonMetric(spec)
    ctx = MetricContext(
        sample_id="s4",
        sample={"gold": "X"},
        model_output={"pred": "Y"},
        judge_output={},
        args={},
        trace=ObservabilityTrace(),
    )
    result = metric.compute(ctx)
    assert result.metadata["prediction"] == "Y"
    assert result.metadata["reference"] == "X"
    assert result.values["acc"] == 0.0


def test_exact_match_inherits_comparison_defaults():
    # ensure default fields and compare logic still work
    from gage_eval.metrics.builtin.text import ExactMatchMetric

    spec = MetricSpec(metric_id="exact_match", implementation="exact_match")
    metric = ExactMatchMetric(spec)
    ctx = MetricContext(
        sample_id="s5",
        sample={"label": "hello"},
        model_output={"answer": "hello"},
        judge_output={},
        args={},
        trace=ObservabilityTrace(),
    )
    result = metric.compute(ctx)
    assert result.values["score"] == 1.0
