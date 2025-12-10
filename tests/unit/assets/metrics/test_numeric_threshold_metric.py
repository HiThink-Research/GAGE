from gage_eval.metrics.base import MetricContext
from gage_eval.metrics.builtin.text import NumericMatchMetric, JudgeThresholdMetric
from gage_eval.config.pipeline_config import MetricSpec
from gage_eval.observability.trace import ObservabilityTrace


def make_ctx(sample=None, model_output=None, judge_output=None):
    return MetricContext(
        sample_id="s1",
        sample=sample or {},
        model_output=model_output or {},
        judge_output=judge_output or {},
        args={},
        trace=ObservabilityTrace(),
    )


def test_numeric_match_within_tolerance():
    spec = MetricSpec(metric_id="num", implementation="numeric_match", params={"tolerance": 0.2})
    metric = NumericMatchMetric(spec)
    ctx = make_ctx(sample={"label": 1.0}, model_output={"answer": 1.1}, judge_output={})
    result = metric.compute(ctx)
    assert result.values["score"] == 1.0
    assert result.metadata["prediction"] == 1.1
    assert result.metadata["reference"] == 1.0
    assert result.metadata["tolerance"] == 0.2


def test_numeric_match_invalid_format_marks_metadata():
    spec = MetricSpec(metric_id="num2", implementation="numeric_match", params={"tolerance": 0.1})
    metric = NumericMatchMetric(spec)
    ctx = make_ctx(sample={"label": "x"}, model_output={"answer": "abc"})
    result = metric.compute(ctx)
    assert result.values["score"] == 0.0
    assert result.metadata["invalid_format"] is True


def test_judge_threshold_ge_mode_and_fallback():
    spec = MetricSpec(metric_id="judge", implementation="judge_threshold", params={"threshold": 0.5, "fallback": 0.0})
    metric = JudgeThresholdMetric(spec)
    ctx = make_ctx(judge_output={"score": 0.6})
    result = metric.compute(ctx)
    assert result.values["score"] == 1.0
    assert result.metadata["reference"] == 0.5

    ctx_missing = make_ctx(judge_output={})
    missing_result = metric.compute(ctx_missing)
    assert missing_result.values["score"] == 0.0
    assert missing_result.metadata["invalid_format"] is True
