from gage_eval.metrics.base import MetricContext
from gage_eval.config.pipeline_config import MetricSpec
from gage_eval.observability.trace import ObservabilityTrace
from gage_eval.metrics.builtin.text import (
    ContainsMatchMetric,
    NumericMatchMetric,
    RegexMatchMetric,
    JudgeThresholdMetric,
    TextLengthMetric,
    LatencyMetric,
)


def _ctx(sample=None, model=None, judge=None):
    return MetricContext(
        sample_id="sample",
        sample=sample or {},
        model_output=model or {},
        judge_output=judge or {},
        args={},
        trace=ObservabilityTrace(),
    )


def test_contains_metadata():
    spec = MetricSpec(metric_id="contains", implementation="contains")
    metric = ContainsMatchMetric(spec)
    result = metric.compute(_ctx(sample={"label": "abc"}, model={"answer": "xxabcxx"}))
    assert result.values["score"] == 1.0
    assert result.metadata["prediction"] == "xxabcxx"
    assert result.metadata["reference"] == "abc"


def test_numeric_metadata():
    spec = MetricSpec(metric_id="num", implementation="numeric_match", params={"tolerance": 0.2})
    metric = NumericMatchMetric(spec)
    result = metric.compute(_ctx(sample={"label": 1.0}, model={"answer": 1.1}))
    assert result.values["score"] == 1.0
    assert result.metadata["tolerance"] == 0.2


def test_regex_metadata():
    spec = MetricSpec(metric_id="regex", implementation="regex_match", params={"pattern": "\\d+"})
    metric = RegexMatchMetric(spec)
    result = metric.compute(_ctx(model={"answer": "test123"}))
    assert result.metadata["target"] == "test123"


def test_judge_threshold_metadata():
    spec = MetricSpec(metric_id="judge", implementation="judge_threshold", params={"threshold": 0.5})
    metric = JudgeThresholdMetric(spec)
    result = metric.compute(_ctx(judge={"score": 0.6}))
    assert result.metadata["judge"] == 0.6
    assert result.metadata["threshold"] == 0.5


def test_text_length_metadata():
    spec = MetricSpec(metric_id="len", implementation="text_length", params={"unit": "word"})
    metric = TextLengthMetric(spec)
    result = metric.compute(_ctx(model={"answer": "one two"}))
    assert result.values["length"] == 2.0
    assert result.metadata["unit"] == "word"
    assert result.metadata["target"] == "one two"


def test_latency_metadata():
    spec = MetricSpec(metric_id="latency", implementation="latency", params={"target_field": "model_output.latency_ms"})
    metric = LatencyMetric(spec)
    result = metric.compute(_ctx(model={"latency_ms": 123}))
    assert result.values["latency_ms"] == 123.0
    assert result.metadata["target_field"] == "model_output.latency_ms"
