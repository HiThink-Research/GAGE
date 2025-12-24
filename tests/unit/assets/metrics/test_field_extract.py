import pytest

from gage_eval.metrics.base import MetricContext
from gage_eval.metrics.utils import extract_field
from gage_eval.observability.trace import ObservabilityTrace


def make_context(sample: dict) -> MetricContext:
    return MetricContext(
        sample_id="s1",
        sample=sample,
        model_output={"answer": "yes", "choices": [{"text": "A"}]},
        judge_output={"score": 0.9},
        args={},
        trace=ObservabilityTrace(),
    )


def test_extract_field_supports_mapping_and_lists():
    data = {"a": {"b": [{"c": 1}, {"c": 2}]}}
    assert extract_field(data, "a.b.1.c") == 2
    assert extract_field(data, "a.b.10.c", default="missing") == "missing"
    assert extract_field(data, None, default="none") == "none"


def test_extract_field_with_metric_context_roots_and_fallback():
    ctx = make_context({"label": "Paris", "arr": [{"v": 3}]})
    # explicit root
    assert extract_field(ctx, "sample.arr.0.v") == 3
    # fall back to sample when the root is not specified
    assert extract_field(ctx, "arr.0.v") == 3
    # model_output root
    assert extract_field(ctx, "model_output.answer") == "yes"
    # missing path returns default
    assert extract_field(ctx, "sample.arr.2.v", default=None) is None
