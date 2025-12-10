from gage_eval.metrics.base import MetricContext
from gage_eval.observability.trace import ObservabilityTrace


def test_metric_context_get_uses_unified_extractor():
    ctx = MetricContext(
        sample_id="s2",
        sample={"label": "A", "nested": {"list": [10, 20]}},
        model_output={"answer": {"value": "B"}},
        judge_output={},
        args={},
        trace=ObservabilityTrace(),
    )

    assert ctx.get("sample.nested.list.1") == 20
    assert ctx.get("model_output.answer.value") == "B"
    # fallback到 sample 根
    assert ctx.get("nested.list.0") == 10
    assert ctx.get("missing.path", default="fallback") == "fallback"
