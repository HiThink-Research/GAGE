from gage_eval.config.pipeline_config import MetricSpec
from gage_eval.metrics import MetricRegistry, MetricContext
from gage_eval.observability.trace import ObservabilityTrace


def _build_metric(metric_id: str, params: dict | None = None, aggregation: str | None = None):
    spec = MetricSpec(metric_id=metric_id, implementation=metric_id, aggregation=aggregation, params=params or {})
    registry = MetricRegistry()
    return registry.build_metric(spec)


def test_exact_match_case_sensitivity():
    insensitive = _build_metric("exact_match", params={"case_sensitive": False})
    sensitive = _build_metric("exact_match", params={"case_sensitive": True})
    sample = {"label": "Answer"}
    output = {"answer": "answer"}
    trace = ObservabilityTrace()

    ctx = MetricContext("s1", sample, output, {}, insensitive.spec.params, trace)
    ins_res = insensitive.evaluate(ctx)
    ctx2 = MetricContext("s1", sample, output, {}, sensitive.spec.params, trace)
    sen_res = sensitive.evaluate(ctx2)

    assert ins_res.values["score"] == 1.0
    assert sen_res.values["score"] == 0.0


def test_regex_match_extracts_option_letter():
    metric = _build_metric("regex_match", params={"pattern": r"\(([A-D])\)"})
    sample = {"id": "s1"}
    output = {"answer": "The answer is (A)."}
    trace = ObservabilityTrace()
    ctx = MetricContext("s1", sample, output, {}, metric.spec.params, trace)

    res = metric.evaluate(ctx)
    assert res.values["score"] == 1.0
    assert res.metadata["target"] == "The answer is (A)."


def test_multi_reference_prefers_best_match():
    metric = _build_metric("docvqa_anls")
    sample = {"metadata": {"answers": ["A", "B"]}}
    output = {"answer": "B"}
    trace = ObservabilityTrace()
    ctx = MetricContext("s1", sample, output, {}, metric.spec.params, trace)

    res = metric.evaluate(ctx)
    assert res.values["anls"] == 1.0
    agg = metric.finalize()
    assert agg["values"]["anls"] == 1.0
