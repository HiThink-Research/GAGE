import pytest

from gage_eval.metrics.builtin.mathvista import MathVistaAccuracyMetric
from gage_eval.metrics.base import MetricContext
from gage_eval.config.pipeline_config import MetricSpec


def build_context(sample, model_output) -> MetricContext:
    return MetricContext(
        sample_id="s1",
        sample=sample,
        model_output=model_output,
        judge_output={},
        args={},
        trace=None,
    )


@pytest.mark.fast
def test_mathvista_accuracy_multi_choice_letter_match():
    metric = MathVistaAccuracyMetric(MetricSpec(metric_id="m1", implementation="mathvista_accuracy"))
    sample = {
        "metadata": {"option_map": {"A": "red", "B": "green"}},
        "choices": [
            {"label": "A", "message": {"content": [{"type": "text", "text": "red"}]}},
            {"label": "B", "message": {"content": [{"type": "text", "text": "green"}]}},
        ],
        "answer": "B",
    }
    ctx = build_context(sample, {"answer": "B"})
    res = metric.compute(ctx)
    assert res.values["acc"] == 1.0
    assert res.metadata["prediction_label"] == "B"
    assert res.metadata["expected_label"] == "B"


@pytest.mark.fast
def test_mathvista_accuracy_multi_choice_text_match():
    metric = MathVistaAccuracyMetric(MetricSpec(metric_id="m1", implementation="mathvista_accuracy"))
    sample = {
        "metadata": {"option_map": {"A": "red", "B": "green"}},
        "answer": "green",  # 文本答案
    }
    ctx = build_context(sample, {"answer": "green"})
    res = metric.compute(ctx)
    assert res.values["acc"] == 1.0
    assert res.metadata["expected_label"] == "B"


@pytest.mark.fast
def test_mathvista_accuracy_open_ended_exact_match():
    metric = MathVistaAccuracyMetric(MetricSpec(metric_id="m1", implementation="mathvista_accuracy"))
    sample = {"answer": "42"}
    ctx = build_context(sample, {"answer": "42"})
    res = metric.compute(ctx)
    assert res.values["acc"] == 1.0
    ctx2 = build_context(sample, {"answer": "43"})
    res2 = metric.compute(ctx2)
    assert res2.values["acc"] == 0.0


@pytest.mark.fast
def test_mathvista_accuracy_numeric_extraction_integer():
    metric = MathVistaAccuracyMetric(MetricSpec(metric_id="m1", implementation="mathvista_accuracy"))
    sample = {"answer": "1000", "answer_type": "integer"}
    ctx = build_context(sample, {"answer": "The total volume is 1000 milliliters (ml)."})
    res = metric.compute(ctx)
    assert res.values["acc"] == 1.0
    assert res.metadata["extracted_prediction"] == "1000"
    assert res.metadata["reference"] == "1000"


@pytest.mark.fast
def test_mathvista_accuracy_numeric_extraction_float():
    metric = MathVistaAccuracyMetric(MetricSpec(metric_id="m1", implementation="mathvista_accuracy"))
    sample = {"answer": 3.14, "answer_type": "float"}
    ctx = build_context(sample, {"answer": "approx 3.140 in total"})
    res = metric.compute(ctx)
    assert res.values["acc"] == 1.0
    assert res.metadata["extracted_prediction"] == "3.14"
    assert res.metadata["reference"] == "3.14"
