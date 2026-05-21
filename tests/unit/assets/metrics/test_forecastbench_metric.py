"""Tests for ForecastBench probability metric."""

from __future__ import annotations

import pytest

from gage_eval.config.pipeline_config import MetricSpec
from gage_eval.metrics.builtin.forecastbench import ForecastBenchProbabilityMetric
from gage_eval.metrics.base import MetricContext
from gage_eval.observability.trace import ObservabilityTrace


def _ctx(
    *,
    sample_id: str,
    references: list,
    model_text: str,
    metadata: dict | None = None,
) -> MetricContext:
    sample = {
        "id": sample_id,
        "references": references,
        "label": str(references[0]) if references else "",
        "metadata": metadata or {},
        "predict_result": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": [{"type": "text", "text": model_text}],
                },
            }
        ],
    }
    return MetricContext(
        sample_id=sample_id,
        sample=sample,
        model_output={},
        judge_output={},
        args={},
        trace=ObservabilityTrace(),
    )


def test_forecastbench_metric_basic_brier() -> None:
    spec = MetricSpec(metric_id="fb", implementation="forecastbench_probability", params={})
    metric = ForecastBenchProbabilityMetric(spec)
    ctx = _ctx(sample_id="s1", references=[0], model_text='{"forecast": 0.3, "reasoning": "x"}')
    result = metric.compute(ctx)
    assert result.values["brier"] == pytest.approx(0.09)
    assert result.values["abs_error"] == pytest.approx(0.3)
    assert result.values["accuracy_at_0_5"] == 1.0
    assert result.values["parse_error"] == 0.0
    assert result.values["forecast"] == pytest.approx(0.3)


def test_forecastbench_metric_parses_official_starred_probability() -> None:
    spec = MetricSpec(metric_id="fb", implementation="forecastbench_probability", params={})
    metric = ForecastBenchProbabilityMetric(spec)
    ctx = _ctx(sample_id="s-star", references=[0], model_text="*0.075*")
    result = metric.compute(ctx)
    assert result.values["forecast"] == pytest.approx(0.075)
    assert result.values["parse_error"] == 0.0
    assert result.values["brier"] == pytest.approx(0.075**2)


def test_forecastbench_metric_parses_plain_numeric_probability() -> None:
    spec = MetricSpec(metric_id="fb", implementation="forecastbench_probability", params={})
    metric = ForecastBenchProbabilityMetric(spec)
    ctx = _ctx(sample_id="s-num", references=[0], model_text="0.42")
    result = metric.compute(ctx)
    assert result.values["forecast"] == pytest.approx(0.42)
    assert result.values["parse_error"] == 0.0
    assert result.values["brier"] == pytest.approx(0.42**2)


def test_forecastbench_metric_parse_failure_fallback() -> None:
    spec = MetricSpec(metric_id="fb", implementation="forecastbench_probability", params={})
    metric = ForecastBenchProbabilityMetric(spec)
    ctx = _ctx(sample_id="s2", references=[1], model_text="not json at all")
    result = metric.compute(ctx)
    assert result.values["forecast"] == pytest.approx(0.5)
    assert result.values["parse_error"] == 1.0
    assert result.values["brier"] == pytest.approx(0.25)


def test_forecastbench_metric_clamp() -> None:
    spec = MetricSpec(metric_id="fb", implementation="forecastbench_probability", params={})
    metric = ForecastBenchProbabilityMetric(spec)
    ctx = _ctx(sample_id="s3", references=[1], model_text='{"forecast": 1.7, "reasoning": "x"}')
    result = metric.compute(ctx)
    assert result.values["forecast"] == pytest.approx(1.0)
    assert result.values["clamp_applied"] == 1.0
    assert result.values["brier"] == pytest.approx(0.0)


def test_forecastbench_metric_market_baseline_null_safe() -> None:
    spec = MetricSpec(metric_id="fb", implementation="forecastbench_probability", params={})
    metric = ForecastBenchProbabilityMetric(spec)
    ctx = _ctx(
        sample_id="s4",
        references=[0.5],
        model_text='{"forecast": 0.5, "reasoning": "x"}',
        metadata={"freeze_datetime_value": None},
    )
    result = metric.compute(ctx)
    assert "market_baseline_brier" not in result.values


def test_forecastbench_metric_last_json_object_fallback() -> None:
    spec = MetricSpec(metric_id="fb", implementation="forecastbench_probability", params={})
    metric = ForecastBenchProbabilityMetric(spec)
    text = 'noise then {"forecast": 0.2, "reasoning": "a"} trailing'
    ctx = _ctx(sample_id="s5", references=[1], model_text=text)
    result = metric.compute(ctx)
    assert result.values["parse_error"] == 0.0
    assert result.values["forecast"] == pytest.approx(0.2)


def test_forecastbench_metric_json_fallback_handles_braces_inside_strings() -> None:
    spec = MetricSpec(metric_id="fb", implementation="forecastbench_probability", params={})
    metric = ForecastBenchProbabilityMetric(spec)
    text = 'noise then {"reasoning": "see prompt }} above", "forecast": 0.7} trailing'
    ctx = _ctx(sample_id="s-json-braces", references=[1], model_text=text)
    result = metric.compute(ctx)
    assert result.values["parse_error"] == 0.0
    assert result.values["forecast"] == pytest.approx(0.7)


def test_forecastbench_metric_json_fallback_prefers_outer_object_over_nested() -> None:
    spec = MetricSpec(metric_id="fb", implementation="forecastbench_probability", params={})
    metric = ForecastBenchProbabilityMetric(spec)
    text = 'noise {"analysis": {"forecast": "bad"}, "forecast": 0.7} trailing'
    ctx = _ctx(sample_id="s-json-nested", references=[1], model_text=text)
    result = metric.compute(ctx)
    assert result.values["parse_error"] == 0.0
    assert result.values["forecast"] == pytest.approx(0.7)


def test_forecastbench_metric_prefers_structured_json_forecast_over_starred_reasoning() -> None:
    spec = MetricSpec(metric_id="fb", implementation="forecastbench_probability", params={})
    metric = ForecastBenchProbabilityMetric(spec)
    text = '{"reasoning": "market was *0.9* before news changed", "forecast": 0.30}'
    ctx = _ctx(sample_id="s-json-priority", references=[0], model_text=text)
    result = metric.compute(ctx)
    assert result.values["parse_error"] == 0.0
    assert result.values["forecast"] == pytest.approx(0.30)
    assert result.values["brier"] == pytest.approx(0.09)


def test_forecastbench_metric_starred_out_of_range_is_clamped_not_parse_error() -> None:
    spec = MetricSpec(metric_id="fb", implementation="forecastbench_probability", params={})
    metric = ForecastBenchProbabilityMetric(spec)
    ctx = _ctx(sample_id="s-star-clamp", references=[1], model_text="*1.5*")
    result = metric.compute(ctx)
    assert result.values["parse_error"] == 0.0
    assert result.values["clamp_applied"] == 1.0
    assert result.values["forecast"] == pytest.approx(1.0)


def test_forecastbench_metric_missing_reference_raises() -> None:
    spec = MetricSpec(metric_id="fb", implementation="forecastbench_probability", params={})
    metric = ForecastBenchProbabilityMetric(spec)
    ctx = MetricContext(
        sample_id="bad",
        sample={"id": "bad", "references": [], "metadata": {}},
        model_output={},
        judge_output={},
        args={},
        trace=ObservabilityTrace(),
    )
    with pytest.raises(ValueError, match="missing a resolved numeric reference"):
        metric.compute(ctx)


def test_forecastbench_metric_non_numeric_reference_raises() -> None:
    spec = MetricSpec(metric_id="fb", implementation="forecastbench_probability", params={})
    metric = ForecastBenchProbabilityMetric(spec)
    ctx = MetricContext(
        sample_id="bad2",
        sample={"id": "bad2", "references": ["not-a-number"], "metadata": {}},
        model_output={},
        judge_output={},
        args={},
        trace=ObservabilityTrace(),
    )
    with pytest.raises(ValueError, match="cannot convert reference"):
        metric.compute(ctx)
