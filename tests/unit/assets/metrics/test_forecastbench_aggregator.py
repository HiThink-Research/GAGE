"""Tests for ForecastBench probability aggregator."""

from __future__ import annotations

import pytest

from gage_eval.config.pipeline_config import MetricSpec
from gage_eval.metrics.base import MetricResult
from gage_eval.metrics.builtin.forecastbench_aggregator import ForecastBenchProbabilitySummaryAggregator


def test_forecastbench_aggregator_brier_index_formula() -> None:
    spec = MetricSpec(
        metric_id="fb_agg",
        implementation="forecastbench_probability",
        aggregation="forecastbench_probability_summary",
        params={},
    )
    agg = ForecastBenchProbabilitySummaryAggregator(spec)
    agg.add(
        MetricResult(
            sample_id="a",
            values={
                "brier": 0.25,
                "accuracy_at_0_5": 1.0,
                "parse_error": 0.0,
                "abs_error": 0.5,
                "clamp_applied": 0.0,
            },
        )
    )
    agg.add(
        MetricResult(
            sample_id="b",
            values={
                "brier": 0.25,
                "accuracy_at_0_5": 0.0,
                "parse_error": 0.0,
                "abs_error": 0.5,
                "clamp_applied": 0.0,
            },
        )
    )
    out = agg.finalize()
    assert out.count == 2
    assert out.values["average_brier"] == 0.25
    expected_index = (1.0 - (0.25**0.5)) * 100.0
    assert out.values["brier_index_simple"] == expected_index


def test_forecastbench_aggregator_market_baseline_partial() -> None:
    spec = MetricSpec(
        metric_id="fb_agg2",
        implementation="forecastbench_probability",
        aggregation="forecastbench_probability_summary",
        params={},
    )
    agg = ForecastBenchProbabilitySummaryAggregator(spec)
    agg.add(
        MetricResult(
            sample_id="a",
            values={
                "brier": 0.1,
                "accuracy_at_0_5": 1.0,
                "parse_error": 0.0,
                "abs_error": 0.1,
                "clamp_applied": 0.0,
                "market_baseline_brier": 0.04,
            },
        )
    )
    agg.add(
        MetricResult(
            sample_id="b",
            values={
                "brier": 0.2,
                "accuracy_at_0_5": 0.0,
                "parse_error": 0.0,
                "abs_error": 0.2,
                "clamp_applied": 0.0,
            },
        )
    )
    out = agg.finalize()
    assert out.values["average_market_baseline_brier"] == pytest.approx(0.04)
    assert out.metadata["market_baseline_samples"] == 1
