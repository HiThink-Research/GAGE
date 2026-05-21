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
    assert out.values["average_model_brier_market_subset"] == pytest.approx(0.1)
    assert out.values["average_model_minus_market_brier"] == pytest.approx(0.06)
    assert out.metadata["market_baseline_samples"] == 1
    assert out.metadata["market_baseline_coverage"] == pytest.approx(0.5)
    assert out.metadata["market_baseline_partial_coverage"] is True


def test_forecastbench_aggregator_model_market_delta_uses_same_subset() -> None:
    spec = MetricSpec(
        metric_id="fb_agg3",
        implementation="forecastbench_probability",
        aggregation="forecastbench_probability_summary",
        params={},
    )
    agg = ForecastBenchProbabilitySummaryAggregator(spec)
    agg.add(
        MetricResult(
            sample_id="market-subset",
            values={
                "brier": 0.30,
                "accuracy_at_0_5": 0.0,
                "parse_error": 0.0,
                "abs_error": 0.3,
                "clamp_applied": 0.0,
                "market_baseline_brier": 0.10,
            },
        )
    )
    agg.add(
        MetricResult(
            sample_id="no-market-baseline",
            values={
                "brier": 0.02,
                "accuracy_at_0_5": 1.0,
                "parse_error": 0.0,
                "abs_error": 0.02,
                "clamp_applied": 0.0,
            },
        )
    )

    out = agg.finalize()

    assert out.values["average_brier"] == pytest.approx(0.16)
    assert out.values["average_market_baseline_brier"] == pytest.approx(0.10)
    assert out.values["average_model_brier_market_subset"] == pytest.approx(0.30)
    assert out.values["average_model_minus_market_brier"] == pytest.approx(0.20)
