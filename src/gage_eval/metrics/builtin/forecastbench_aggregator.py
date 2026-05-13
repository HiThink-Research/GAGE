"""ForecastBench run-level aggregation for probability metrics."""

from __future__ import annotations

import math
from typing import Optional

from loguru import logger

from gage_eval.config.pipeline_config import MetricSpec
from gage_eval.metrics.aggregators import MetricAggregator
from gage_eval.metrics.base import AggregatedMetric, MetricResult
from gage_eval.metrics.runtime_context import AggregationRuntimeContext


class ForecastBenchProbabilitySummaryAggregator(MetricAggregator):
    """Aggregate per-sample ForecastBench probability scores into run-level summaries."""

    def __init__(
        self,
        spec: MetricSpec,
        runtime_context: Optional[AggregationRuntimeContext] = None,
    ) -> None:
        super().__init__(spec, runtime_context=runtime_context)
        self._count = 0
        self._sum_brier = 0.0
        self._sum_accuracy = 0.0
        self._sum_parse_error = 0.0
        self._sum_abs_error = 0.0
        self._sum_clamp = 0.0
        self._sum_market_baseline = 0.0
        self._count_market_baseline = 0

    def add(self, result: MetricResult) -> None:
        values = result.values
        self._count += 1
        self._sum_brier += float(values.get("brier", 0.0))
        self._sum_accuracy += float(values.get("accuracy_at_0_5", 0.0))
        self._sum_parse_error += float(values.get("parse_error", 0.0))
        self._sum_abs_error += float(values.get("abs_error", 0.0))
        self._sum_clamp += float(values.get("clamp_applied", 0.0))
        mb = values.get("market_baseline_brier")
        if mb is not None:
            self._sum_market_baseline += float(mb)
            self._count_market_baseline += 1

    def finalize(self) -> AggregatedMetric:
        count = self._count
        if count <= 0:
            average_brier = 0.0
            values_out = {
                "average_brier": 0.0,
                "brier_index_simple": 0.0,
                "accuracy_at_0_5": 0.0,
                "parse_error_rate": 0.0,
                "avg_abs_error": 0.0,
                "clamp_rate": 0.0,
            }
            meta = {"samples": 0}
            return AggregatedMetric(
                metric_id=self.spec.metric_id,
                aggregation=self.spec.aggregation or "forecastbench_probability_summary",
                values=values_out,
                count=0,
                metadata=meta,
            )

        average_brier = self._sum_brier / count
        brier_index_simple = (1.0 - math.sqrt(max(0.0, average_brier))) * 100.0

        values_out = {
            "average_brier": float(average_brier),
            "brier_index_simple": float(brier_index_simple),
            "accuracy_at_0_5": float(self._sum_accuracy / count),
            "parse_error_rate": float(self._sum_parse_error / count),
            "avg_abs_error": float(self._sum_abs_error / count),
            "clamp_rate": float(self._sum_clamp / count),
        }

        meta: dict = {"samples": count}
        if self._count_market_baseline > 0:
            avg_mb = self._sum_market_baseline / self._count_market_baseline
            values_out["average_market_baseline_brier"] = float(avg_mb)
            meta["market_baseline_samples"] = self._count_market_baseline

        logger.debug(
            "ForecastBenchProbabilitySummaryAggregator finalized metric={} samples={} average_brier={:.6f}",
            self.spec.metric_id,
            count,
            average_brier,
        )

        return AggregatedMetric(
            metric_id=self.spec.metric_id,
            aggregation=self.spec.aggregation or "forecastbench_probability_summary",
            values=values_out,
            count=count,
            metadata=meta,
        )


__all__ = ["ForecastBenchProbabilitySummaryAggregator"]
