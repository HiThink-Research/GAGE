"""Metric aggregator implementations."""

from __future__ import annotations

from collections import defaultdict
from typing import Dict, List

from loguru import logger
from gage_eval.config.pipeline_config import MetricSpec
from gage_eval.metrics.base import AggregatedMetric, MetricResult


class MetricAggregator:
    """Base class for all aggregators."""

    def __init__(self, spec: MetricSpec) -> None:
        self.spec = spec

    def add(self, result: MetricResult) -> None:  # pragma: no cover
        raise NotImplementedError

    def finalize(self) -> AggregatedMetric:  # pragma: no cover
        raise NotImplementedError


class MeanAggregator(MetricAggregator):
    """Computes the mean for each metric value key."""

    def __init__(self, spec: MetricSpec) -> None:
        super().__init__(spec)
        self._sums: Dict[str, float] = defaultdict(float)
        self._counts: Dict[str, int] = defaultdict(int)
        self._total_results = 0

    def add(self, result: MetricResult) -> None:
        for key, value in result.values.items():
            self._sums[key] += float(value)
            self._counts[key] += 1
        self._total_results += 1

    def finalize(self) -> AggregatedMetric:
        values = {
            key: (self._sums[key] / self._counts[key]) if self._counts[key] else 0.0
            for key in self._sums
        }
        logger.debug(
            "MeanAggregator finalized metric={} samples={}",
            self.spec.metric_id,
            self._total_results,
        )
        return AggregatedMetric(
            metric_id=self.spec.metric_id,
            aggregation=self.spec.aggregation or "mean",
            values=values,
            count=self._total_results,
        )


class WeightedMeanAggregator(MetricAggregator):
    """Computes a weighted mean for each value key.

    The weight is read from `MetricResult.metadata["weight"]` (defaults to 1.0).
    """

    def __init__(self, spec: MetricSpec) -> None:
        super().__init__(spec)
        self._weighted_sums: Dict[str, float] = defaultdict(float)
        self._weight_totals: Dict[str, float] = defaultdict(float)
        self._total_samples = 0

    def add(self, result: MetricResult) -> None:
        weight = float(result.metadata.get("weight", 1.0))
        for key, value in result.values.items():
            self._weighted_sums[key] += float(value) * weight
            self._weight_totals[key] += weight
        self._total_samples += 1

    def finalize(self) -> AggregatedMetric:
        values = {
            key: (self._weighted_sums[key] / self._weight_totals[key]) if self._weight_totals[key] else 0.0
            for key in self._weighted_sums
        }
        logger.debug(
            "WeightedMeanAggregator finalized metric={} samples={}",
            self.spec.metric_id,
            self._total_samples,
        )
        return AggregatedMetric(
            metric_id=self.spec.metric_id,
            aggregation=self.spec.aggregation or "weighted_mean",
            values=values,
            count=self._total_samples,
        )


class IdentityAggregator(MetricAggregator):
    """Returns all per-sample values without aggregation.

    This is mostly useful for debugging or custom aggregations at higher layers.
    """

    def __init__(self, spec: MetricSpec) -> None:
        super().__init__(spec)
        self._results: List[MetricResult] = []

    def add(self, result: MetricResult) -> None:
        self._results.append(result)

    def finalize(self) -> AggregatedMetric:
        values = {str(idx): res.values for idx, res in enumerate(self._results)}
        logger.debug("IdentityAggregator captured {} samples for metric={}", len(self._results), self.spec.metric_id)
        return AggregatedMetric(
            metric_id=self.spec.metric_id,
            aggregation=self.spec.aggregation or "identity",
            values=values,
            count=len(self._results),
            metadata={"samples": [res.to_dict() for res in self._results]},
        )


class CategoricalCountAggregator(MetricAggregator):
    """Counts occurrences of a categorical field in per-sample metadata."""

    def __init__(self, spec: MetricSpec) -> None:
        super().__init__(spec)
        self._counts: Dict[str, int] = defaultdict(int)
        self._total = 0
        self._category_field = str(spec.params.get("category_field", "failure_reason"))
        self._include_none = bool(spec.params.get("include_none", False))
        self._none_label = str(spec.params.get("none_label", "unknown"))

    def add(self, result: MetricResult) -> None:
        category = result.metadata.get(self._category_field)
        if category is None:
            if not self._include_none:
                return
            category = self._none_label
        key = str(category)
        self._counts[key] += 1
        self._total += 1

    def finalize(self) -> AggregatedMetric:
        logger.debug(
            "CategoricalCountAggregator finalized metric={} samples={}",
            self.spec.metric_id,
            self._total,
        )
        return AggregatedMetric(
            metric_id=self.spec.metric_id,
            aggregation=self.spec.aggregation or "categorical_count",
            values=dict(self._counts),
            count=self._total,
        )


