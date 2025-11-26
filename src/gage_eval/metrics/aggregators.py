"""聚合器实现。"""

from __future__ import annotations

from collections import defaultdict
from typing import Dict, List

from loguru import logger
from gage_eval.config.pipeline_config import MetricSpec
from gage_eval.metrics.base import AggregatedMetric, MetricResult


class MetricAggregator:
    """所有聚合器的基础类。"""

    def __init__(self, spec: MetricSpec) -> None:
        self.spec = spec

    def add(self, result: MetricResult) -> None:  # pragma: no cover
        raise NotImplementedError

    def finalize(self) -> AggregatedMetric:  # pragma: no cover
        raise NotImplementedError


class MeanAggregator(MetricAggregator):
    """对每个 value key 做均值。"""

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
    """尊重样本权重的均值聚合器，weight 来自 MetricResult.metadata.weight。"""

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
    """直接返回所有样本值，主要用于调试或上层定制聚合。"""

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
