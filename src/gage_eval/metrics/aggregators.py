"""Metric aggregator implementations."""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Dict, List, Optional

from loguru import logger
from gage_eval.config.pipeline_config import MetricSpec
from gage_eval.metrics.base import AggregatedMetric, MetricResult
from gage_eval.metrics.runtime_context import AggregationRuntimeContext


_DEFAULT_PREVIEW_ITEMS = 16
_DEFAULT_MAX_INLINE_ITEMS = 128


class MetricAggregator:
    """Base class for all aggregators."""

    def __init__(
        self,
        spec: MetricSpec,
        runtime_context: Optional[AggregationRuntimeContext] = None,
    ) -> None:
        self.spec = spec
        self.runtime_context = runtime_context

    def add(self, result: MetricResult) -> None:  # pragma: no cover
        raise NotImplementedError

    def finalize(self) -> AggregatedMetric:  # pragma: no cover
        raise NotImplementedError


class MeanAggregator(MetricAggregator):
    """Computes the mean for each metric value key."""

    def __init__(
        self,
        spec: MetricSpec,
        runtime_context: Optional[AggregationRuntimeContext] = None,
    ) -> None:
        super().__init__(spec, runtime_context=runtime_context)
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

    def __init__(
        self,
        spec: MetricSpec,
        runtime_context: Optional[AggregationRuntimeContext] = None,
    ) -> None:
        super().__init__(spec, runtime_context=runtime_context)
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

    def __init__(
        self,
        spec: MetricSpec,
        runtime_context: Optional[AggregationRuntimeContext] = None,
    ) -> None:
        super().__init__(spec, runtime_context=runtime_context)
        self._detail_mode = _normalize_detail_mode(spec.params.get("detail_mode"))
        self._preview_limit = _coerce_non_negative_int(
            spec.params.get("preview_items"),
            default=_DEFAULT_PREVIEW_ITEMS,
        )
        self._max_inline_items = _coerce_positive_int(
            spec.params.get("max_inline_items"),
            default=_DEFAULT_MAX_INLINE_ITEMS,
        )
        self._sample_count = 0
        self._preview_items: List[Dict[str, Any]] = []
        self._inline_items: Optional[List[Dict[str, Any]]] = (
            [] if self._should_track_inline_items() else None
        )
        self._inline_overflowed = False

    def add(self, result: MetricResult) -> None:
        result_dict = result.to_dict()
        self._sample_count += 1
        if len(self._preview_items) < self._preview_limit:
            self._preview_items.append(result_dict)
        if self._inline_items is not None:
            if len(self._inline_items) < self._max_inline_items:
                self._inline_items.append(result_dict)
            else:
                self._inline_overflowed = True

    def finalize(self) -> AggregatedMetric:
        preview_payload = list(self._preview_items)
        metadata = {
            "preview_count": len(preview_payload),
            "preview_truncated": self._sample_count > len(preview_payload),
        }
        if preview_payload:
            metadata["samples_preview"] = preview_payload

        storage_mode = "preview_only"
        values: Dict[str, Any] = {}

        if self._detail_mode == "spill":
            metadata["switch_reason"] = "spill_mode_not_implemented"
            metadata["error"] = {
                "code": "aggregation_detail_sink_unavailable",
                "message": "detail_mode='spill' is not available; falling back to preview_only",
            }
        elif self._should_use_cache_ref():
            storage_mode = "cache_ref"
            metadata.update(self._build_cache_ref_metadata())
        elif self._inline_items is not None and not self._inline_overflowed:
            storage_mode = "inline"
            values = {
                str(idx): item["values"]
                for idx, item in enumerate(self._inline_items)
            }
            metadata["samples"] = list(self._inline_items)
        else:
            metadata["switch_reason"] = self._resolve_preview_only_reason()

        metadata["storage_mode"] = storage_mode
        logger.debug(
            "IdentityAggregator finalized metric={} samples={} storage_mode={}",
            self.spec.metric_id,
            self._sample_count,
            storage_mode,
        )
        return AggregatedMetric(
            metric_id=self.spec.metric_id,
            aggregation=self.spec.aggregation or "identity",
            values=values,
            count=self._sample_count,
            metadata=metadata,
        )

    def _should_track_inline_items(self) -> bool:
        if self._detail_mode == "spill":
            return False
        if self._detail_mode == "cache_ref":
            return False
        if self._detail_mode == "auto" and self._has_cache_store():
            return False
        return True

    def _should_use_cache_ref(self) -> bool:
        if self._detail_mode == "inline_preview":
            return False
        if self._detail_mode == "spill":
            return False
        if self._detail_mode == "cache_ref":
            return self._has_cache_store()
        return self._has_cache_store()

    def _has_cache_store(self) -> bool:
        return (
            self.runtime_context is not None
            and self.runtime_context.cache_store is not None
        )

    def _build_cache_ref_metadata(self) -> Dict[str, Any]:
        if self.runtime_context is None or self.runtime_context.cache_store is None:
            return {}
        metadata: Dict[str, Any] = {
            "details_source": "eval_cache",
            "details_file": str(self.runtime_context.cache_store.samples_jsonl),
            "details_metric_id": self.spec.metric_id,
            "switch_reason": "cache_store_available",
        }
        if self.runtime_context.details_namespace:
            metadata["details_namespace"] = self.runtime_context.details_namespace
        return metadata

    def _resolve_preview_only_reason(self) -> str:
        if self._detail_mode == "cache_ref":
            return "cache_store_unavailable"
        if self._inline_overflowed:
            return "max_inline_items_exceeded"
        if self._detail_mode == "spill":
            return "spill_mode_not_implemented"
        return "detail_mode_requires_preview_only"


class CategoricalCountAggregator(MetricAggregator):
    """Counts occurrences of a categorical field in per-sample metadata."""

    def __init__(
        self,
        spec: MetricSpec,
        runtime_context: Optional[AggregationRuntimeContext] = None,
    ) -> None:
        super().__init__(spec, runtime_context=runtime_context)
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


def _normalize_detail_mode(value: Any) -> str:
    normalized = str(value or "auto").strip().lower()
    if normalized in {"auto", "cache_ref", "inline_preview", "spill"}:
        return normalized
    return "auto"


def _coerce_non_negative_int(value: Any, *, default: int) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return default
    return max(0, parsed)


def _coerce_positive_int(value: Any, *, default: int) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return default
    return max(1, parsed)
