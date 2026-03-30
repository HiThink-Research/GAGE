"""Metric registry for assembling metric implementations and aggregators."""

from __future__ import annotations

import importlib
from dataclasses import replace
from functools import lru_cache
import threading
from typing import Callable, Dict, Optional, Type

from loguru import logger

from gage_eval.config.pipeline_config import MetricSpec
from gage_eval.metrics.aggregators import MetricAggregator
from gage_eval.metrics.runtime_context import AggregationRuntimeContext
from gage_eval.metrics.base import BaseMetric, MetricContext, MetricResult
from gage_eval.registry import import_asset_from_manifest, registry
from gage_eval.registry.entry import RegistryEntry

AggregatorFactory = Callable[..., "MetricAggregator"]

_OPTIONAL_BUILTIN_AGGREGATORS: tuple[tuple[str, str, str], ...] = (
    ("mme_acc_plus", "gage_eval.metrics.builtin.mme_aggregator", "MMEAccPlusAggregator"),
    ("tau2_pass_hat", "gage_eval.metrics.builtin.tau2_aggregator", "Tau2PassHatAggregator"),
)
_OPTIONAL_BUILTIN_AGGREGATORS_BY_ID: dict[str, tuple[str, str]] = {
    aggregation_id: (module_name, class_name)
    for aggregation_id, module_name, class_name in _OPTIONAL_BUILTIN_AGGREGATORS
}


@lru_cache(maxsize=None)
def _resolve_optional_builtin_aggregator(
    aggregation_id: str,
    module_name: str,
    class_name: str,
) -> tuple[type[MetricAggregator] | None, ImportError | None]:
    """Imports an optional builtin aggregator and caches the result."""

    try:
        module = importlib.import_module(module_name)
    except ImportError as exc:
        logger.warning(
            "Optional metric aggregator '{}' is unavailable because '{}' failed to import: {}",
            aggregation_id,
            module_name,
            exc,
        )
        return None, exc

    aggregator_cls = getattr(module, class_name)
    if not issubclass(aggregator_cls, MetricAggregator):
        raise TypeError(
            f"Optional metric aggregator '{aggregation_id}' must inherit from MetricAggregator "
            f"(found {aggregator_cls})"
        )

    return aggregator_cls, None


class MetricRegistry:
    """Holds all metric/aggregation registrations and creates runtime instances."""

    def __init__(
        self,
        runtime_context: Optional[AggregationRuntimeContext] = None,
        registry_view=None,
    ) -> None:
        from gage_eval.metrics.aggregators import (
            IdentityAggregator,
            MeanAggregator,
            WeightedMeanAggregator,
            CategoricalCountAggregator,
        )

        self._aggregators: Dict[str, AggregatorFactory] = {}
        self._optional_aggregator_errors: Dict[str, ImportError] = {}
        self._runtime_context = runtime_context
        self._registry_view = registry_view
        self.register_aggregator(
            "mean",
            lambda spec, context=None: MeanAggregator(spec, runtime_context=context),
        )
        self.register_aggregator(
            "weighted_mean",
            lambda spec, context=None: WeightedMeanAggregator(
                spec,
                runtime_context=context,
            ),
        )
        self.register_aggregator(
            "identity",
            lambda spec, context=None: IdentityAggregator(
                spec,
                runtime_context=context,
            ),
        )
        self.register_aggregator(
            "categorical_count",
            lambda spec, context=None: CategoricalCountAggregator(
                spec,
                runtime_context=context,
            ),
        )

    # ------------------------------------------------------------------ #
    # Registration API
    # ------------------------------------------------------------------ #
    def register_aggregator(self, agg_id: str, factory: AggregatorFactory) -> None:
        self._aggregators[agg_id] = factory

    def set_runtime_context(
        self,
        runtime_context: Optional[AggregationRuntimeContext],
    ) -> None:
        self._runtime_context = runtime_context

    def _registry_lookup(self):
        return self._registry_view or registry

    def _ensure_optional_builtin_aggregator(self, aggregation_id: str) -> None:
        if aggregation_id in self._aggregators or aggregation_id in self._optional_aggregator_errors:
            return
        optional_spec = _OPTIONAL_BUILTIN_AGGREGATORS_BY_ID.get(aggregation_id)
        if optional_spec is None:
            return
        module_name, class_name = optional_spec
        aggregator_cls, import_error = _resolve_optional_builtin_aggregator(
            aggregation_id,
            module_name,
            class_name,
        )
        if import_error is not None:
            self._optional_aggregator_errors[aggregation_id] = import_error
            return
        if aggregator_cls is None:
            return
        self.register_aggregator(
            aggregation_id,
            lambda spec, context=None, aggregator_cls=aggregator_cls: aggregator_cls(
                spec,
                runtime_context=context,
            ),
        )

    # ------------------------------------------------------------------ #
    # Build API
    # ------------------------------------------------------------------ #
    def build_metric(
        self,
        spec: MetricSpec,
        runtime_context: Optional[AggregationRuntimeContext] = None,
    ) -> "MetricInstance":
        impl_key, entry = self._resolve_metric_registration(spec)
        runtime_spec = self._resolve_runtime_spec(spec, entry)
        metric = self._build_metric_impl(runtime_spec, impl_key=impl_key, entry=entry)
        aggregation_id = runtime_spec.aggregation or "mean"
        self._ensure_optional_builtin_aggregator(aggregation_id)
        if aggregation_id not in self._aggregators:
            import_error = self._optional_aggregator_errors.get(aggregation_id)
            if import_error is not None:
                raise KeyError(
                    f"Aggregator '{aggregation_id}' not registered because its optional import "
                    f"failed: {import_error}"
                ) from import_error
            raise KeyError(f"Aggregator '{aggregation_id}' not registered")
        aggregator = self._build_aggregator(
            self._aggregators[aggregation_id],
            runtime_spec,
            runtime_context if runtime_context is not None else self._runtime_context,
        )
        return MetricInstance(runtime_spec, metric, aggregator)

    @staticmethod
    def _build_aggregator(
        factory: AggregatorFactory,
        spec: MetricSpec,
        runtime_context: Optional[AggregationRuntimeContext],
    ) -> "MetricAggregator":
        try:
            return factory(spec, runtime_context)
        except TypeError as original_exc:
            try:
                return factory(spec)
            except TypeError:
                raise original_exc

    def _resolve_metric_registration(self, spec: MetricSpec) -> tuple[str, RegistryEntry | None]:
        impl_key = spec.implementation or spec.metric_id
        if not impl_key:
            raise ValueError(f"Metric '{spec.metric_id}' must declare implementation")
        lookup = self._registry_lookup()
        try:
            return impl_key, lookup.entry("metrics", impl_key)
        except KeyError:
            if self._registry_view is not None:
                return impl_key, None
            _import_metric_asset_module(impl_key)
            try:
                return impl_key, registry.entry("metrics", impl_key)
            except KeyError:
                return impl_key, None

    def _resolve_runtime_spec(self, spec: MetricSpec, entry: RegistryEntry | None) -> MetricSpec:
        if spec.aggregation:
            return spec

        default_aggregation = None if entry is None else entry.extra.get("default_aggregation")
        aggregation_id = str(default_aggregation) if default_aggregation else "mean"
        return replace(spec, aggregation=aggregation_id)

    def _build_metric_impl(
        self,
        spec: MetricSpec,
        *,
        impl_key: str | None = None,
        entry: RegistryEntry | None = None,
    ) -> BaseMetric:
        impl_key = impl_key or spec.implementation or spec.metric_id
        if not impl_key:
            raise ValueError(f"Metric '{spec.metric_id}' must declare implementation")
        lookup = self._registry_lookup()
        if entry is not None:
            metric_cls = lookup.get("metrics", impl_key)
            return metric_cls(spec)
        try:
            metric_cls = lookup.get("metrics", impl_key)
            return metric_cls(spec)
        except KeyError:
            if self._registry_view is not None:
                raise
            _import_metric_asset_module(impl_key)
        try:
            metric_cls = registry.get("metrics", impl_key)
            return metric_cls(spec)
        except KeyError:
            cls = self._import_metric_class(impl_key)
            return cls(spec)

    @staticmethod
    def _import_metric_class(implementation: str) -> Type[BaseMetric]:
        """Imports a metric class from `pkg.module:ClassName` or `pkg.module.ClassName`."""

        if ":" in implementation:
            module_name, class_name = implementation.split(":", 1)
        elif "." in implementation:
            module_name, class_name = implementation.rsplit(".", 1)
        else:
            raise ValueError(
                f"Implementation '{implementation}' must contain module path, e.g. 'pkg.mod:Class'"
            )
        module = importlib.import_module(module_name)
        candidate = getattr(module, class_name)
        
        from gage_eval.metrics.base import BaseMetric
        if not issubclass(candidate, BaseMetric):
            raise TypeError(
                f"Metric class '{implementation}' must inherit from BaseMetric (found {candidate})"
            )
        return candidate


def _import_metric_asset_module(metric_id: str) -> None:
    import_asset_from_manifest("metrics", str(metric_id), registry=registry, source="metric_registry")


class MetricInstance:
    """Runtime metric glue that records per-sample values and aggregates them."""

    def __init__(self, spec: MetricSpec, metric: BaseMetric, aggregator: MetricAggregator) -> None:
        self.spec = spec
        self.metric = metric
        self.aggregator = aggregator
        self._lock = threading.Lock()

    def evaluate(self, context: MetricContext) -> MetricResult:
        with self._lock:
            # Metric implementations may keep mutable runtime state in `setup()`;
            # compute and aggregation therefore need a single critical section.
            result = self.metric.compute(context)
            self.aggregator.add(result)
        return result

    def finalize(self) -> Dict:
        with self._lock:
            aggregated = self.aggregator.finalize()
        return aggregated.to_dict()
