"""Metric registry for assembling metric implementations and aggregators."""

from __future__ import annotations

import importlib
import threading
from dataclasses import replace
from typing import Callable, Dict, Type, TYPE_CHECKING

from gage_eval.config.pipeline_config import MetricSpec
from gage_eval.metrics.aggregators import (
    IdentityAggregator,
    MeanAggregator,
    MetricAggregator,
    WeightedMeanAggregator,
    CategoricalCountAggregator,
)
# Import MME-specific aggregator from builtin module
try:
    from gage_eval.metrics.builtin.mme_aggregator import MMEAccPlusAggregator
except ImportError:
    MMEAccPlusAggregator = None
try:
    from gage_eval.metrics.builtin.tau2_aggregator import Tau2PassHatAggregator
except ImportError:
    Tau2PassHatAggregator = None
from gage_eval.metrics.base import BaseMetric, MetricContext, MetricResult
from gage_eval.registry import registry
from gage_eval.registry.entry import RegistryEntry

if TYPE_CHECKING:
    from gage_eval.metrics.aggregators import (
        IdentityAggregator,
        MeanAggregator,
        MetricAggregator,
        WeightedMeanAggregator,
        CategoricalCountAggregator,
    )
    from gage_eval.metrics.base import BaseMetric, MetricContext, MetricResult

MetricFactory = Callable[[MetricSpec], "BaseMetric"]
AggregatorFactory = Callable[[MetricSpec], "MetricAggregator"]


class MetricRegistry:
    """Holds all metric/aggregation registrations and creates runtime instances."""

    def __init__(self) -> None:
        from gage_eval.metrics.aggregators import (
            IdentityAggregator,
            MeanAggregator,
            WeightedMeanAggregator,
            CategoricalCountAggregator,
        )

        self._aggregators: Dict[str, AggregatorFactory] = {}
        self.register_aggregator("mean", lambda spec: MeanAggregator(spec))
        self.register_aggregator("weighted_mean", lambda spec: WeightedMeanAggregator(spec))
        self.register_aggregator("identity", lambda spec: IdentityAggregator(spec))
        self.register_aggregator("categorical_count", lambda spec: CategoricalCountAggregator(spec))
        # Register MME-specific aggregator if available
        if MMEAccPlusAggregator is not None:
            self.register_aggregator("mme_acc_plus", lambda spec: MMEAccPlusAggregator(spec))
        if Tau2PassHatAggregator is not None:
            self.register_aggregator("tau2_pass_hat", lambda spec: Tau2PassHatAggregator(spec))

    # ------------------------------------------------------------------ #
    # Registration API
    # ------------------------------------------------------------------ #
    def register_aggregator(self, agg_id: str, factory: AggregatorFactory) -> None:
        self._aggregators[agg_id] = factory

    # ------------------------------------------------------------------ #
    # Build API
    # ------------------------------------------------------------------ #
    def build_metric(self, spec: MetricSpec) -> "MetricInstance":
        impl_key, entry = self._resolve_metric_registration(spec)
        runtime_spec = self._resolve_runtime_spec(spec, entry)
        metric = self._build_metric_impl(runtime_spec, impl_key=impl_key, entry=entry)
        aggregation_id = runtime_spec.aggregation or "mean"
        if aggregation_id not in self._aggregators:
            raise KeyError(f"Aggregator '{aggregation_id}' not registered")
        aggregator = self._aggregators[aggregation_id](runtime_spec)
        return MetricInstance(runtime_spec, metric, aggregator)

    def _resolve_metric_registration(self, spec: MetricSpec) -> tuple[str, RegistryEntry | None]:
        impl_key = spec.implementation or spec.metric_id
        if not impl_key:
            raise ValueError(f"Metric '{spec.metric_id}' must declare implementation")
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
        if entry is not None:
            metric_cls = registry.get("metrics", impl_key)
            return metric_cls(spec)
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


class MetricInstance:
    """Runtime metric glue that records per-sample values and aggregates them."""

    def __init__(self, spec: MetricSpec, metric: BaseMetric, aggregator: MetricAggregator) -> None:
        self.spec = spec
        self.metric = metric
        self.aggregator = aggregator
        self._lock = threading.Lock()

    def evaluate(self, context: MetricContext) -> MetricResult:
        result = self.metric.compute(context)
        with self._lock:
            self.aggregator.add(result)
        return result

    def finalize(self) -> Dict:
        with self._lock:
            aggregated = self.aggregator.finalize()
        return aggregated.to_dict()
