"""Metric registry for assembling metric implementations and aggregators."""

from __future__ import annotations

import importlib
import threading
from typing import Callable, Dict, Type, TYPE_CHECKING

from gage_eval.config.pipeline_config import MetricSpec
from gage_eval.registry import registry

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

    # ------------------------------------------------------------------ #
    # Registration API
    # ------------------------------------------------------------------ #
    def register_aggregator(self, agg_id: str, factory: AggregatorFactory) -> None:
        self._aggregators[agg_id] = factory

    # ------------------------------------------------------------------ #
    # Build API
    # ------------------------------------------------------------------ #
    def build_metric(self, spec: MetricSpec) -> "MetricInstance":
        metric = self._build_metric_impl(spec)
        aggregation_id = spec.aggregation or "mean"
        if aggregation_id not in self._aggregators:
            raise KeyError(f"Aggregator '{aggregation_id}' not registered")
        aggregator = self._aggregators[aggregation_id](spec)
        return MetricInstance(spec, metric, aggregator)

    def _build_metric_impl(self, spec: MetricSpec) -> BaseMetric:
        impl_key = spec.implementation or spec.metric_id
        if not impl_key:
            raise ValueError(f"Metric '{spec.metric_id}' must declare implementation")
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
