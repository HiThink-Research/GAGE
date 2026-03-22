"""Metric registry for assembling metric implementations and aggregators."""

from __future__ import annotations

import importlib
from functools import lru_cache
from pathlib import Path
import re
import threading
from typing import Callable, Dict, Optional, Type, TYPE_CHECKING

from gage_eval.config.pipeline_config import MetricSpec
from gage_eval.metrics.aggregators import (
    IdentityAggregator,
    MeanAggregator,
    MetricAggregator,
    WeightedMeanAggregator,
    CategoricalCountAggregator,
)
from gage_eval.metrics.runtime_context import AggregationRuntimeContext
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
AggregatorFactory = Callable[..., "MetricAggregator"]
_METRIC_ASSET_PATTERN = re.compile(
    r'registry\.asset\(\s*["\']metrics["\']\s*,\s*["\']([^"\']+)["\']',
    re.S,
)


class MetricRegistry:
    """Holds all metric/aggregation registrations and creates runtime instances."""

    def __init__(
        self,
        runtime_context: Optional[AggregationRuntimeContext] = None,
    ) -> None:
        from gage_eval.metrics.aggregators import (
            IdentityAggregator,
            MeanAggregator,
            WeightedMeanAggregator,
            CategoricalCountAggregator,
        )

        self._aggregators: Dict[str, AggregatorFactory] = {}
        self._runtime_context = runtime_context
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
        # Register MME-specific aggregator if available
        if MMEAccPlusAggregator is not None:
            self.register_aggregator(
                "mme_acc_plus",
                lambda spec, context=None: MMEAccPlusAggregator(
                    spec,
                    runtime_context=context,
                ),
            )
        if Tau2PassHatAggregator is not None:
            self.register_aggregator(
                "tau2_pass_hat",
                lambda spec, context=None: Tau2PassHatAggregator(
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

    # ------------------------------------------------------------------ #
    # Build API
    # ------------------------------------------------------------------ #
    def build_metric(
        self,
        spec: MetricSpec,
        runtime_context: Optional[AggregationRuntimeContext] = None,
    ) -> "MetricInstance":
        metric = self._build_metric_impl(spec)
        aggregation_id = self._resolve_aggregation_id(spec)
        if aggregation_id not in self._aggregators:
            raise KeyError(f"Aggregator '{aggregation_id}' not registered")
        aggregator = self._build_aggregator(
            self._aggregators[aggregation_id],
            spec,
            runtime_context if runtime_context is not None else self._runtime_context,
        )
        return MetricInstance(spec, metric, aggregator)

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

    def _build_metric_impl(self, spec: MetricSpec) -> BaseMetric:
        impl_key = spec.implementation or spec.metric_id
        if not impl_key:
            raise ValueError(f"Metric '{spec.metric_id}' must declare implementation")
        try:
            metric_cls = registry.get("metrics", impl_key)
            return metric_cls(spec)
        except KeyError:
            _import_metric_asset_module(impl_key)
        try:
            metric_cls = registry.get("metrics", impl_key)
            return metric_cls(spec)
        except KeyError:
            cls = self._import_metric_class(impl_key)
            return cls(spec)

    @staticmethod
    def _resolve_aggregation_id(spec: MetricSpec) -> str:
        if spec.aggregation:
            return spec.aggregation
        impl_key = spec.implementation or spec.metric_id
        if impl_key:
            try:
                entry = registry.entry("metrics", impl_key)
            except KeyError:
                _import_metric_asset_module(impl_key)
                try:
                    entry = registry.entry("metrics", impl_key)
                except KeyError:
                    entry = None
            if entry is not None:
                default_aggregation = entry.extra.get("default_aggregation")
                if default_aggregation:
                    return str(default_aggregation)
        return "mean"

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
    module_name = _metric_asset_modules().get(str(metric_id))
    if module_name:
        importlib.import_module(module_name)


@lru_cache(maxsize=1)
def _metric_asset_modules() -> Dict[str, str]:
    root = Path(__file__).resolve().parent / "builtin"
    module_prefix = "gage_eval.metrics.builtin"
    mapping: Dict[str, str] = {}
    for path in root.rglob("*.py"):
        if path.name in {"__init__.py"}:
            continue
        try:
            source = path.read_text(encoding="utf-8")
        except OSError:
            continue
        module_name = _module_name_for_path(path, root=root, module_prefix=module_prefix)
        for asset_name in _METRIC_ASSET_PATTERN.findall(source):
            mapping.setdefault(str(asset_name), module_name)
    return mapping


def _module_name_for_path(path: Path, *, root: Path, module_prefix: str) -> str:
    relative = path.relative_to(root).with_suffix("")
    parts = relative.parts
    if not parts:
        return module_prefix
    return ".".join((module_prefix, *parts))


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
