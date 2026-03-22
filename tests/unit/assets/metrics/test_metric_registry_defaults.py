from __future__ import annotations

from uuid import uuid4

from gage_eval.config.pipeline_config import MetricSpec
from gage_eval.metrics.base import MetricContext, SimpleMetric
from gage_eval.metrics.registry import MetricRegistry
from gage_eval.registry import registry


class _StaticMetric(SimpleMetric):
    def compute_value(self, context: MetricContext) -> float:
        return 1.0


def test_metric_registry_uses_registered_default_aggregation(mock_trace) -> None:
    metric_id = f"metric_default_{uuid4().hex}"
    scoped_registry = registry.clone()

    with registry.route_to(scoped_registry):
        scoped_registry.register(
            "metrics",
            metric_id,
            _StaticMetric,
            desc="test metric with non-mean default aggregation",
            default_aggregation="identity",
        )
        instance = MetricRegistry().build_metric(
            MetricSpec(metric_id=metric_id, implementation=metric_id)
        )

    assert instance.aggregator.__class__.__name__ == "IdentityAggregator"


def test_metric_registry_prefers_explicit_aggregation_over_registry_default(
    mock_trace,
) -> None:
    metric_id = f"metric_override_{uuid4().hex}"
    scoped_registry = registry.clone()

    with registry.route_to(scoped_registry):
        scoped_registry.register(
            "metrics",
            metric_id,
            _StaticMetric,
            desc="test metric with overridable default aggregation",
            default_aggregation="identity",
        )
        instance = MetricRegistry().build_metric(
            MetricSpec(
                metric_id=metric_id,
                implementation=metric_id,
                aggregation="mean",
            )
        )

    assert instance.aggregator.__class__.__name__ == "MeanAggregator"
