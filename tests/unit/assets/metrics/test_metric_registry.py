from __future__ import annotations

from gage_eval.config.pipeline_config import MetricSpec
from gage_eval.metrics import MetricContext, MetricRegistry, SimpleMetric
from gage_eval.metrics.aggregators import CategoricalCountAggregator, MeanAggregator
from gage_eval.registry import registry


@registry.asset(
    "metrics",
    "_test_registry_default_aggregation_metric",
    desc="Validates registry-provided default aggregation during metric construction.",
    default_aggregation="categorical_count",
)
class _RegistryDefaultAggregationMetric(SimpleMetric):
    value_key = "score"

    def compute_value(self, context: MetricContext) -> float:
        return 1.0

    def compute_metadata(self, context: MetricContext) -> dict[str, str]:
        return {"failure_reason": "timeout"}


def _make_context(mock_trace) -> MetricContext:
    return MetricContext(
        sample_id="sample-1",
        sample={},
        model_output={},
        judge_output={},
        args={},
        trace=mock_trace,
    )


def test_build_metric_uses_registry_default_aggregation(mock_trace) -> None:
    spec = MetricSpec(
        metric_id="test_registry_default_aggregation",
        implementation="_test_registry_default_aggregation_metric",
        params={},
    )

    instance = MetricRegistry().build_metric(spec)

    assert instance.spec.aggregation == "categorical_count"
    assert isinstance(instance.aggregator, CategoricalCountAggregator)

    instance.evaluate(_make_context(mock_trace))
    aggregated = instance.finalize()

    assert aggregated["aggregation"] == "categorical_count"
    assert aggregated["values"] == {"timeout": 1}


def test_build_metric_prefers_explicit_aggregation(mock_trace) -> None:
    spec = MetricSpec(
        metric_id="test_registry_default_aggregation_override",
        implementation="_test_registry_default_aggregation_metric",
        aggregation="mean",
        params={},
    )

    instance = MetricRegistry().build_metric(spec)

    assert instance.spec.aggregation == "mean"
    assert isinstance(instance.aggregator, MeanAggregator)

    instance.evaluate(_make_context(mock_trace))
    aggregated = instance.finalize()

    assert aggregated["aggregation"] == "mean"
    assert aggregated["values"] == {"score": 1.0}
