from __future__ import annotations

import threading
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any

from gage_eval.config.pipeline_config import MetricSpec
from gage_eval.metrics import BaseMetric, MetricContext, MetricRegistry, MetricResult, SimpleMetric
from gage_eval.metrics.aggregators import CategoricalCountAggregator, MeanAggregator
from gage_eval.observability.trace import ObservabilityTrace
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


@registry.asset(
    "metrics",
    "_test_registry_serialized_compute_metric",
    desc="Verifies MetricInstance serializes metric.compute calls.",
)
class _SerializedComputeMetric(BaseMetric):
    def setup(self) -> None:
        self._active = 0
        self._max_active = 0
        self._state_lock = threading.Lock()

    @property
    def max_active(self) -> int:
        return self._max_active

    def compute(self, context: MetricContext) -> MetricResult:
        with self._state_lock:
            self._active += 1
            self._max_active = max(self._max_active, self._active)
        time.sleep(0.01)
        with self._state_lock:
            self._active -= 1
        return MetricResult(sample_id=context.sample_id, values={"score": 1.0})


def _make_context(trace: Any, sample_id: str = "sample-1") -> MetricContext:
    return MetricContext(
        sample_id=sample_id,
        sample={},
        model_output={},
        judge_output={},
        args={},
        trace=trace,
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


def test_metric_instance_serializes_stateful_compute() -> None:
    spec = MetricSpec(
        metric_id="test_registry_serialized_compute",
        implementation="_test_registry_serialized_compute_metric",
        aggregation="mean",
        params={},
    )
    instance = MetricRegistry().build_metric(spec)
    trace = ObservabilityTrace()
    contexts = [_make_context(trace, sample_id=f"sample-{index}") for index in range(8)]
    start_event = threading.Event()

    def _evaluate(context: MetricContext) -> None:
        start_event.wait(timeout=1.0)
        instance.evaluate(context)

    with ThreadPoolExecutor(max_workers=len(contexts)) as executor:
        futures = [executor.submit(_evaluate, context) for context in contexts]
        start_event.set()
        for future in futures:
            future.result()

    metric = instance.metric
    assert isinstance(metric, _SerializedComputeMetric)
    assert metric.max_active == 1

    aggregated = instance.finalize()
    assert aggregated["count"] == len(contexts)
    assert aggregated["values"] == {"score": 1.0}
