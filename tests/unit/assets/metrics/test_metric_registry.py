from __future__ import annotations

import importlib
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace
from typing import Any, Iterator

import pytest

import gage_eval.metrics.registry as metric_registry_module
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


@pytest.fixture(autouse=True)
def clear_optional_aggregator_cache() -> Iterator[None]:
    metric_registry_module._resolve_optional_builtin_aggregator.cache_clear()
    yield
    metric_registry_module._resolve_optional_builtin_aggregator.cache_clear()


def _install_optional_import_failure(
    monkeypatch: pytest.MonkeyPatch,
    *,
    module_name: str,
    error_message: str,
    warnings: list[tuple[str, tuple[object, ...]]],
) -> None:
    original_import_module = importlib.import_module

    def fake_import_module(name: str, package: str | None = None) -> object:
        if name == module_name:
            raise ImportError(error_message)
        return original_import_module(name, package)

    monkeypatch.setattr(metric_registry_module.importlib, "import_module", fake_import_module)
    monkeypatch.setattr(
        metric_registry_module,
        "logger",
        SimpleNamespace(warning=lambda message, *args: warnings.append((message, args))),
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


@pytest.mark.parametrize(
    ("aggregation_id", "module_name"),
    [
        ("mme_acc_plus", "gage_eval.metrics.builtin.mme_aggregator"),
        ("tau2_pass_hat", "gage_eval.metrics.builtin.tau2_aggregator"),
    ],
)
def test_metric_registry_logs_optional_aggregator_import_failures(
    monkeypatch: pytest.MonkeyPatch,
    aggregation_id: str,
    module_name: str,
) -> None:
    warnings: list[tuple[str, tuple[object, ...]]] = []
    error_message = f"missing dependency for {aggregation_id}"

    _install_optional_import_failure(
        monkeypatch,
        module_name=module_name,
        error_message=error_message,
        warnings=warnings,
    )

    registry_instance = MetricRegistry()
    target_warnings = [call for call in warnings if call[1][0] == aggregation_id]

    assert aggregation_id not in registry_instance._aggregators
    assert len(target_warnings) == 1
    assert target_warnings[0][0] == (
        "Optional metric aggregator '{}' is unavailable because '{}' failed to import: {}"
    )
    assert target_warnings[0][1][0] == aggregation_id
    assert target_warnings[0][1][1] == module_name
    assert str(target_warnings[0][1][2]) == error_message


@pytest.mark.parametrize(
    ("aggregation_id", "module_name"),
    [
        ("mme_acc_plus", "gage_eval.metrics.builtin.mme_aggregator"),
        ("tau2_pass_hat", "gage_eval.metrics.builtin.tau2_aggregator"),
    ],
)
def test_build_metric_reports_optional_aggregator_import_failures(
    monkeypatch: pytest.MonkeyPatch,
    aggregation_id: str,
    module_name: str,
) -> None:
    warnings: list[tuple[str, tuple[object, ...]]] = []
    error_message = f"missing dependency for {aggregation_id}"

    _install_optional_import_failure(
        monkeypatch,
        module_name=module_name,
        error_message=error_message,
        warnings=warnings,
    )

    registry_instance = MetricRegistry()
    spec = MetricSpec(
        metric_id=f"test_missing_{aggregation_id}",
        implementation="_test_registry_default_aggregation_metric",
        aggregation=aggregation_id,
        params={},
    )

    with pytest.raises(
        KeyError,
        match=rf"Aggregator '{aggregation_id}' not registered because its optional import failed:",
    ) as exc_info:
        registry_instance.build_metric(spec)

    assert isinstance(exc_info.value.__cause__, ImportError)
    assert error_message in str(exc_info.value)


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
