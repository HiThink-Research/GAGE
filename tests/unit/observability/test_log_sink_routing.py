from __future__ import annotations

import gc
import time
from dataclasses import dataclass
from datetime import datetime, timezone

import pytest

from gage_eval.observability.config import ObservabilityConfig, get_observability_config, set_observability_config
from gage_eval.observability.log_sink import ObservabilityLogSink
from gage_eval.observability.logger import ObservableLogger
from gage_eval.observability.trace import ObservabilityTrace
from gage_eval.reporting.recorders import InMemoryRecorder


@dataclass
class _Message:
    record: dict


class _DummyLogger:
    def bind(self, **kwargs):  # noqa: ANN003
        _ = kwargs
        return self

    def log(self, level, message, *args, **kwargs):  # noqa: ANN001, ANN003
        _ = level, message, args, kwargs
        return None


def _make_message(run_id: str | None, message: str) -> _Message:
    extra = {"stage": "unit_log_test"}
    if run_id is not None:
        extra["trace_run_id"] = run_id
    return _Message(
        record={
            "extra": extra,
            "name": "unit_log_test",
            "message": message,
            "level": {"name": "INFO"},
            "time": datetime.now(timezone.utc),
            "file": {"name": "test_file.py"},
            "function": "test_function",
            "line": 1,
        }
    )


@pytest.fixture(autouse=True)
def _enable_observability():
    original = get_observability_config()
    set_observability_config(ObservabilityConfig(enabled=True))
    try:
        yield
    finally:
        set_observability_config(original)


@pytest.mark.fast
def test_log_sink_routes_records_by_run_id() -> None:
    sink = ObservabilityLogSink(batch_size=4, flush_interval_s=1.0)
    trace_a = ObservabilityTrace(recorder=InMemoryRecorder(run_id="route-a"), run_id="route-a")
    trace_b = ObservabilityTrace(recorder=InMemoryRecorder(run_id="route-b"), run_id="route-b")
    try:
        sink.register_trace(trace_a)
        sink.register_trace(trace_b)

        sink(_make_message("route-a", "message-a"))
        sink(_make_message("route-b", "message-b"))

        assert sink.flush_run("route-a", timeout_s=1.0, close_mode="drain").closed_cleanly is True
        assert sink.flush_run("route-b", timeout_s=1.0, close_mode="drain").closed_cleanly is True

        logs_a = [event for event in trace_a.events if event["event"] == "log"]
        logs_b = [event for event in trace_b.events if event["event"] == "log"]

        assert len(logs_a) == 1
        assert logs_a[0]["payload"]["message"] == "message-a"
        assert len(logs_b) == 1
        assert logs_b[0]["payload"]["message"] == "message-b"
    finally:
        sink.close()


@pytest.mark.fast
def test_log_sink_ignores_records_without_trace_run_id() -> None:
    sink = ObservabilityLogSink(batch_size=2, flush_interval_s=1.0)
    trace = ObservabilityTrace(recorder=InMemoryRecorder(run_id="route-miss"), run_id="route-miss")
    try:
        sink.register_trace(trace)
        sink(_make_message(None, "missing-run"))

        assert sink.pending_count("route-miss") == 0
        assert [event for event in trace.events if event["event"] == "log"] == []
    finally:
        sink.close()


@pytest.mark.fast
def test_log_sink_drain_is_not_throttled_by_idle_interval() -> None:
    sink = ObservabilityLogSink(batch_size=1, flush_interval_s=2.0)
    trace = ObservabilityTrace(recorder=InMemoryRecorder(run_id="route-drain"), run_id="route-drain")
    try:
        sink.register_trace(trace)
        for idx in range(5):
            sink(_make_message("route-drain", f"message-{idx}"))

        started = time.monotonic()
        result = sink.flush_run("route-drain", timeout_s=1.0, close_mode="drain")
        elapsed = time.monotonic() - started

        assert result.closed_cleanly is True
        assert result.remaining_queue_size == 0
        assert elapsed < 0.5
    finally:
        sink.close()


@pytest.mark.fast
def test_log_sink_reaps_dead_routes() -> None:
    sink = ObservabilityLogSink(batch_size=2, flush_interval_s=1.0, route_sweep_interval_s=1.0)
    trace = ObservabilityTrace(recorder=InMemoryRecorder(run_id="route-zombie"), run_id="route-zombie")
    try:
        sink.register_trace(trace)
        assert sink.route_count() == 1
        del trace
        gc.collect()

        sink.reap_zombie_routes()

        assert sink.route_count() == 0
    finally:
        sink.close()


@pytest.mark.fast
def test_log_sink_reaps_orphan_pending_state_after_unregister() -> None:
    sink = ObservabilityLogSink(
        batch_size=1,
        flush_interval_s=1.0,
        zombie_route_ttl_s=0.01,
        route_sweep_interval_s=0.01,
    )
    trace = ObservabilityTrace(recorder=InMemoryRecorder(run_id="route-orphan"), run_id="route-orphan")
    try:
        sink.register_trace(trace)
        with sink._cv:  # noqa: SLF001 - white-box coverage for orphan cleanup.
            sink._pending_by_run["route-orphan"] = 1  # noqa: SLF001
        sink.unregister_trace("route-orphan")

        assert sink.pending_count("route-orphan") == 1
        assert sink.orphan_count() == 1

        sink.reap_zombie_routes(now=time.monotonic() + 1.0)

        assert sink.pending_count("route-orphan") == 0
        assert sink.orphan_count() == 0
    finally:
        sink.close()


@pytest.mark.fast
def test_observable_logger_does_not_reregister_closed_trace(monkeypatch) -> None:
    sink = ObservabilityLogSink(batch_size=1, flush_interval_s=1.0)
    trace = ObservabilityTrace(recorder=InMemoryRecorder(run_id="route-closed"), run_id="route-closed")
    test_logger = ObservableLogger()
    try:
        close_result = trace.close()

        monkeypatch.setattr("gage_eval.observability.logger.base_logger", _DummyLogger())
        monkeypatch.setattr("gage_eval.observability.logger.is_log_sink_active", lambda: True)
        monkeypatch.setattr("gage_eval.observability.logger.register_observable_trace", sink.register_trace)

        test_logger.info("unit_log_test", "late-message", trace=trace)

        assert close_result.closed_cleanly is True
        assert sink.route_count() == 0
        assert [event for event in trace.events if event["event"] == "log"] == []
    finally:
        sink.close()
