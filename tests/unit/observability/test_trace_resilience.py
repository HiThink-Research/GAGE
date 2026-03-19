from __future__ import annotations

import pytest

from gage_eval.observability.trace import ObservabilityTrace
from gage_eval.reporting.recorders import InMemoryRecorder, RecorderBase, ResilientRecorder, TraceEvent


class _AlwaysFailRecorder(RecorderBase):
    def _flush_events_internal(self, events: tuple[TraceEvent, ...] | list[TraceEvent]) -> None:
        raise RuntimeError("primary down")


@pytest.mark.fast
def test_trace_falls_back_without_interrupting_main_flow() -> None:
    fallback = InMemoryRecorder(run_id="trace-fallback")
    primary = _AlwaysFailRecorder(run_id="trace-fallback", min_flush_events=1, min_flush_seconds=10_000.0)
    trace = ObservabilityTrace(
        recorder=ResilientRecorder(primary, fallback=fallback),
        run_id="trace-fallback",
    )

    trace.emit("runtime_ready", {"selected_dataset": "ds"}, sample_id="sample-1")

    health = trace.health_snapshot()
    buffered = fallback.buffered_events()

    assert health["observability_degraded"] is True
    assert health["observability_mode"] == "fallback"
    assert buffered
    assert buffered[0]["event"] == "runtime_ready"
    assert trace.events[-1]["event"] == "runtime_ready"


@pytest.mark.fast
def test_trace_switches_to_noop_when_no_fallback_exists() -> None:
    primary = _AlwaysFailRecorder(run_id="trace-noop", min_flush_events=1, min_flush_seconds=10_000.0)
    trace = ObservabilityTrace(
        recorder=ResilientRecorder(primary),
        run_id="trace-noop",
    )

    trace.emit("runtime_ready", {"selected_dataset": "ds"})

    health = trace.health_snapshot()

    assert health["observability_degraded"] is True
    assert health["observability_mode"] == "noop"
