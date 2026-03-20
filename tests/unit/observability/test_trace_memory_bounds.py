from __future__ import annotations

import pytest

from gage_eval.observability.trace import ObservabilityTrace
from gage_eval.reporting.recorders import InMemoryRecorder


@pytest.mark.fast
def test_trace_events_keep_recent_snapshot(monkeypatch) -> None:
    monkeypatch.setenv("GAGE_EVAL_TRACE_BUFFER_MAX_EVENTS", "2")
    trace = ObservabilityTrace(
        recorder=InMemoryRecorder(run_id="trace-buffer"),
        run_id="trace-buffer",
    )

    trace.emit("first", {"value": 1})
    trace.emit("second", {"value": 2})
    trace.emit("third", {"value": 3})

    assert [event["event"] for event in trace.events] == ["second", "third"]

    health = trace.health_snapshot()

    assert health["events_emitted_total"] == 3
    assert health["events_retained_in_memory"] == 2
    assert health["events_dropped_by_ring_buffer"] == 1
    assert health["events_flushed_total"] == 0
