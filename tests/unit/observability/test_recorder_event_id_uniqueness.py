from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor

import pytest

from gage_eval.observability.trace import ObservabilityTrace
from gage_eval.reporting.recorders import InMemoryRecorder


@pytest.mark.fast
def test_recorder_direct_path_assigns_monotonic_unique_event_ids() -> None:
    recorder = InMemoryRecorder(
        run_id="direct-event-id",
        min_flush_events=10_000,
        min_flush_seconds=10_000.0,
    )
    total_events = 32

    def _record(idx: int) -> None:
        recorder.record_event("runtime_ready", {"idx": idx}, sample_id=f"sample-{idx}")

    with ThreadPoolExecutor(max_workers=8) as pool:
        list(pool.map(_record, range(total_events)))

    event_ids = sorted(event.event_id for event in recorder.get_events())

    assert event_ids == list(range(total_events))


@pytest.mark.fast
def test_trace_and_direct_recorder_path_share_event_id_sequence() -> None:
    recorder = InMemoryRecorder(
        run_id="mixed-event-id",
        min_flush_events=10_000,
        min_flush_seconds=10_000.0,
    )
    trace = ObservabilityTrace(recorder=recorder, run_id="mixed-event-id")

    trace.emit("trace_start", {"step": 1})
    recorder.record_event("direct_mid", {"step": 2})
    trace.emit("trace_end", {"step": 3})

    event_ids = [event.event_id for event in recorder.get_events()]

    assert event_ids == [0, 1, 2]
