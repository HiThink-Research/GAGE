from __future__ import annotations

import threading

import pytest

from gage_eval.reporting.recorders import InMemoryRecorder, RecorderBase, TraceEvent


class _BlockingRecorder(RecorderBase):
    def __init__(self, run_id: str) -> None:
        super().__init__(run_id, min_flush_events=100, min_flush_seconds=10_000.0)
        self.started = threading.Event()
        self.release = threading.Event()
        self.batches: list[list[str]] = []

    def _flush_events_internal(self, events: tuple[TraceEvent, ...] | list[TraceEvent]) -> None:
        self.started.set()
        assert self.release.wait(timeout=1.0)
        self.batches.append([event.event for event in events])


@pytest.mark.fast
def test_successful_flush_compacts_committed_events() -> None:
    recorder = InMemoryRecorder(
        run_id="recorder-compaction",
        min_flush_events=100,
        min_flush_seconds=10_000.0,
    )
    recorder.record_event("first", {"value": 1})
    recorder.record_event("second", {"value": 2})

    recorder.flush_events()

    assert list(recorder.get_events()) == []
    assert list(recorder.pending_events()) == []
    assert [event["event"] for event in recorder.buffered_events()] == ["first", "second"]
    stats = recorder.buffer_stats()
    assert stats.events_flushed_total == 2
    assert stats.recorder_compactions_total == 1
    assert stats.recorder_retained_events == 0


@pytest.mark.fast
def test_flush_compaction_preserves_concurrent_appends() -> None:
    recorder = _BlockingRecorder(run_id="recorder-concurrent")
    recorder.record_event("first", {"value": 1})

    worker = threading.Thread(target=recorder.flush_events)
    worker.start()
    assert recorder.started.wait(timeout=1.0)

    recorder.record_event("second", {"value": 2})
    recorder.release.set()
    worker.join(timeout=1.0)
    assert not worker.is_alive()

    assert recorder.batches == [["first"]]
    assert [event.event for event in recorder.pending_events()] == ["second"]
    assert [event.event for event in recorder.get_events()] == ["second"]

    recorder.flush_events()

    assert recorder.batches == [["first"], ["second"]]
    assert list(recorder.get_events()) == []
    stats = recorder.buffer_stats()
    assert stats.events_flushed_total == 2
    assert stats.recorder_compactions_total == 2
