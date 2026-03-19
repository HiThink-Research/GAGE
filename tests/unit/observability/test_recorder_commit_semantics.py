from __future__ import annotations

import pytest

from gage_eval.reporting.recorders import RecorderBase, TraceEvent


class _AlwaysFailFlushRecorder(RecorderBase):
    def _flush_events_internal(self, events: tuple[TraceEvent, ...] | list[TraceEvent]) -> None:
        raise RuntimeError("flush boom")


@pytest.mark.fast
def test_flush_failure_keeps_pending_events() -> None:
    recorder = _AlwaysFailFlushRecorder(
        run_id="commit-semantics",
        min_flush_events=100,
        min_flush_seconds=10_000.0,
    )
    recorder.record_event("runtime_ready", {"ok": True}, sample_id="sample-1")

    with pytest.raises(RuntimeError, match="flush boom"):
        recorder.flush_events()

    pending = recorder.pending_events()

    assert len(pending) == 1
    assert pending[0].event == "runtime_ready"
    assert recorder._written == 0
