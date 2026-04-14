from __future__ import annotations

import json
from pathlib import Path

import pytest

from gage_eval.reporting.recorders import FileRecorder


@pytest.mark.fast
def test_file_recorder_flushes_100ms_event_batches_without_waiting_for_close(tmp_path: Path) -> None:
    events_path = tmp_path / "events.jsonl"
    recorder = FileRecorder(run_id="file-batch", output_path=events_path)

    recorder.record_event("first", {"value": 1})
    recorder._last_flush_ts -= 0.2  # noqa: SLF001
    recorder.record_event("second", {"value": 2})

    lines = events_path.read_text(encoding="utf-8").splitlines()

    assert [json.loads(line)["event"] for line in lines] == ["first", "second"]
    assert recorder.pending_events() == []


@pytest.mark.fast
def test_file_recorder_flushes_when_100_events_accumulate(tmp_path: Path) -> None:
    events_path = tmp_path / "events.jsonl"
    recorder = FileRecorder(
        run_id="file-batch",
        output_path=events_path,
        min_flush_seconds=999.0,
    )

    for index in range(99):
        recorder.record_event(f"event-{index}", {"value": index})

    assert not events_path.exists()

    recorder.record_event("event-99", {"value": 99})
    lines = events_path.read_text(encoding="utf-8").splitlines()

    assert len(lines) == 100
    assert json.loads(lines[0])["event"] == "event-0"
    assert json.loads(lines[-1])["event"] == "event-99"
    assert recorder.pending_events() == []
