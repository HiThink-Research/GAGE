from __future__ import annotations

import pytest

from gage_eval.observability.progress_sink import ProgressSnapshot, ProgressSink, TraceProgressSink
from gage_eval.observability.trace import ObservabilityTrace
from gage_eval.reporting.recorders import InMemoryRecorder


class _RecordingSink:
    def __init__(self) -> None:
        self.updates: list[dict] = []

    def update(
        self,
        *,
        completed: int,
        total: int,
        phase: str,
        elapsed_s: float,
        **extra,
    ) -> None:
        self.updates.append(
            {
                "completed": completed,
                "total": total,
                "phase": phase,
                "elapsed_s": elapsed_s,
                "extra": extra,
            }
        )


@pytest.mark.fast
def test_progress_sink_protocol_is_defined_in_observability_namespace() -> None:
    sink: ProgressSink = _RecordingSink()

    sink.update(completed=1, total=3, phase="running", elapsed_s=12.5, sample_id="sample-1")

    assert isinstance(sink, ProgressSink)
    assert sink.updates == [
        {
            "completed": 1,
            "total": 3,
            "phase": "running",
            "elapsed_s": 12.5,
            "extra": {"sample_id": "sample-1"},
        }
    ]


@pytest.mark.fast
def test_trace_progress_sink_emits_appendix_e_progress_event() -> None:
    recorder = InMemoryRecorder(run_id="progress-sink", min_flush_events=1)
    trace = ObservabilityTrace(recorder=recorder, run_id="progress-sink")
    sink = TraceProgressSink(trace=trace, job_name="tb2_job")

    sink.update(completed=1, total=3, phase="running", elapsed_s=12.5)
    trace.flush()

    assert sink.snapshot == ProgressSnapshot(
        completed=1,
        total=3,
        phase="running",
        elapsed_s=12.5,
    )
    assert recorder.buffered_events() == [
        {
            "run_id": "progress-sink",
            "event_id": 0,
            "event": "external_harness_progress",
            "payload": {
                "job_name": "tb2_job",
                "completed": 1,
                "total": 3,
                "phase": "running",
                "elapsed_s": 12.5,
            },
            "sample_id": None,
            "created_at": recorder.buffered_events()[0]["created_at"],
        }
    ]


@pytest.mark.fast
def test_trace_progress_sink_rejects_negative_counts() -> None:
    sink = TraceProgressSink(trace=ObservabilityTrace(), job_name="tb2_job")

    with pytest.raises(ValueError, match="completed"):
        sink.update(completed=-1, total=3, phase="running", elapsed_s=0.0)

    with pytest.raises(ValueError, match="total"):
        sink.update(completed=0, total=-1, phase="running", elapsed_s=0.0)
