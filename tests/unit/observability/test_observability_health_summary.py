from __future__ import annotations

import json

import pytest

from gage_eval.evaluation.cache import EvalCache
from gage_eval.observability.trace import ObservabilityTrace
from gage_eval.pipeline.steps.report import ReportStep
from gage_eval.reporting.recorders import InMemoryRecorder, RecorderBase, ResilientRecorder, TraceEvent


class _AlwaysFailRecorder(RecorderBase):
    def _flush_events_internal(self, events: tuple[TraceEvent, ...] | list[TraceEvent]) -> None:
        raise RuntimeError("summary recorder down")


class _CloseFailRecorder(RecorderBase):
    def _flush_events_internal(self, events: tuple[TraceEvent, ...] | list[TraceEvent]) -> None:
        return

    def close(self) -> None:
        super().close()
        raise RuntimeError("summary close boom")


@pytest.mark.fast
def test_report_summary_includes_observability_health(tmp_path) -> None:
    trace = ObservabilityTrace(
        recorder=ResilientRecorder(
            _AlwaysFailRecorder(
                run_id="summary-trace",
                min_flush_events=1,
                min_flush_seconds=10_000.0,
            )
        ),
        run_id="summary-trace",
    )
    trace.emit("runtime_ready", {"selected_dataset": "ds"})
    cache = EvalCache(base_dir=tmp_path, run_id=trace.run_id)
    report = ReportStep(auto_eval_step=None, cache_store=cache)

    payload = report.finalize(trace)
    summary = json.loads((cache.run_dir / "summary.json").read_text(encoding="utf-8"))

    assert payload["observability_degraded"] is True
    assert payload["observability_mode"] == "noop"
    assert payload["backlog_events"] == 1
    assert payload["events_emitted_total"] == 1
    assert payload["events_retained_in_memory"] == 1
    assert payload["events_dropped_by_ring_buffer"] == 0
    assert payload["events_flushed_total"] == 0
    assert summary["observability_degraded"] is True
    assert summary["observability_mode"] == "noop"
    assert summary["backlog_events"] == 1
    assert summary["events_emitted_total"] == 1
    assert summary["events_retained_in_memory"] == 1
    assert summary["events_dropped_by_ring_buffer"] == 0
    assert summary["events_flushed_total"] == 0


@pytest.mark.fast
def test_trace_close_patches_summary_with_best_effort_warning(tmp_path) -> None:
    trace = ObservabilityTrace(
        recorder=InMemoryRecorder(run_id="summary-close-warning"),
        run_id="summary-close-warning",
    )
    trace.emit("runtime_ready", {"selected_dataset": "ds"})
    cache = EvalCache(base_dir=tmp_path, run_id=trace.run_id)
    report = ReportStep(auto_eval_step=None, cache_store=cache)

    report.finalize(trace)
    close_result = trace.close(close_mode="best_effort", cache_store=cache)
    summary = json.loads((cache.run_dir / "summary.json").read_text(encoding="utf-8"))

    assert close_result.close_mode == "best_effort"
    assert close_result.warning == "best_effort mode may leave observability data incomplete"
    assert summary["observability_close_mode"] == "best_effort"
    assert summary["observability_close_warning"] == "best_effort mode may leave observability data incomplete"


@pytest.mark.fast
def test_trace_close_rejects_late_events(tmp_path) -> None:
    trace = ObservabilityTrace(
        recorder=InMemoryRecorder(run_id="summary-close-guard"),
        run_id="summary-close-guard",
    )
    trace.emit("runtime_ready", {"selected_dataset": "ds"})
    cache = EvalCache(base_dir=tmp_path, run_id=trace.run_id)
    report = ReportStep(auto_eval_step=None, cache_store=cache)

    report.finalize(trace)
    close_result = trace.close(cache_store=cache)
    events_before = list(trace.events)

    trace.emit("late_event", {"selected_dataset": "ds"})

    assert close_result.closed_cleanly is True
    assert trace.events == events_before


@pytest.mark.fast
def test_trace_close_summary_marks_resilient_close_failure(tmp_path) -> None:
    trace = ObservabilityTrace(
        recorder=_CloseFailRecorder(run_id="summary-close-fail"),
        run_id="summary-close-fail",
    )
    trace.emit("runtime_ready", {"selected_dataset": "ds"})
    cache = EvalCache(base_dir=tmp_path, run_id=trace.run_id)
    report = ReportStep(auto_eval_step=None, cache_store=cache)

    report.finalize(trace)
    close_result = trace.close(cache_store=cache)
    summary = json.loads((cache.run_dir / "summary.json").read_text(encoding="utf-8"))

    assert close_result.closed_cleanly is False
    assert close_result.error_type == "RecorderCloseError"
    assert "summary close boom" in (close_result.error_message or "")
    assert summary["observability_closed_cleanly"] is False
    assert summary["observability_close_error_type"] == "RecorderCloseError"
