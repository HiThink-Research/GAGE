from __future__ import annotations

import json

import pytest

from gage_eval.evaluation.cache import EvalCache
from gage_eval.observability.trace import ObservabilityTrace
from gage_eval.pipeline.steps.report import ReportStep
from gage_eval.reporting.recorders import RecorderBase, ResilientRecorder, TraceEvent


class _AlwaysFailRecorder(RecorderBase):
    def _flush_events_internal(self, events: tuple[TraceEvent, ...] | list[TraceEvent]) -> None:
        raise RuntimeError("summary recorder down")


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
    assert summary["observability_degraded"] is True
    assert summary["observability_mode"] == "noop"
    assert summary["backlog_events"] == 1
