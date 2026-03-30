from __future__ import annotations

import re

import pytest

from gage_eval.observability.trace import ObservabilityTrace
from gage_eval.reporting.recorders import InMemoryRecorder


@pytest.mark.fast
def test_observability_trace_generates_unique_run_ids(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("GAGE_EVAL_SAVE_DIR", str(tmp_path))

    trace_one = ObservabilityTrace()
    trace_two = ObservabilityTrace()

    assert trace_one.run_id != trace_two.run_id
    assert re.fullmatch(r"run-\d{14}-[0-9a-f]{8}", trace_one.run_id)
    assert trace_one.run_identity.source == "generated"


@pytest.mark.fast
def test_observability_trace_preserves_explicit_run_id() -> None:
    trace = ObservabilityTrace(
        recorder=InMemoryRecorder(run_id="explicit-run-id"),
        run_id="explicit-run-id",
    )

    assert trace.run_id == "explicit-run-id"
    assert trace.run_identity.source == "provided"
