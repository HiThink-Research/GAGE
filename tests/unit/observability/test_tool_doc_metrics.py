from __future__ import annotations

import pytest

from gage_eval.observability.trace import ObservabilityTrace
from gage_eval.reporting.recorders import InMemoryRecorder


@pytest.mark.fast
def test_trace_emits_tool_documentation_event() -> None:
    recorder = InMemoryRecorder(run_id="tool-docs")
    trace = ObservabilityTrace(recorder=recorder, run_id="tool-docs")
    trace.emit_tool_documentation({"chars": 120, "apps": 2}, sample_id="sample-1")

    events = trace.events
    assert events[-1]["event"] == "tool_documentation_built"
    assert events[-1]["payload"]["chars"] == 120
    assert events[-1]["sample_id"] == "sample-1"
