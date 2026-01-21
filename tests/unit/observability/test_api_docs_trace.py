from __future__ import annotations

import pytest

from gage_eval.observability.trace import ObservabilityTrace
from gage_eval.pipeline.steps.support import SupportStep
from gage_eval.reporting.recorders import InMemoryRecorder


class StubRole:
    def __init__(self, output: dict) -> None:
        self._output = output

    def invoke(self, _payload: dict, _trace: ObservabilityTrace) -> dict:
        return dict(self._output)


class StubRoleManager:
    def __init__(self, output: dict) -> None:
        self._output = output

    def borrow_role(self, _adapter_id: str):
        output = dict(self._output)

        class Lease:
            def __enter__(self_inner):
                return StubRole(output)

            def __exit__(self_inner, exc_type, exc, tb) -> bool:
                return False

        return Lease()


@pytest.mark.fast
def test_support_step_emits_api_docs_events() -> None:
    recorder = InMemoryRecorder(run_id="api-docs")
    trace = ObservabilityTrace(recorder=recorder, run_id="api-docs")
    output = {
        "observability_events": [
            {"event": "api_docs_query", "payload": {"cache_hit": False}}
        ]
    }
    role_manager = StubRoleManager(output)
    step = SupportStep(steps=[{"adapter_id": "api_docs_context"}])
    sample = {"id": "sample-1"}

    step.execute(sample, role_manager, trace)

    events = [event for event in trace.events if event["event"] == "api_docs_query"]
    assert events
    assert events[0]["payload"]["cache_hit"] is False
    assert events[0]["sample_id"] == "sample-1"
