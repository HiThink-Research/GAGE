from __future__ import annotations

from contextlib import contextmanager
from typing import Iterator

import pytest

from gage_eval.observability.trace import ObservabilityTrace
from gage_eval.pipeline.steps.inference import InferenceStep
from gage_eval.reporting.recorders import InMemoryRecorder


class _RoleWithObservabilityEvents:
    def invoke(self, _payload: dict, _trace: ObservabilityTrace) -> dict:
        return {
            "answer": "",
            "observability_events": [
                {
                    "event": "agent_retry_missing_tool_call",
                    "payload": {"turn_index": 1, "consecutive_retries": 1},
                },
                {
                    "event": "agent_loop_exhausted",
                    "payload": {"reason": "tool_call_retry_budget", "turn_index": 3},
                },
            ],
        }


class _RoleManager:
    @contextmanager
    def borrow_role(self, _adapter_id: str) -> Iterator[_RoleWithObservabilityEvents]:
        yield _RoleWithObservabilityEvents()


def _build_trace() -> ObservabilityTrace:
    return ObservabilityTrace(
        recorder=InMemoryRecorder(run_id="inference-observability-events"),
        run_id="inference-observability-events",
    )


@pytest.mark.fast
def test_inference_step_emits_role_observability_events_before_inference_end() -> None:
    trace = _build_trace()
    sample = {"id": "sample-1"}
    step = InferenceStep(adapter_id="dut-agent")

    output = step.execute(sample, _RoleManager(), trace)

    assert output["answer"] == ""
    event_names = [event["event"] for event in trace.events]
    assert event_names == [
        "inference_start",
        "agent_retry_missing_tool_call",
        "agent_loop_exhausted",
        "inference_end",
    ]
    retry_event = trace.events[1]
    exhausted_event = trace.events[2]
    assert retry_event["sample_id"] == "sample-1"
    assert retry_event["payload"]["consecutive_retries"] == 1
    assert exhausted_event["sample_id"] == "sample-1"
    assert exhausted_event["payload"]["reason"] == "tool_call_retry_budget"
