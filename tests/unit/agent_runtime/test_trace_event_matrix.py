from __future__ import annotations

from gage_eval.agent_runtime.tooling.trace import TOOLING_TRACE_EVENT_MATRIX


def test_tooling_trace_event_matrix_actor_and_payload_contract() -> None:
    expected = {
        "trial.start": ("runtime", {"trial_id", "trial_index", "trial_policy", "scheduler_type"}),
        "environment.acquire": ("environment", {"environment_descriptor", "role"}),
        "model.request": (
            "scheduler",
            {"turn_index", "provider", "backend_id", "tool_schema_count", "raw_request_ref"},
        ),
        "model.response": (
            "scheduler",
            {"turn_index", "raw_response_ref", "finish_reason", "tool_call_count"},
        ),
        "client.environment_handle.projected": ("scheduler", {"environment_handle", "capabilities"}),
        "tool.call.raw": ("agent", {"turn_index", "raw_message"}),
        "tool.call.normalized": ("agent", {"turn_index", "tool_call"}),
        "tool.result": (
            "runtime",
            {"tool_result", "tool_call_id", "name", "status", "latency_ms", "artifact_refs"},
        ),
        "tool.result.injected": ("scheduler", {"tool_call_id", "message"}),
        "verifier.result": ("verifier", {"metric", "verifier_result"}),
        "trial.end": ("runtime", {"trial_id", "status", "failure", "duration_ms"}),
    }

    assert set(TOOLING_TRACE_EVENT_MATRIX) == set(expected)
    for event_type, (actor, payload_fields) in expected.items():
        contract = TOOLING_TRACE_EVENT_MATRIX[event_type]
        assert contract.actor == actor
        assert contract.payload_fields == payload_fields
