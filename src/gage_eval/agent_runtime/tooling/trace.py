from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


TraceActor = Literal["scheduler", "agent", "environment", "verifier", "runtime"]


@dataclass(frozen=True)
class TraceEventContract:
    actor: TraceActor
    payload_fields: set[str]


TOOLING_TRACE_EVENT_MATRIX: dict[str, TraceEventContract] = {
    "trial.start": TraceEventContract("runtime", {"trial_id", "trial_index", "trial_policy", "scheduler_type"}),
    "environment.acquire": TraceEventContract("environment", {"environment_descriptor", "role"}),
    "model.request": TraceEventContract(
        "scheduler", {"turn_index", "provider", "backend_id", "tool_schema_count", "raw_request_ref"}
    ),
    "model.response": TraceEventContract(
        "scheduler", {"turn_index", "raw_response_ref", "finish_reason", "tool_call_count"}
    ),
    "client.environment_handle.projected": TraceEventContract(
        "scheduler", {"environment_handle", "capabilities"}
    ),
    "tool.call.raw": TraceEventContract("agent", {"turn_index", "raw_message"}),
    "tool.call.normalized": TraceEventContract("agent", {"turn_index", "tool_call"}),
    "tool.result": TraceEventContract(
        "runtime", {"tool_result", "tool_call_id", "name", "status", "latency_ms", "artifact_refs"}
    ),
    "tool.result.injected": TraceEventContract("scheduler", {"tool_call_id", "message"}),
    "verifier.result": TraceEventContract("verifier", {"metric", "verifier_result"}),
    "trial.end": TraceEventContract("runtime", {"trial_id", "status", "failure", "duration_ms"}),
}
