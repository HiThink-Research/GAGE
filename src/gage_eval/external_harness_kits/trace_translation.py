"""Shared SPI for importing external harness traces into AgentKitV2 shape."""

from __future__ import annotations

from typing import Any, Mapping, Protocol, TypedDict
import re
import time

from gage_eval.agent_runtime.serialization import to_json_compatible


class TraceTranslationContext(TypedDict, total=False):
    trial_id: str
    agent_info: Mapping[str, Any]
    final_metrics: Mapping[str, Any]
    primary_raw_trial: Mapping[str, Any]


class AgentTraceTranslator(Protocol):
    source_format: str

    def translate(
        self,
        raw_trace: Any,
        *,
        context: TraceTranslationContext | Mapping[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        ...


def fallback_minimal_step(
    *,
    trace_step: int,
    source_format: str,
    raw_step: Any,
    timestamp: int | None = None,
) -> dict[str, Any]:
    """Build the shared fallback for an unrecognized raw trace step."""

    raw_mapping = raw_step if isinstance(raw_step, Mapping) else {"value": raw_step}
    step = {
        "trace_step": trace_step,
        "trace_role": _fallback_trace_role(raw_mapping),
        "name": _fallback_name(raw_mapping),
        "input": None,
        "output": None,
        "status": _fallback_status(raw_mapping),
        "latency_ms": 0.0,
        "timestamp": int(timestamp if timestamp is not None else time.time()),
        "metadata": {
            f"raw_{_metadata_key(source_format)}_step": to_json_compatible(raw_step),
        },
    }
    return step


def _metadata_key(source_format: str) -> str:
    key = re.sub(r"[^0-9a-zA-Z]+", "_", source_format.strip().lower()).strip("_")
    return key or "trace"


def _fallback_name(raw_step: Mapping[str, Any]) -> str:
    for key in ("name", "tool_name", "function_name", "source", "role"):
        value = raw_step.get(key)
        if value is not None:
            return str(value)
    return "unknown"


def _fallback_trace_role(raw_step: Mapping[str, Any]) -> str:
    source = str(raw_step.get("source") or raw_step.get("role") or "").lower()
    if source in {"system", "user", "model", "environment", "verifier"}:
        return source
    if source in {"tool", "observation"}:
        return "tool"
    if source in {"agent", "assistant"}:
        return "assistant"
    return "assistant"


def _fallback_status(raw_step: Mapping[str, Any]) -> str:
    if raw_step.get("error_info") or raw_step.get("error"):
        return "error"
    status = str(raw_step.get("status") or "").lower()
    if status in {"success", "aborted"}:
        return status
    if status in {"error", "failed", "failure"}:
        return "error"
    return "success"


__all__ = [
    "AgentTraceTranslator",
    "TraceTranslationContext",
    "fallback_minimal_step",
]
