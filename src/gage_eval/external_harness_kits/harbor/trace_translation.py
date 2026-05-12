"""Harbor ATIF trace translation into AgentKitV2 trace steps."""

from __future__ import annotations

from datetime import datetime
import json
import time
from typing import Any, Mapping

from gage_eval.agent_runtime.serialization import to_json_compatible
from gage_eval.external_harness_kits.trace_translation import (
    TraceTranslationContext,
    fallback_minimal_step,
)


class HarborATIFTranslator:
    source_format = "ATIF-v1.7"

    def translate(
        self,
        raw_trace: Any,
        *,
        context: TraceTranslationContext | Mapping[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        del context
        steps = _atif_steps(raw_trace)
        trace: list[dict[str, Any]] = []
        index = 0
        while index < len(steps):
            step = steps[index]
            if not isinstance(step, Mapping):
                trace.append(_fallback_trace_step(len(trace) + 1, {"value": step}))
                index += 1
                continue

            tool_calls = _step_tool_calls(step)
            if tool_calls:
                next_step = steps[index + 1] if index + 1 < len(steps) else None
                for call in tool_calls:
                    trace.append(
                        _agentkit_trace_step(
                            len(trace) + 1,
                            trace_role="tool",
                            name=_tool_call_name(call) or _step_name(step),
                            input_payload=_tool_call_arguments(call),
                            output_payload=_tool_output(step, next_step),
                            status=_trace_status(step),
                            latency_ms=_trace_latency_ms(step),
                            timestamp=_trace_timestamp(step),
                            usage=_step_usage(step),
                            turn_index=_trace_turn_index(step),
                        )
                    )
                if _is_observation_step(next_step):
                    index += 2
                else:
                    index += 1
                continue

            if _is_observation_step(step):
                index += 1
                continue

            if _is_message_step(step):
                content = _step_message_content(step)
                trace.append(
                    _agentkit_trace_step(
                        len(trace) + 1,
                        trace_role=_trace_role(step),
                        name=_step_name(step),
                        input_payload=content,
                        output_payload={"content": content} if content is not None else None,
                        status=_trace_status(step),
                        latency_ms=_trace_latency_ms(step),
                        timestamp=_trace_timestamp(step),
                        usage=_step_usage(step),
                        turn_index=_trace_turn_index(step),
                    )
                )
                index += 1
                continue

            trace.append(_fallback_trace_step(len(trace) + 1, dict(step)))
            index += 1
        return trace


def _atif_steps(raw_trace: Any) -> list[Any]:
    if isinstance(raw_trace, Mapping):
        for key in ("steps", "trajectory", "messages", "events"):
            value = raw_trace.get(key)
            if isinstance(value, list):
                return list(value)
        return [dict(raw_trace)] if raw_trace else []
    if isinstance(raw_trace, list):
        return list(raw_trace)
    return []


def _agentkit_trace_step(
    trace_step: int,
    *,
    trace_role: str,
    name: str | None,
    input_payload: Any,
    output_payload: Any,
    status: str,
    latency_ms: float,
    timestamp: int,
    usage: Mapping[str, Any] | None = None,
    turn_index: int | None = None,
) -> dict[str, Any]:
    step: dict[str, Any] = {
        "trace_step": trace_step,
        "trace_role": trace_role,
        "name": name or "",
        "input": to_json_compatible(input_payload),
        "output": to_json_compatible(output_payload),
        "status": status,
        "latency_ms": float(latency_ms or 0.0),
        "timestamp": timestamp,
    }
    if turn_index is not None:
        step["turn_index"] = turn_index
    if usage:
        if usage.get("input_tokens") is not None:
            step["input_tokens"] = usage.get("input_tokens")
        if usage.get("output_tokens") is not None:
            step["output_tokens"] = usage.get("output_tokens")
        if usage.get("cost_usd") is not None:
            step["cost_usd"] = usage.get("cost_usd")
    return step


def _fallback_trace_step(trace_step: int, raw_step: Mapping[str, Any]) -> dict[str, Any]:
    return fallback_minimal_step(
        trace_step=trace_step,
        source_format=HarborATIFTranslator.source_format,
        raw_step=dict(raw_step),
        timestamp=_trace_timestamp(raw_step),
    )


def _step_tool_calls(step: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    tool_calls = step.get("tool_calls")
    if isinstance(tool_calls, list):
        return [call for call in tool_calls if isinstance(call, Mapping)]
    message = step.get("message")
    if isinstance(message, Mapping):
        message_calls = message.get("tool_calls")
        if isinstance(message_calls, list):
            return [call for call in message_calls if isinstance(call, Mapping)]
    if step.get("type") == "tool_call" or step.get("tool_name"):
        return [step]
    return []


def _tool_call_name(call: Mapping[str, Any]) -> str:
    function = call.get("function")
    if isinstance(function, Mapping):
        name = function.get("name")
        if name is not None:
            return str(name)
    for key in ("function_name", "tool_name", "name"):
        if call.get(key) is not None:
            return str(call[key])
    return ""


def _tool_call_arguments(call: Mapping[str, Any]) -> Any:
    function = call.get("function")
    if isinstance(function, Mapping) and "arguments" in function:
        return _maybe_json(function.get("arguments"))
    for key in ("arguments", "args", "input"):
        if key in call:
            return _maybe_json(call.get(key))
    return {}


def _tool_output(step: Mapping[str, Any], next_step: Any = None) -> Any:
    for key in ("observation", "tool_result", "output", "result"):
        if key in step and step.get(key) is not None:
            return _observation_payload(step.get(key))
    if _is_observation_step(next_step):
        next_mapping = _mapping(next_step)
        return _observation_payload(next_mapping.get("observation", next_mapping.get("message")))
    return None


def _observation_payload(value: Any) -> Any:
    if isinstance(value, Mapping):
        content = _first_observation_content(value)
        if content is not None:
            return {"content": content}
        if "content" in value:
            return {"content": value.get("content")}
        return dict(value)
    if isinstance(value, list):
        content = _first_observation_content({"results": value})
        return {"content": content} if content is not None else list(value)
    if value is None:
        return None
    return {"content": str(value)}


def _first_observation_content(value: Mapping[str, Any]) -> str | None:
    results = value.get("results")
    if isinstance(results, list):
        parts = []
        for result in results:
            if isinstance(result, Mapping) and isinstance(result.get("content"), str):
                parts.append(result["content"])
            elif isinstance(result, str):
                parts.append(result)
        if parts:
            return "\n".join(parts)
    output = value.get("output")
    if isinstance(output, Mapping):
        return _first_observation_content(output)
    if isinstance(output, str):
        return output
    return None


def _is_observation_step(value: Any) -> bool:
    if not isinstance(value, Mapping):
        return False
    source = str(value.get("source") or "").lower()
    return source in {"tool", "environment", "observation"} or (
        "observation" in value and not _step_tool_calls(value)
    )


def _is_message_step(step: Mapping[str, Any]) -> bool:
    if "content" in step or "text" in step:
        return True
    if "message" not in step:
        return False
    message = step.get("message")
    if isinstance(message, Mapping):
        return any(key in message for key in ("content", "text", "message"))
    return message is not None


def _step_message_content(step: Mapping[str, Any]) -> Any:
    message = step.get("message")
    if isinstance(message, Mapping):
        for key in ("content", "text", "message"):
            if key in message:
                return message.get(key)
        return dict(message)
    if message is not None:
        return message
    for key in ("content", "text"):
        if key in step:
            return step.get(key)
    return None


def _trace_role(step: Mapping[str, Any]) -> str:
    source = str(step.get("source") or step.get("role") or "").lower()
    if source in {"system", "user"}:
        return source
    if source in {"tool", "environment", "observation"}:
        return "tool"
    return "assistant"


def _step_name(step: Mapping[str, Any]) -> str:
    for key in ("name", "tool_name", "function_name", "source", "role"):
        if step.get(key) is not None:
            return str(step[key])
    return "step"


def _trace_status(step: Mapping[str, Any]) -> str:
    if step.get("error_info") or step.get("error"):
        return "error"
    status = str(step.get("status") or "").lower()
    if status in {"success", "aborted"}:
        return status
    if status in {"error", "failed", "failure"}:
        return "error"
    return "success"


def _trace_latency_ms(step: Mapping[str, Any]) -> float:
    metadata = _mapping(step.get("metadata"))
    metrics = _mapping(step.get("metrics"))
    for value in (step.get("duration_ms"), metadata.get("latency_ms"), metrics.get("latency_ms")):
        try:
            return float(value)
        except (TypeError, ValueError):
            continue
    return 0.0


def _trace_timestamp(step: Mapping[str, Any]) -> int:
    value = step.get("timestamp")
    if isinstance(value, (int, float)):
        return int(value)
    if isinstance(value, str) and value:
        normalized = value.replace("Z", "+00:00")
        try:
            return int(datetime.fromisoformat(normalized).timestamp())
        except ValueError:
            pass
    return int(time.time())


def _trace_turn_index(step: Mapping[str, Any]) -> int | None:
    metadata = _mapping(step.get("metadata"))
    for value in (metadata.get("episode_index"), metadata.get("turn_index"), step.get("turn_index")):
        try:
            return int(value)
        except (TypeError, ValueError):
            continue
    return None


def _step_usage(step: Mapping[str, Any]) -> dict[str, Any]:
    metrics = _mapping(step.get("metrics"))
    if not metrics:
        return {}
    usage: dict[str, Any] = {}
    input_tokens = metrics.get("input_tokens", metrics.get("prompt_tokens"))
    output_tokens = metrics.get("output_tokens", metrics.get("completion_tokens"))
    if input_tokens is not None:
        usage["input_tokens"] = input_tokens
    if output_tokens is not None:
        usage["output_tokens"] = output_tokens
    if metrics.get("cost_usd") is not None:
        usage["cost_usd"] = metrics.get("cost_usd")
    return usage


def _maybe_json(value: Any) -> Any:
    if isinstance(value, str):
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            return value
    return value


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


__all__ = ["HarborATIFTranslator"]
