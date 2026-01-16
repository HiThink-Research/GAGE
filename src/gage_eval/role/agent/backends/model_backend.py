"""Agent backend wrapper that delegates to model backends."""

from __future__ import annotations

import asyncio
import threading
from typing import Any, Dict, List, Optional

from gage_eval.registry.utils import run_sync
from gage_eval.role.agent.backends.base import AgentBackend, normalize_agent_output
from gage_eval.role.model.backends import build_backend, wrap_backend
from gage_eval.role.model.backends.base_backend import Backend as ModelBackendBase


class ModelBackend(AgentBackend):
    """Agent backend that reuses model backends for tool-capable generation."""

    def __init__(self, config: Dict[str, Any]) -> None:
        backend = config.get("backend") or config.get("backend_spec")
        if backend is None:
            raise ValueError("model_backend requires 'backend' or 'backend_spec'")
        if isinstance(backend, dict):
            backend = wrap_backend(build_backend(backend))
        elif isinstance(backend, ModelBackendBase):
            backend = wrap_backend(backend)
        self._backend = backend
        self._default_sampling = dict(config.get("sampling_params") or {})
        self._force_tool_choice_mode = _resolve_force_tool_choice_mode(config.get("force_tool_choice"))

    def invoke(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        request = _build_backend_request(payload, self._default_sampling, self._force_tool_choice_mode)
        response = _call_backend(self._backend, request)
        result = normalize_agent_output(response)
        tool_calls = _extract_tool_calls(response)
        if tool_calls:
            result["tool_calls"] = tool_calls
        return result


def _build_backend_request(
    payload: Dict[str, Any],
    defaults: Dict[str, Any],
    force_tool_choice_mode: str,
) -> Dict[str, Any]:
    request: Dict[str, Any] = {
        "messages": payload.get("messages") or [],
        "sample": payload.get("sample") or {},
    }
    tools = payload.get("tools")
    if tools is not None:
        request["tools"] = tools
    tool_choice = payload.get("tool_choice")
    if force_tool_choice_mode != "never" and tools:
        if tool_choice is None or tool_choice == "auto":
            turn_index = _coerce_int(payload.get("turn_index"), default=1)
            if force_tool_choice_mode == "always":
                tool_choice = "required"
            elif force_tool_choice_mode == "first_turn" and turn_index <= 1:
                tool_choice = "required"
    if tool_choice is not None:
        request["tool_choice"] = tool_choice
    sampling_params = dict(defaults)
    sampling_params.update(payload.get("sampling_params") or {})
    if sampling_params:
        request["sampling_params"] = sampling_params
    return request


def _call_backend(backend: Any, request: Dict[str, Any]) -> Any:
    async_backend_call = getattr(backend, "ainvoke", None)
    if async_backend_call:
        return _run_async(async_backend_call(request))
    backend_call = getattr(backend, "invoke", None)
    if callable(backend_call):
        return backend_call(request)
    if callable(backend):
        return backend(request)
    raise RuntimeError("model_backend_missing_invoke")


def _extract_tool_calls(response: Any) -> List[Dict[str, Any]]:
    if not isinstance(response, dict):
        return []
    tool_calls = response.get("tool_calls")
    if isinstance(tool_calls, list) and tool_calls:
        return tool_calls
    raw = response.get("raw_response") or response.get("response") or response.get("data")
    return _extract_from_raw(raw)


def _run_async(awaitable):
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return run_sync(awaitable)
    return _run_async_in_thread(awaitable)


def _run_async_in_thread(awaitable):
    result: Dict[str, Any] = {}
    error: Dict[str, BaseException] = {}

    def runner() -> None:
        try:
            result["value"] = asyncio.run(awaitable)
        except BaseException as exc:
            error["value"] = exc

    thread = threading.Thread(target=runner, daemon=True)
    thread.start()
    thread.join()
    if error:
        raise error["value"]
    return result.get("value")


def _extract_from_raw(raw: Any) -> List[Dict[str, Any]]:
    if isinstance(raw, dict):
        if isinstance(raw.get("tool_calls"), list):
            return raw.get("tool_calls") or []
        choices = raw.get("choices")
        if isinstance(choices, list) and choices:
            message = choices[0].get("message") or choices[0].get("delta") or {}
            return _extract_from_message(message)
    return []


def _extract_from_message(message: Any) -> List[Dict[str, Any]]:
    if not isinstance(message, dict):
        return []
    tool_calls = message.get("tool_calls")
    if isinstance(tool_calls, list) and tool_calls:
        return tool_calls
    function_call = message.get("function_call")
    if isinstance(function_call, dict) and function_call.get("name"):
        return [
            {
                "id": "call_0",
                "type": "function",
                "function": {
                    "name": function_call.get("name"),
                    "arguments": function_call.get("arguments", {}),
                },
            }
        ]
    return []


def _resolve_force_tool_choice_mode(value: Any) -> str:
    if value is None:
        return "always"
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"first_turn", "first", "once"}:
            return "first_turn"
        if normalized in {"never", "false", "0", "off", "no"}:
            return "never"
        if normalized in {"always", "true", "1", "on", "yes"}:
            return "always"
    return "always" if bool(value) else "never"


def _coerce_int(value: Any, *, default: int) -> int:
    if value is None:
        return default
    try:
        return int(value)
    except (TypeError, ValueError):
        return default
