"""Agent backend wrapper that delegates to model backends."""

from __future__ import annotations

import asyncio
import ast
import json
import re
from typing import Any, Dict, Iterable, List, Optional

from gage_eval.registry.utils import ensure_async, run_sync
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
        response = _call_backend_sync(self._backend, request)
        return _normalize_backend_response(response, request.get("tools"))

    async def ainvoke(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        request = _build_backend_request(payload, self._default_sampling, self._force_tool_choice_mode)
        response = await _call_backend_async(self._backend, request)
        return _normalize_backend_response(response, request.get("tools"))


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


def _call_backend_sync(backend: Any, request: Dict[str, Any]) -> Any:
    async_backend_call = getattr(backend, "ainvoke", None)
    if callable(async_backend_call):
        _raise_if_active_event_loop()
        return run_sync(async_backend_call(request))
    backend_call = getattr(backend, "invoke", None)
    if callable(backend_call):
        return backend_call(request)
    if callable(backend):
        return backend(request)
    raise RuntimeError("model_backend_missing_invoke")


async def _call_backend_async(backend: Any, request: Dict[str, Any]) -> Any:
    async_backend_call = getattr(backend, "ainvoke", None)
    if callable(async_backend_call):
        return await async_backend_call(request)
    backend_call = getattr(backend, "invoke", None)
    if callable(backend_call):
        return await ensure_async(backend_call)(request)
    if callable(backend):
        return await ensure_async(backend)(request)
    raise RuntimeError("model_backend_missing_invoke")


def _normalize_backend_response(response: Any, tools: Any = None) -> Dict[str, Any]:
    result = normalize_agent_output(response)
    tool_calls = _extract_tool_calls(response)
    if not tool_calls:
        tool_calls = _extract_tool_calls_from_answer(result.get("answer"))
        if tools:
            tool_calls = _filter_tool_calls_by_schema(tool_calls, tools)
    if tool_calls:
        result["tool_calls"] = tool_calls
        if isinstance(result.get("answer"), str):
            result["raw_answer"] = result.get("answer")
            result["answer"] = ""
    return result


def _filter_tool_calls_by_schema(
    tool_calls: List[Dict[str, Any]], tools: Any
) -> List[Dict[str, Any]]:
    if not tool_calls:
        return []
    allowed_names = _extract_tool_names(tools)
    if not allowed_names:
        return []
    return [
        call
        for call in tool_calls
        if _tool_call_name(call) in allowed_names
    ]


def _extract_tool_names(tools: Any) -> set[str]:
    if isinstance(tools, dict):
        tools = (tools,)
    if not isinstance(tools, (list, tuple)):
        return set()
    names: set[str] = set()
    for tool in tools:
        if not isinstance(tool, dict):
            continue
        function_name = tool.get("function", {}).get("name") if isinstance(tool.get("function"), dict) else None
        name = function_name if function_name else tool.get("name")
        if isinstance(name, str):
            stripped = name.strip()
            if stripped:
                names.add(stripped)
    return names


def _tool_call_name(tool_call: Dict[str, Any]) -> Optional[str]:
    function = tool_call.get("function")
    if isinstance(function, dict):
        name = function.get("name")
    else:
        name = tool_call.get("name")
    if isinstance(name, str):
        return name.strip()
    return None


def _raise_if_active_event_loop() -> None:
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return
    raise RuntimeError("run_sync() cannot be used inside an active event loop")


def _extract_tool_calls(response: Any) -> List[Dict[str, Any]]:
    if not isinstance(response, dict):
        return []
    tool_calls = response.get("tool_calls")
    if isinstance(tool_calls, list) and tool_calls:
        return tool_calls
    raw = response.get("raw_response") or response.get("response") or response.get("data")
    return _extract_from_raw(raw)


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


def _extract_tool_calls_from_answer(answer: Any) -> List[Dict[str, Any]]:
    if not isinstance(answer, str) or not answer.strip():
        return []
    calls: List[Dict[str, Any]] = []
    calls.extend(_extract_xml_function_calls(answer))
    for raw_payload in _iter_tool_call_payloads(answer):
        parsed = _parse_tool_payload(raw_payload)
        calls.extend(_normalize_text_tool_calls(parsed))
        if not calls:
            calls.extend(_parse_pythonic_tool_calls(raw_payload))
    return _dedupe_tool_calls(calls)


def _dedupe_tool_calls(calls: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    deduped: List[Dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    for call in calls:
        if not isinstance(call, dict):
            continue
        function = call.get("function") if isinstance(call.get("function"), dict) else {}
        name = str(function.get("name") or call.get("name") or "")
        arguments = function.get("arguments", call.get("arguments", {}))
        key = (name, _canonical_tool_arguments(arguments))
        if key in seen:
            continue
        seen.add(key)
        deduped.append(call)
    return deduped


def _canonical_tool_arguments(arguments: Any) -> str:
    try:
        return json.dumps(arguments, sort_keys=True, default=str)
    except TypeError:
        return str(arguments)


def _iter_tool_call_payloads(answer: str) -> List[str]:
    payloads: List[str] = []
    for tag in ("tool_call", "tool_calls", "function_call", "function_calls"):
        payloads.extend(
            re.findall(
                rf"<{tag}>\s*(.*?)\s*</{tag}>",
                answer,
                flags=re.DOTALL | re.IGNORECASE,
            )
        )
    payloads.extend(
        re.findall(
            r"\[TOOL_CALLS\]\s*(.*?)(?:\[/TOOL_CALLS\]|$)",
            answer,
            flags=re.DOTALL | re.IGNORECASE,
        )
    )
    payloads.extend(
        re.findall(
            r"```(?:json|tool_calls?|function_calls?)?\s*(.*?)```",
            answer,
            flags=re.DOTALL | re.IGNORECASE,
        )
    )
    stripped = answer.strip()
    if stripped.startswith(("{", "[")) and stripped.endswith(("}", "]")):
        payloads.append(stripped)
    for tail in _iter_think_tails(answer):
        balanced = _extract_balanced_json(tail)
        if balanced is not None:
            payloads.append(balanced)
        if _has_pythonic_tool_call(tail):
            payloads.append(tail)
    return [payload for payload in payloads if payload.strip()]


def _iter_think_tails(answer: str) -> Iterable[str]:
    for match in re.finditer(r"</think\s*>", answer, flags=re.IGNORECASE):
        tail = answer[match.end() :].strip()
        if tail:
            yield tail


def _has_pythonic_tool_call(text: str) -> bool:
    return any(True for _ in _iter_pythonic_call_expressions(text))


def _parse_tool_payload(raw: str) -> Any:
    stripped = raw.strip()
    parsed = _try_parse_json(stripped)
    if parsed is not None:
        return parsed
    balanced = _extract_balanced_json(stripped)
    if balanced is not None:
        parsed = _try_parse_json(balanced)
        if parsed is not None:
            return parsed
    try:
        return ast.literal_eval(stripped)
    except Exception:
        return None


def _try_parse_json(raw: str) -> Any:
    try:
        return json.loads(raw)
    except Exception:
        return None


def _normalize_text_tool_calls(payload: Any) -> List[Dict[str, Any]]:
    if isinstance(payload, list):
        calls: List[Dict[str, Any]] = []
        for item in payload:
            calls.extend(_normalize_text_tool_calls(item))
        return calls
    if not isinstance(payload, dict):
        return []
    tool_calls = payload.get("tool_calls")
    if isinstance(tool_calls, list) and tool_calls:
        return [
            normalized
            for call in tool_calls
            for normalized in _normalize_text_tool_calls(call)
        ]
    if isinstance(payload.get("function"), dict):
        function = payload["function"]
        name = function.get("name")
        arguments = function.get("arguments", {})
        if not name:
            return []
        return [_build_openai_tool_call(name, arguments, call_id=payload.get("id"))]
    function_call = payload.get("function_call")
    if isinstance(function_call, dict):
        payload = function_call
    name = payload.get("name") or payload.get("tool_name") or payload.get("function_name")
    arguments = payload.get("arguments", payload.get("parameters", payload.get("args", {})))
    if not name:
        return []
    return [_build_openai_tool_call(name, arguments, call_id=payload.get("id"))]


def _build_openai_tool_call(name: Any, arguments: Any, *, call_id: Any = None) -> Dict[str, Any]:
    return {
        "id": str(call_id or "call_0"),
        "type": "function",
        "function": {
            "name": str(name),
            "arguments": arguments,
        },
    }


def _extract_balanced_json(text: str) -> Optional[str]:
    starts = [idx for idx in (text.find("{"), text.find("[")) if idx >= 0]
    if not starts:
        return None
    start = min(starts)
    opener = text[start]
    closer = "}" if opener == "{" else "]"
    depth = 0
    in_string = False
    escape = False
    for idx in range(start, len(text)):
        char = text[idx]
        if in_string:
            if escape:
                escape = False
            elif char == "\\":
                escape = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char == opener:
            depth += 1
        elif char == closer:
            depth -= 1
            if depth == 0:
                return text[start : idx + 1]
    return None


def _extract_xml_function_calls(answer: str) -> List[Dict[str, Any]]:
    calls: List[Dict[str, Any]] = []
    function_blocks = re.findall(
        r"<function(?:\s+name=\"([^\"]+)\"|=([A-Za-z_][\w.-]*))>\s*(.*?)\s*</function>",
        answer,
        flags=re.DOTALL | re.IGNORECASE,
    )
    for quoted_name, equals_name, body in function_blocks:
        name = quoted_name or equals_name
        if not name:
            continue
        parsed_body = _parse_tool_payload(body)
        if isinstance(parsed_body, dict):
            arguments = parsed_body.get("arguments", parsed_body)
        else:
            arguments = _extract_xml_parameters(body)
        calls.append(_build_openai_tool_call(name, arguments, call_id=f"call_{len(calls)}"))
    return calls


def _extract_xml_parameters(body: str) -> Dict[str, Any]:
    arguments: Dict[str, Any] = {}
    parameter_blocks = re.findall(
        r"<parameter(?:\s+name=\"([^\"]+)\"|=([A-Za-z_][\w.-]*))>\s*(.*?)\s*</parameter>",
        body,
        flags=re.DOTALL | re.IGNORECASE,
    )
    for quoted_name, equals_name, value in parameter_blocks:
        name = quoted_name or equals_name
        if not name:
            continue
        parsed_value = _parse_tool_payload(value)
        arguments[name] = parsed_value if parsed_value is not None else value.strip()
    return arguments


def _parse_pythonic_tool_calls(raw: str) -> List[Dict[str, Any]]:
    calls: List[Dict[str, Any]] = []
    for expression in _iter_pythonic_call_expressions(raw):
        try:
            node = ast.parse(expression, mode="eval")
        except SyntaxError:
            continue
        call = node.body
        if not isinstance(call, ast.Call):
            continue
        name = _pythonic_call_name(call.func)
        if not name:
            continue
        arguments: Dict[str, Any] = {}
        for idx, arg in enumerate(call.args):
            arguments[f"arg{idx}"] = _literal_ast_value(arg)
        for keyword in call.keywords:
            if keyword.arg:
                arguments[keyword.arg] = _literal_ast_value(keyword.value)
        calls.append(_build_openai_tool_call(name, arguments, call_id=f"call_{len(calls)}"))
    return calls


def _iter_pythonic_call_expressions(raw: str) -> Iterable[str]:
    for line in raw.splitlines():
        stripped = line.strip().rstrip(",")
        if re.match(r"^[A-Za-z_][\w.]*\s*\(.*\)$", stripped):
            yield stripped


def _pythonic_call_name(node: ast.AST) -> str:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        parent = _pythonic_call_name(node.value)
        return f"{parent}.{node.attr}" if parent else node.attr
    return ""


def _literal_ast_value(node: ast.AST) -> Any:
    try:
        return ast.literal_eval(node)
    except Exception:
        return ast.unparse(node) if hasattr(ast, "unparse") else None


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
