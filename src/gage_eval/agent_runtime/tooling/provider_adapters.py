from __future__ import annotations

import ast
import json
import re
from typing import Any

from gage_eval.agent_runtime.tooling.contracts import ToolCallIR, ToolResultIR, ToolingError, compact_json


class OpenAIProviderAdapter:
    """OpenAI-like provider adapter covering request/response/tool responsibilities."""

    def serialize_request(self, *, messages: list[dict[str, Any]], tools: list[dict[str, Any]], **kwargs: Any) -> dict[str, Any]:
        request = {"messages": list(messages), "tools": list(tools)}
        request.update(kwargs)
        return request

    def capture_raw_response(self, response: Any) -> Any:
        return response

    def extract_tool_calls(
        self,
        raw_response: Any,
        *,
        turn_index: int,
        required: bool = False,
        require_call_id: bool = False,
    ) -> list[ToolCallIR]:
        payload = _coerce_response_payload(raw_response)
        raw_calls = _extract_raw_tool_calls(payload)
        if required and not raw_calls:
            raise ToolingError("client_execution.tool_protocol_missing_call", "required tool call is missing")
        calls: list[ToolCallIR] = []
        for index, raw_call in enumerate(raw_calls, start=1):
            calls.append(
                ToolCallIR.from_provider_call(
                    raw_call,
                    turn_index=turn_index,
                    call_index=index,
                    provider="openai",
                    require_call_id=require_call_id,
                )
            )
        return calls

    def serialize_tool_result(self, result: ToolResultIR) -> dict[str, Any]:
        return ToolResultIR.serialize_for_injection(
            result,
            serializer=lambda item: {
                "role": "tool",
                "tool_call_id": item.call_id,
                "name": item.name,
                "content": json.dumps(item.output_json, ensure_ascii=True, separators=(",", ":")),
            },
        )

    def extract_final_answer(self, raw_response: Any) -> str:
        payload = _coerce_response_payload(raw_response)
        if isinstance(payload.get("answer"), str):
            return payload["answer"]
        if isinstance(payload.get("content"), str):
            return payload["content"]
        if isinstance(payload.get("output_text"), str):
            return payload["output_text"]
        message = _first_message(payload)
        if isinstance(message, dict) and isinstance(message.get("content"), str):
            return message["content"]
        return ""


class Tau2ToolDialectParser:
    """Parser for Tau2/tau2patch tool-call dialect variants."""

    def parse(
        self,
        message: Any,
        *,
        dialect: str = "auto",
        turn_index: int,
        call_index: int = 1,
    ) -> list[ToolCallIR]:
        parser_name = f"_parse_{dialect}"
        parser = getattr(self, parser_name, None)
        if callable(parser):
            call = parser(message, turn_index=turn_index, call_index=call_index)
            if call is None:
                return []
            if isinstance(call, list):
                return _dedupe_tool_calls(call)
            return [call]
        return self._parse_auto(message, turn_index=turn_index, call_index=call_index)

    def _parse_auto(self, message: Any, *, turn_index: int, call_index: int) -> list[ToolCallIR]:
        for dialect in (
            "openai_like",
            "harmony_xml",
            "qwen_xml",
            "qwen_arg_key",
            "glm_arg_key",
            "minimax",
            "function_gemma",
            "gemma",
            "fenced_json",
            "raw_json",
            "pythonic_call",
        ):
            try:
                calls = self.parse(message, dialect=dialect, turn_index=turn_index, call_index=call_index)
            except ToolingError:
                continue
            if calls:
                return calls
        return []

    def _parse_openai_like(self, message: Any, *, turn_index: int, call_index: int) -> list[ToolCallIR] | None:
        if not isinstance(message, dict):
            return None
        calls = OpenAIProviderAdapter().extract_tool_calls(message, turn_index=turn_index)
        return calls or None

    def _parse_qwen_xml(self, message: Any, *, turn_index: int, call_index: int) -> ToolCallIR | None:
        if not isinstance(message, str):
            return None
        match = re.search(
            r"<tool_call>\s*<name>(?P<name>.*?)</name>\s*<arguments>(?P<arguments>.*?)</arguments>\s*</tool_call>",
            message,
            flags=re.DOTALL,
        )
        if match:
            return ToolCallIR.from_provider_call(
                {
                    "name": match.group("name").strip(),
                    "arguments": match.group("arguments").strip(),
                },
                turn_index=turn_index,
                call_index=call_index,
                provider="qwen_xml",
            )
        arg_key_payload = _parse_arg_key_xml(message)
        if arg_key_payload is None:
            return None
        return ToolCallIR.from_provider_call(
            {"name": "respond", "arguments": arg_key_payload},
            turn_index=turn_index,
            call_index=call_index,
            provider="qwen_xml",
        )

    def _parse_harmony_xml(self, message: Any, *, turn_index: int, call_index: int) -> list[ToolCallIR] | None:
        if not isinstance(message, str):
            return None
        calls: list[ToolCallIR] = []
        for match in re.finditer(
            r"<function=(?P<name>[^>\n]+)>\s*(?P<body>.*?)</function>",
            message,
            flags=re.DOTALL,
        ):
            arguments = {
                param.group("key").split("<|channel|>", 1)[0].strip(): param.group("value").strip()
                for param in re.finditer(
                    r"<parameter=(?P<key>[^>\n]+)>\s*(?P<value>.*?)\s*</parameter>",
                    match.group("body"),
                    flags=re.DOTALL,
                )
            }
            calls.append(
                ToolCallIR.from_provider_call(
                    {"name": match.group("name").strip(), "arguments": arguments},
                    turn_index=turn_index,
                    call_index=call_index + len(calls),
                    provider="harmony_xml",
                )
            )
        return calls or None

    def _parse_qwen_arg_key(self, message: Any, *, turn_index: int, call_index: int) -> ToolCallIR | None:
        if isinstance(message, str):
            payload = _parse_arg_key_xml(message)
            if payload is None:
                return None
            return ToolCallIR.from_provider_call(
                {"name": "respond", "arguments": payload},
                turn_index=turn_index,
                call_index=call_index,
                provider="qwen_arg_key",
            )
        if not isinstance(message, dict) or "name" not in message:
            return None
        return ToolCallIR.from_provider_call(message, turn_index=turn_index, call_index=call_index, provider="qwen_arg_key")

    def _parse_glm_arg_key(self, message: Any, *, turn_index: int, call_index: int) -> ToolCallIR | None:
        if isinstance(message, str):
            payload = _parse_arg_key_xml(message)
            if payload is None:
                return None
            return ToolCallIR.from_provider_call(
                {"name": "respond", "arguments": payload},
                turn_index=turn_index,
                call_index=call_index,
                provider="glm_arg_key",
            )
        if not isinstance(message, dict) or not isinstance(message.get("function_call"), dict):
            return None
        return ToolCallIR.from_provider_call(message, turn_index=turn_index, call_index=call_index, provider="glm_arg_key")

    def _parse_minimax(self, message: Any, *, turn_index: int, call_index: int) -> ToolCallIR | None:
        if not isinstance(message, str):
            return self._parse_glm_arg_key(message, turn_index=turn_index, call_index=call_index)
        match = re.search(
            r"<minimax:tool_call>\s*<invoke\s+name=[\"'](?P<name>[^\"']+)[\"']>(?P<body>.*?)</invoke>\s*</minimax:tool_call>",
            message,
            flags=re.DOTALL,
        )
        if not match:
            return None
        arguments = {
            arg.group("name"): arg.group("value").strip()
            for arg in re.finditer(
                r"<arg\s+name=[\"'](?P<name>[^\"']+)[\"']>(?P<value>.*?)</arg>",
                match.group("body"),
                flags=re.DOTALL,
            )
        }
        return ToolCallIR.from_provider_call(
            {"name": match.group("name"), "arguments": arguments},
            turn_index=turn_index,
            call_index=call_index,
            provider="minimax",
        )

    def _parse_gemma(self, message: Any, *, turn_index: int, call_index: int) -> ToolCallIR | list[ToolCallIR] | None:
        if not isinstance(message, str):
            return None
        match = re.search(r"<tool_call>(?P<payload>.*?)</tool_call>", message, flags=re.DOTALL)
        if match:
            payload_text = match.group("payload").strip()
            try:
                payload = _parse_json_text(payload_text)
                return ToolCallIR.from_provider_call(payload, turn_index=turn_index, call_index=call_index, provider="gemma")
            except ToolingError:
                gemma_calls = _parse_gemma_call_chunks(payload_text, turn_index=turn_index, call_index=call_index)
                return gemma_calls[0] if len(gemma_calls) == 1 else gemma_calls or None
        calls = _parse_gemma_call_chunks(message, turn_index=turn_index, call_index=call_index)
        return calls[0] if len(calls) == 1 else calls or None

    def _parse_fenced_json(self, message: Any, *, turn_index: int, call_index: int) -> list[ToolCallIR] | None:
        if not isinstance(message, str):
            return None
        match = re.search(r"```(?:json)?\s*(?P<payload>\{.*?\})\s*```", message, flags=re.DOTALL)
        if not match:
            return None
        payload = _parse_json_text(match.group("payload"))
        return _calls_from_payload(payload, turn_index=turn_index, call_index=call_index, provider="fenced_json")

    def _parse_raw_json(self, message: Any, *, turn_index: int, call_index: int) -> list[ToolCallIR] | None:
        if not isinstance(message, str):
            return None
        for candidate in _iter_json_text_candidates(message):
            if not candidate.startswith("{"):
                continue
            try:
                payload = _parse_json_text(candidate)
            except ToolingError:
                continue
            calls = _calls_from_payload(payload, turn_index=turn_index, call_index=call_index, provider="raw_json")
            if calls:
                return calls
        return None

    def _parse_pythonic_call(self, message: Any, *, turn_index: int, call_index: int) -> ToolCallIR | None:
        if not isinstance(message, str):
            return None
        try:
            expression = ast.parse(message.strip(), mode="eval")
        except SyntaxError:
            return None
        if not isinstance(expression.body, ast.Call):
            return None
        call = expression.body
        if isinstance(call.func, ast.Name):
            name = call.func.id
        elif isinstance(call.func, ast.Attribute):
            name = call.func.attr
        else:
            return None
        arguments: dict[str, Any] = {}
        for index, arg in enumerate(call.args, start=1):
            arguments[f"arg{index}"] = ast.literal_eval(arg)
        for keyword in call.keywords:
            if keyword.arg:
                arguments[keyword.arg] = ast.literal_eval(keyword.value)
        return ToolCallIR.from_provider_call(
            {"name": name, "arguments": compact_json(arguments)},
            turn_index=turn_index,
            call_index=call_index,
            provider="pythonic_call",
        )

    def _parse_function_gemma(self, message: Any, *, turn_index: int, call_index: int) -> ToolCallIR | None:
        if not isinstance(message, str):
            return None
        match = re.search(r"<\|tool_call>(?P<body>.*?)<tool_call\|>", message, flags=re.DOTALL)
        if not match:
            return None
        return self._parse_gemma(match.group("body").strip(), turn_index=turn_index, call_index=call_index)

    def _parse_plain_text(self, message: Any, *, turn_index: int, call_index: int) -> None:
        return None


def _coerce_response_payload(raw_response: Any) -> dict[str, Any]:
    if isinstance(raw_response, dict):
        inner_response = raw_response.get("raw_response")
        if isinstance(inner_response, dict) and (
            "choices" in inner_response
            or "tool_calls" in inner_response
            or "function_call" in inner_response
        ):
            return _coerce_response_payload(inner_response)
        return raw_response
    model_dump = getattr(raw_response, "model_dump", None)
    if callable(model_dump):
        try:
            return _coerce_response_payload(model_dump(mode="json"))
        except TypeError:
            try:
                return _coerce_response_payload(model_dump())
            except TypeError:
                pass
    if isinstance(raw_response, str):
        try:
            parsed = json.loads(raw_response)
        except json.JSONDecodeError as exc:
            raise ToolingError(
                "client_execution.tool_protocol_parse_error",
                "provider raw response is not valid JSON",
            ) from exc
        if isinstance(parsed, dict):
            return parsed
    raise ToolingError("client_execution.tool_protocol_parse_error", "provider raw response must be a dict")


def _extract_raw_tool_calls(payload: dict[str, Any]) -> list[dict[str, Any]]:
    direct = payload.get("tool_calls")
    if isinstance(direct, list):
        return [call for call in direct if isinstance(call, dict)]
    function_call = payload.get("function_call")
    if isinstance(function_call, dict):
        return [{"function": function_call}]
    message = _first_message(payload)
    if isinstance(message, dict) and isinstance(message.get("tool_calls"), list):
        return [call for call in message["tool_calls"] if isinstance(call, dict)]
    if isinstance(message, dict) and isinstance(message.get("function_call"), dict):
        return [{"function": message["function_call"]}]
    return []


def _first_message(payload: dict[str, Any]) -> dict[str, Any] | None:
    choices = payload.get("choices")
    if isinstance(choices, list) and choices:
        choice = choices[0]
        if isinstance(choice, dict):
            message = choice.get("message") or choice.get("delta")
            if isinstance(message, dict):
                return message
    return None


def _parse_json_text(text: str) -> dict[str, Any]:
    try:
        payload = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ToolingError("client_execution.tool_protocol_parse_error", "tool call JSON parse failed") from exc
    if not isinstance(payload, dict):
        raise ToolingError("client_execution.tool_protocol_parse_error", "tool call JSON must be an object")
    return payload


def _calls_from_payload(payload: dict[str, Any], *, turn_index: int, call_index: int, provider: str) -> list[ToolCallIR]:
    if "tool" in payload and "name" not in payload:
        payload = {**payload, "name": payload["tool"]}
    raw_calls = payload.get("tool_calls")
    if isinstance(raw_calls, list) and raw_calls:
        calls: list[ToolCallIR] = []
        for offset, raw_call in enumerate(raw_calls):
            if not isinstance(raw_call, dict):
                continue
            if "tool" in raw_call and "name" not in raw_call:
                raw_call = {**raw_call, "name": raw_call["tool"]}
            calls.append(
                ToolCallIR.from_provider_call(
                    raw_call,
                    turn_index=turn_index,
                    call_index=call_index + offset,
                    provider=provider,
                )
            )
        return calls
    return [ToolCallIR.from_provider_call(payload, turn_index=turn_index, call_index=call_index, provider=provider)]


def _dedupe_tool_calls(calls: list[ToolCallIR]) -> list[ToolCallIR]:
    seen: set[tuple[str, str]] = set()
    deduped: list[ToolCallIR] = []
    for call in calls:
        key = (call.name, call.arguments_json)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(call)
    return deduped


def _iter_json_text_candidates(message: str) -> list[str]:
    stripped = message.strip()
    candidates = [stripped] if stripped else []
    candidates.extend(_iter_think_tails(stripped))
    return candidates


def _iter_think_tails(message: str) -> list[str]:
    tails: list[str] = []
    for match in re.finditer(r"</think\s*>\s*(?P<tail>.*)", message, flags=re.DOTALL | re.IGNORECASE):
        tail = match.group("tail").strip()
        if tail:
            tails.append(tail)
    return tails


def _parse_gemma_call_chunks(text: str | None, *, turn_index: int, call_index: int) -> list[ToolCallIR]:
    if text is None:
        return []
    calls: list[ToolCallIR] = []
    for name, body in _iter_gemma_call_chunks(text):
        calls.append(
            ToolCallIR.from_provider_call(
                {"name": name, "arguments": _parse_loose_object(body)},
                turn_index=turn_index,
                call_index=call_index + len(calls),
                provider="gemma",
            )
        )
    return calls


def _iter_gemma_call_chunks(text: str) -> list[tuple[str, str]]:
    chunks: list[tuple[str, str]] = []
    for match in re.finditer(
        r"(?m)(?:^|\n)\s*(?:call:)?(?P<name>[A-Za-z_][\w]*)\s*\{",
        text,
    ):
        open_brace = match.end() - 1
        body = _extract_balanced_brace_body(text, open_brace)
        if body is None:
            continue
        chunks.append((match.group("name"), body))
    return chunks


def _extract_balanced_brace_body(text: str, open_brace: int) -> str | None:
    depth = 0
    quote: str | None = None
    escape = False
    for index in range(open_brace, len(text)):
        char = text[index]
        if quote:
            if escape:
                escape = False
            elif char == "\\":
                escape = True
            elif char == quote:
                quote = None
            continue
        if char in {"'", '"'}:
            quote = char
            continue
        if char == "{":
            depth += 1
            continue
        if char == "}":
            depth -= 1
            if depth == 0:
                return text[open_brace + 1 : index]
    return None


def _parse_arg_key_xml(message: str) -> dict[str, Any] | None:
    pairs = list(
        re.finditer(
            r"<arg_key>(?P<key>.*?)</arg_key>\s*<arg_value>(?P<value>.*?)</arg_value>",
            message,
            flags=re.DOTALL,
        )
    )
    if not pairs:
        return None
    return {pair.group("key").strip(): pair.group("value").strip() for pair in pairs}


def _parse_loose_object(body: str) -> dict[str, Any]:
    body = body.strip()
    if not body:
        return {}
    arguments: dict[str, Any] = {}
    for part in _split_loose_arguments(body):
        if not part.strip() or ":" not in part:
            continue
        key, value = part.split(":", 1)
        arguments[key.strip().strip("\"'")] = _parse_loose_value(value.strip())
    return arguments


def _split_loose_arguments(body: str) -> list[str]:
    parts: list[str] = []
    current: list[str] = []
    quote: str | None = None
    for index, char in enumerate(body):
        if quote:
            current.append(char)
            if char == quote:
                quote = None
            continue
        if char in {"\"", "'"}:
            quote = char
            current.append(char)
            continue
        if char == "," and _looks_like_next_loose_key(body, index + 1):
            parts.append("".join(current))
            current = []
            continue
        current.append(char)
    if current:
        parts.append("".join(current))
    return parts


def _looks_like_next_loose_key(body: str, start: int) -> bool:
    return re.match(r"\s*['\"]?[A-Za-z_][\w-]*['\"]?\s*:", body[start:]) is not None


def _parse_loose_value(value: str) -> Any:
    if not value:
        return ""
    if value[0] in {"\"", "'"}:
        try:
            return ast.literal_eval(value)
        except (SyntaxError, ValueError):
            return value.strip("\"'")
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return value
