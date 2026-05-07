from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Callable, Literal

from gage_eval.agent_runtime.resources.contracts import ResourceLease
from gage_eval.agent_runtime.serialization import to_json_compatible


TOOLING_FAILURE_CODES = {
    "client_execution.tool_schema_invalid",
    "client_execution.tool_protocol_parse_error",
    "client_execution.tool_protocol_missing_call",
    "client_execution.tool_protocol_missing_call_id",
    "client_execution.tool_router.not_found",
    "client_execution.tool_argument_invalid",
    "client_execution.tool_result_injection_failed",
    "client_execution.tool_retry_budget_exhausted",
    "client_execution.tool_registry.mcp_discovery_failed",
    "client_execution.tool_registry.skill_policy_denied",
    "client_execution.tool_registry.skill_unavailable",
    "client_execution.tool_router.environment_unavailable",
    "client_execution.tool_router.human_gateway_unavailable",
}


class ToolingError(RuntimeError):
    """Stable tooling error with a runtime failure code."""

    def __init__(self, code: str, message: str, *, details: dict[str, Any] | None = None) -> None:
        self.code = code
        self.details = dict(details or {})
        super().__init__(message)


@dataclass(frozen=True)
class ToolSchemaIR:
    """Provider-neutral tool schema plus its original provider payload."""

    name: str
    description: str
    input_schema: dict[str, Any]
    raw_schema: dict[str, Any]
    provider_format: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name.strip():
            raise ToolingError("client_execution.tool_schema_invalid", "tool schema requires name")
        if not isinstance(self.input_schema, dict):
            raise ToolingError("client_execution.tool_schema_invalid", "tool schema input_schema must be a dict")
        if not isinstance(self.raw_schema, dict):
            raise ToolingError("client_execution.tool_schema_invalid", "tool schema raw_schema must be a dict")

    @classmethod
    def from_provider_schema(cls, raw_schema: dict[str, Any], *, provider: str | None = None) -> "ToolSchemaIR":
        """Normalize an OpenAI/MCP/plain schema while keeping the raw payload."""

        if not isinstance(raw_schema, dict):
            raise ToolingError("client_execution.tool_schema_invalid", "provider schema must be a dict")

        function = raw_schema.get("function") if isinstance(raw_schema.get("function"), dict) else None
        if function is not None:
            raw_name = function.get("name")
            description = function.get("description") or raw_schema.get("description") or ""
            input_schema = function.get("parameters") or function.get("input_schema") or {}
        else:
            raw_name = raw_schema.get("name") or raw_schema.get("tool")
            description = raw_schema.get("description") or ""
            input_schema = raw_schema.get("input_schema") or raw_schema.get("inputSchema") or raw_schema.get("parameters") or {}

        if not raw_name:
            raise ToolingError("client_execution.tool_schema_invalid", "tool schema requires name")
        if not isinstance(input_schema, dict):
            raise ToolingError("client_execution.tool_schema_invalid", "tool schema input schema must be a dict")

        name, metadata = normalize_provider_tool_name(str(raw_name))
        if provider:
            metadata["provider"] = provider
        return cls(
            name=name,
            description=str(description or ""),
            input_schema=dict(input_schema),
            raw_schema=dict(raw_schema),
            provider_format=provider,
            metadata=metadata,
        )

    def to_provider_schema(self) -> dict[str, Any]:
        """Project the IR into the OpenAI-compatible function schema used by the legacy loop."""

        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": to_json_compatible(self.input_schema),
            },
        }


@dataclass(frozen=True)
class ToolCallIR:
    """Provider-neutral tool call intent with raw provider evidence."""

    call_id: str
    name: str
    arguments_json: str
    raw_message: Any
    provider: str = "openai"
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.call_id:
            raise ToolingError("client_execution.tool_protocol_missing_call_id", "tool call id is required")
        if not self.name:
            raise ToolingError("client_execution.tool_protocol_missing_call", "tool call name is required")
        if not isinstance(self.arguments_json, str):
            raise ToolingError("client_execution.tool_argument_invalid", "tool call arguments_json must be a string")

    @classmethod
    def from_provider_call(
        cls,
        raw_call: dict[str, Any],
        *,
        turn_index: int,
        call_index: int,
        provider: str = "openai",
        require_call_id: bool = False,
    ) -> "ToolCallIR":
        if not isinstance(raw_call, dict):
            raise ToolingError("client_execution.tool_protocol_parse_error", "tool call must be a dict")

        raw_id = raw_call.get("id") or raw_call.get("call_id") or raw_call.get("tool_call_id")
        if require_call_id and not raw_id:
            raise ToolingError("client_execution.tool_protocol_missing_call_id", "provider omitted tool call id")
        call_id = str(raw_id or f"call_{turn_index}_{call_index}")

        function = raw_call.get("function") if isinstance(raw_call.get("function"), dict) else None
        function_call = raw_call.get("function_call") if isinstance(raw_call.get("function_call"), dict) else None
        source = function or function_call or raw_call
        raw_name = source.get("name") or source.get("tool")
        if not raw_name:
            raise ToolingError("client_execution.tool_protocol_missing_call", "tool call name is missing")

        arguments = (
            source.get("arguments")
            if "arguments" in source
            else source.get("args")
            if "args" in source
            else source.get("parameters")
            if "parameters" in source
            else {}
        )
        arguments = unwrap_provider_arguments(arguments)
        arguments_json = compact_json(arguments)
        name, metadata = normalize_provider_tool_name(str(raw_name))
        return cls(
            call_id=call_id,
            name=name,
            arguments_json=arguments_json,
            raw_message=raw_call,
            provider=provider,
            metadata=metadata,
        )

    def arguments(self) -> Any:
        try:
            return json.loads(self.arguments_json or "{}")
        except json.JSONDecodeError as exc:
            raise ToolingError(
                "client_execution.tool_argument_invalid",
                "tool call arguments_json is not valid JSON",
                details={"call_id": self.call_id, "name": self.name},
            ) from exc


ToolResultStatus = Literal["success", "error"]


@dataclass(frozen=True)
class ToolResultIR:
    """Provider-neutral tool result with raw output and artifact references."""

    call_id: str
    name: str
    provider: str
    status: ToolResultStatus
    output_text: str = ""
    output_json: Any = field(default_factory=dict)
    raw_output: Any | None = None
    artifact_refs: list[Any] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def output(self) -> Any:
        """Compatibility alias for older callers while Task06b removes legacy usage."""

        return self.output_json

    @staticmethod
    def serialize_for_injection(
        result: "ToolResultIR",
        *,
        serializer: Callable[["ToolResultIR"], Any],
    ) -> Any:
        try:
            return serializer(result)
        except Exception as exc:
            raise ToolingError(
                "client_execution.tool_result_injection_failed",
                "failed to serialize tool result for model injection",
                details={"call_id": result.call_id, "name": result.name},
            ) from exc


@dataclass(frozen=True)
class ToolExecutionContext:
    """Per-dispatch runtime context supplied by the scheduler."""

    run_id: str
    task_id: str
    sample_id: str
    trial_id: str
    resource_lease: ResourceLease | None = None
    environment_lease: Any | None = None
    artifact_sink: Any | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


def normalize_provider_tool_name(raw_name: str) -> tuple[str, dict[str, Any]]:
    """Strip a provider suffix while preserving raw_name evidence."""

    name = raw_name.split("<|channel|", 1)[0].strip()
    metadata: dict[str, Any] = {}
    if "__" in name:
        base, suffix = name.rsplit("__", 1)
        if base and suffix in _PROVIDER_SUFFIXES:
            metadata["raw_name"] = name
            metadata["provider_suffix"] = suffix
            name = base
    return name, metadata


_PROVIDER_SUFFIXES = {
    "anthropic",
    "deepseek",
    "functiongemma",
    "gemini",
    "gemma",
    "glm",
    "minimax",
    "openai",
    "qwen",
}


def unwrap_provider_arguments(arguments: Any) -> Any:
    """Normalize provider arg-key wrappers without parsing already raw JSON text."""

    if isinstance(arguments, str):
        return arguments
    if isinstance(arguments, dict):
        current = dict(arguments)
        for _ in range(3):
            if set(current) == {"arg_key"} and isinstance(current.get("arg_key"), dict):
                current = dict(current["arg_key"])
                continue
            for key in ("arguments", "input", "payload"):
                nested = current.get(key)
                if isinstance(nested, dict) and len(current) == 1:
                    current = dict(nested)
                    break
            else:
                break
        return current
    return arguments


def compact_json(value: Any) -> str:
    """Return provider raw JSON unchanged or compact stable JSON for structured values."""

    if isinstance(value, str):
        return value
    return json.dumps(to_json_compatible(value), ensure_ascii=True, separators=(",", ":"), sort_keys=True)
