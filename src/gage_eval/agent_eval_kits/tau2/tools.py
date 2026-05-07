from __future__ import annotations

from gage_eval.agent_eval_kits.common import extract_instruction, normalize_messages, normalize_tools
from gage_eval.agent_runtime.tooling.contracts import ToolSchemaIR
from gage_eval.agent_runtime.tooling.registry import RuntimeToolRegistry


def build_tau2_messages(sample: dict[str, object]) -> list[dict[str, object]]:
    """Build framework-loop Tau2 messages."""

    return normalize_messages(sample, fallback_text=extract_instruction(sample))


def build_tau2_tools(sample: dict[str, object], initialize_result: dict[str, object]) -> list[dict[str, object]]:
    """Build Tau2 tool schemas."""

    return normalize_tools(sample, list(initialize_result.get("tools_schema") or []))


def build_tool_registry() -> RuntimeToolRegistry:
    registry = RuntimeToolRegistry()
    registry.register_environment_tool(
        ToolSchemaIR(
            name="respond",
            description="Return a response to the Tau2 user simulator.",
            input_schema={
                "type": "object",
                "properties": {"message": {"type": "string"}},
                "required": ["message"],
            },
            raw_schema={
                "type": "function",
                "function": {
                    "name": "respond",
                    "description": "Return a response to the Tau2 user simulator.",
                    "parameters": {
                        "type": "object",
                        "properties": {"message": {"type": "string"}},
                        "required": ["message"],
                    },
                },
            },
            provider_format="tau2",
        )
    )
    return registry
