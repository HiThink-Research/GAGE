from __future__ import annotations

from typing import Any

from gage_eval.agent_runtime.tooling.contracts import ToolSchemaIR


def mcp_tool_to_schema_ir(raw_tool: Any, *, server_id: str) -> ToolSchemaIR:
    """Normalize one MCP tool schema into ToolSchemaIR."""

    return ToolSchemaIR.from_provider_schema(dict(raw_tool), provider=f"mcp:{server_id}")
