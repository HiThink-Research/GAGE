from __future__ import annotations

from typing import Any

from gage_eval.agent_runtime.tooling.contracts import ToolSchemaIR, ToolingError
from gage_eval.agent_runtime.tooling.mcp.schema_adapter import mcp_tool_to_schema_ir


def discover_mcp_tools(client: Any, *, server_id: str) -> list[ToolSchemaIR]:
    """Discover MCP tool schemas and normalize them into ToolSchemaIR."""

    try:
        raw_tools = client.list_tools()
    except Exception as exc:
        raise ToolingError(
            "client_execution.tool_registry.mcp_discovery_failed",
            f"MCP discovery failed for server {server_id}",
            details={"server_id": server_id, "error": str(exc)},
        ) from exc

    schemas: list[ToolSchemaIR] = []
    for raw_tool in raw_tools or []:
        schemas.append(mcp_tool_to_schema_ir(raw_tool, server_id=server_id))
    return schemas
