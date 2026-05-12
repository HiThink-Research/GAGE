from __future__ import annotations

from typing import Any

from gage_eval.agent_runtime.tooling.contracts import ToolingError


class McpToolProvider:
    """Executes a registered tool through an MCP client."""

    def __init__(self, client: Any) -> None:
        self._client = client

    def call(self, name: str, arguments: Any) -> Any:
        call_tool = getattr(self._client, "call_tool", None)
        if not callable(call_tool):
            raise ToolingError("client_execution.tool_router.not_found", "MCP client has no call_tool method")
        return call_tool(name, arguments)
