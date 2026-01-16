from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List


_DEFAULT_TOOLS = [
    {
        "name": "get_state",
        "description": "Return environment state",
        "inputSchema": {"type": "object", "properties": {}},
    },
    {
        "name": "step",
        "description": "Apply an action and return observation",
        "inputSchema": {
            "type": "object",
            "properties": {"action": {"type": "string"}},
            "required": ["action"],
        },
    },
]


@dataclass
class AppWorldMcpStub:
    """In-memory MCP stub for AppWorld-style tool calls."""

    tools: List[Dict[str, Any]] = field(default_factory=lambda: list(_DEFAULT_TOOLS))
    tool_calls: List[Dict[str, Any]] = field(default_factory=list)
    observations: List[Dict[str, Any]] = field(default_factory=list)

    def list_tools(self) -> List[Dict[str, Any]]:
        """Return MCP tool schemas."""

        return [dict(tool) for tool in self.tools]

    def call_tool(self, name: str, arguments: Any) -> Dict[str, Any]:
        """Handle tool execution for the stubbed MCP server."""

        self.tool_calls.append({"name": name, "arguments": arguments})
        if name == "get_state":
            return {"state": "ready"}
        if name == "step":
            observation = {"observation": {"status": "ok", "result": arguments}}
            self.observations.append(observation)
            return observation
        return {"error": f"unknown_tool:{name}"}

    def requester(self, method: str, payload: Dict[str, Any]) -> Any:
        """Mimic MCP request routing for the client requester hook."""

        if method == "list_tools":
            return self.list_tools()
        if method == "call_tool":
            return self.call_tool(payload.get("name", ""), payload.get("arguments"))
        if method == "list_resources":
            return {"resources": []}
        if method == "read_resource":
            return {"content": ""}
        if method == "sample":
            return {"samples": []}
        raise ValueError(f"unsupported_method:{method}")
