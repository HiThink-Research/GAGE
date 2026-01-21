import asyncio

import pytest

from gage_eval.role.adapters.base import RoleAdapterState
from gage_eval.role.agent.tool_router import ToolRouter
from gage_eval.role.toolchain import ToolchainAdapter
from gage_eval.mcp import McpClient


@pytest.mark.fast
def test_mcp_tools_mapping_and_allowlist():
    executed = {}

    def executor(name, arguments):
        executed["name"] = name
        executed["arguments"] = arguments
        return {"status": "ok"}

    client = McpClient(
        mcp_client_id="mcp_main",
        transport="stub",
        endpoint="http://example.com",
        allowlist=["allowed_tool"],
        params={
            "executor": executor,
            "tools": [
                {
                    "name": "allowed_tool",
                    "description": "Allowed",
                    "inputSchema": {"type": "object", "properties": {"q": {"type": "string"}}},
                },
                {
                    "name": "blocked_tool",
                    "description": "Blocked",
                    "inputSchema": {"type": "object", "properties": {"x": {"type": "string"}}},
                },
            ]
        },
    )
    adapter = ToolchainAdapter(adapter_id="toolchain_main", mcp_client=client)
    result = asyncio.run(adapter.ainvoke({"sample": {}}, RoleAdapterState()))
    tools = result["tools_schema"]
    names = [tool["function"]["name"] for tool in tools]
    assert names == ["allowed_tool"]
    assert tools[0]["function"]["parameters"]["properties"]["q"]["type"] == "string"
    assert tools[0]["x-gage"]["mcp_client_id"] == "mcp_main"

    router = ToolRouter(mcp_clients={"mcp_main": client})
    tool_call = {"function": {"name": "allowed_tool", "arguments": {"q": "ping"}}}
    tool_registry = {"allowed_tool": {"x-gage": {"mcp_client_id": "mcp_main"}}}
    result = router.execute(tool_call, None, tool_registry=tool_registry)
    assert result["status"] == "success"
    assert executed["name"] == "allowed_tool"
