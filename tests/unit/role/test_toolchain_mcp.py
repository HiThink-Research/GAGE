from __future__ import annotations

import asyncio

import pytest

from gage_eval.mcp import McpClient
from gage_eval.role.adapters.base import RoleAdapterState
from gage_eval.role.adapters.toolchain import ToolchainAdapter


@pytest.mark.fast
def test_toolchain_merges_mcp_tools() -> None:
    mcp_tools = [
        {
            "name": "step",
            "description": "AppWorld step",
            "inputSchema": {
                "type": "object",
                "properties": {"action": {"type": "string"}},
                "required": ["action"],
            },
        }
    ]
    client = McpClient(
        mcp_client_id="appworld_env",
        transport="stub",
        endpoint="http://example.com",
        params={"tools": mcp_tools},
    )
    adapter = ToolchainAdapter(
        adapter_id="toolchain_main",
        tools=[
            {
                "type": "function",
                "function": {
                    "name": "local_tool",
                    "description": "Local helper",
                    "parameters": {"type": "object", "properties": {}},
                },
            }
        ],
        mcp_client=client,
    )

    result = asyncio.run(adapter.ainvoke({"sample": {}}, RoleAdapterState()))
    tools_schema = result["tools_schema"]
    names = {tool["function"]["name"] for tool in tools_schema}

    assert {"local_tool", "step"}.issubset(names)
    mcp_entry = next(tool for tool in tools_schema if tool["function"]["name"] == "step")
    assert mcp_entry["x-gage"]["mcp_client_id"] == "appworld_env"
