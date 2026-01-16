from __future__ import annotations

import asyncio

import pytest

from gage_eval.mcp import McpClient
from gage_eval.role.adapters.base import RoleAdapterState
from gage_eval.role.adapters.toolchain import ToolchainAdapter


@pytest.mark.fast
def test_toolchain_builds_meta_tools() -> None:
    mcp_tools = [
        {
            "name": "spotify__search",
            "description": "Search Spotify",
            "inputSchema": {
                "type": "object",
                "properties": {"query": {"type": "string"}},
            },
        },
        {
            "name": "gmail__search",
            "description": "Search Gmail",
            "inputSchema": {
                "type": "object",
                "properties": {"query": {"type": "string"}},
            },
        },
        {
            "name": "supervisor__show_active_task",
            "description": "Task",
            "inputSchema": {"type": "object", "properties": {}},
        },
    ]
    client = McpClient(
        mcp_client_id="appworld_env",
        transport="stub",
        endpoint="http://example.com",
        params={"tools": mcp_tools},
    )
    adapter = ToolchainAdapter(
        adapter_id="toolchain_main",
        mcp_client=client,
        meta_tool_mode=True,
    )

    result = asyncio.run(adapter.ainvoke({"sample": {}}, RoleAdapterState()))
    tools_schema = result["tools_schema"]
    names = [tool["function"]["name"] for tool in tools_schema]

    assert {"call_spotify", "call_gmail", "call_supervisor"}.issubset(names)
    assert "spotify__search" not in names
    call_spotify = next(tool for tool in tools_schema if tool["function"]["name"] == "call_spotify")
    assert call_spotify["x-gage"]["meta_tool"] is True
    assert call_spotify["x-gage"]["app_name"] == "spotify"
    assert "search" in call_spotify["x-gage"]["allowed_endpoints"]
