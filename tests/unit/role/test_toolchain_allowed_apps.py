from __future__ import annotations

import asyncio

import pytest

from gage_eval.mcp import McpClient
from gage_eval.role.adapters.base import RoleAdapterState
from gage_eval.role.toolchain import ToolchainAdapter


@pytest.mark.fast
def test_toolchain_respects_allowed_apps() -> None:
    mcp_tools = [
        {
            "name": "spotify__search",
            "description": "Search Spotify",
            "inputSchema": {"type": "object", "properties": {"query": {"type": "string"}}},
        },
        {
            "name": "gmail__search",
            "description": "Search Gmail",
            "inputSchema": {"type": "object", "properties": {"query": {"type": "string"}}},
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

    sample = {"metadata": {"appworld": {"allowed_apps": ["spotify"]}}}
    result = asyncio.run(adapter.ainvoke({"sample": sample}, RoleAdapterState()))
    tools_schema = result["tools_schema"]
    names = {tool["function"]["name"] for tool in tools_schema}

    assert "call_spotify" in names
    assert "call_gmail" not in names
    assert "gmail.search" not in (result.get("tool_documentation") or "")
