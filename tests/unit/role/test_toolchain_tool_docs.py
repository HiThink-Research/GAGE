from __future__ import annotations

import asyncio

import pytest

from gage_eval.mcp import McpClient
from gage_eval.role.adapters.base import RoleAdapterState
from gage_eval.role.toolchain import ToolchainAdapter


@pytest.mark.fast
def test_toolchain_emits_tool_documentation() -> None:
    mcp_tools = [
        {
            "name": "spotify__search",
            "description": "Search Spotify",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                    "page_index": {"type": "integer", "minimum": 0},
                    "label": {"type": "string", "enum": ["inbox", "sent"]},
                },
                "required": ["query"],
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
        mcp_client=client,
        meta_tool_mode=True,
        tool_doc_format="text",
    )

    result = asyncio.run(adapter.ainvoke({"sample": {}}, RoleAdapterState()))
    tool_doc = result.get("tool_documentation") or ""
    meta = result.get("tool_documentation_meta") or {}

    assert "spotify.search" in tool_doc
    assert "note: page_index starts at 0" in tool_doc
    assert "label in [inbox, sent]" in tool_doc
    assert meta.get("apps") == 1
    assert meta.get("endpoints") == 1


@pytest.mark.fast
def test_toolchain_emits_tool_documentation_native_mode() -> None:
    mcp_tools = [
        {
            "name": "spotify__search",
            "description": "Search Spotify",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                    "page_index": {"type": "integer", "minimum": 0},
                    "label": {"type": "string", "enum": ["inbox", "sent"]},
                },
                "required": ["query"],
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
        mcp_client=client,
        meta_tool_mode=False,
        tool_doc_enabled=True,
        tool_doc_format="app_kv",
    )

    result = asyncio.run(adapter.ainvoke({"sample": {}}, RoleAdapterState()))
    tool_doc = result.get("tool_documentation") or ""
    meta = result.get("tool_documentation_meta") or {}

    assert "spotify:" in tool_doc
    assert "search: Search Spotify" in tool_doc
    assert "note: page_index starts at 0" in tool_doc
    assert "label in [inbox, sent]" in tool_doc
    assert meta.get("apps") == 1
    assert meta.get("endpoints") == 1
