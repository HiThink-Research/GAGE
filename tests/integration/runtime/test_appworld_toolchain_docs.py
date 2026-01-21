from __future__ import annotations

import asyncio

import pytest

from gage_eval.mcp import McpClient
from gage_eval.role.adapters.base import RoleAdapterState
from gage_eval.role.toolchain import ToolchainAdapter


@pytest.mark.fast
def test_toolchain_schema_yaml_documentation() -> None:
    tools = [
        {
            "name": "notes__create",
            "description": "Create a note",
            "inputSchema": {
                "type": "object",
                "properties": {"text": {"type": "string"}},
                "required": ["text"],
            },
            "outputSchema": {
                "type": "object",
                "properties": {"note_id": {"type": "string"}},
            },
        }
    ]
    client = McpClient(
        mcp_client_id="appworld_env",
        transport="stub",
        endpoint="http://example.com",
        params={"tools": tools},
    )
    adapter = ToolchainAdapter(
        adapter_id="toolchain_main",
        mcp_client=client,
        meta_tool_mode=False,
        tool_doc_enabled=True,
        tool_doc_format="schema_yaml",
    )

    result = asyncio.run(adapter.ainvoke({"sample": {}}, RoleAdapterState()))

    assert "response_schema:" in result.get("tool_documentation", "")
    assert result.get("tool_documentation_meta", {}).get("doc_format") == "schema_yaml"
