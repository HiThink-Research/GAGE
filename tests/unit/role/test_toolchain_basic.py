import asyncio
from typing import Any

import pytest

from gage_eval.role.toolchain import ToolchainAdapter
from gage_eval.role.adapters.base import RoleAdapterState


@pytest.mark.fast
def test_toolchain_passthrough_schema():
    adapter = ToolchainAdapter(
        adapter_id="toolchain_main",
        tools=[{"name": "default_tool", "parameters": {"type": "object", "properties": {}}}],
    )
    payload = {
        "sample": {
            "tools": [{"name": "sample_tool", "parameters": {"type": "object", "properties": {}}}],
        }
    }
    result = asyncio.run(adapter.ainvoke(payload, RoleAdapterState()))
    names = [tool["function"]["name"] for tool in result["tools_schema"]]
    assert "default_tool" in names
    assert "sample_tool" in names


@pytest.mark.fast
def test_toolchain_preserves_x_gage_metadata():
    adapter = ToolchainAdapter(
        adapter_id="toolchain_main",
        tools=[
            {
                "name": "submit_patch_tool",
                "parameters": {"type": "object", "properties": {}},
                "x-gage": {"final_answer_from": "stdout"},
            }
        ],
    )
    payload = {"sample": {}}
    result = asyncio.run(adapter.ainvoke(payload, RoleAdapterState()))

    tool = next(entry for entry in result["tools_schema"] if entry["function"]["name"] == "submit_patch_tool")
    assert tool.get("x-gage", {}).get("final_answer_from") == "stdout"


class _FakeMcpClient:
    def __init__(self, tools: list[dict[str, Any]]) -> None:
        self.mcp_client_id = "fake-mcp"
        self._tools = tools

    def list_tools(self) -> list[dict[str, Any]]:
        return list(self._tools)


@pytest.mark.fast
def test_toolchain_merges_duplicate_tools_in_linear_order() -> None:
    adapter = ToolchainAdapter(
        adapter_id="toolchain_main",
        tools=[
            {"name": "alpha", "parameters": {"type": "object", "properties": {}}},
            {
                "name": "dup",
                "description": "default",
                "parameters": {"type": "object", "properties": {}},
            },
        ],
        mcp_client=_FakeMcpClient(
            [
                {
                    "name": "gamma",
                    "description": "gamma",
                    "inputSchema": {"type": "object", "properties": {}},
                },
                {
                    "name": "dup",
                    "description": "mcp",
                    "inputSchema": {"type": "object", "properties": {}},
                },
            ]
        ),
    )
    payload = {
        "sample": {
            "tools": [
                {"name": "beta", "parameters": {"type": "object", "properties": {}}},
                {
                    "name": "dup",
                    "description": "sample",
                    "parameters": {"type": "object", "properties": {}},
                },
            ]
        }
    }

    result = asyncio.run(adapter.ainvoke(payload, RoleAdapterState()))

    names = [tool["function"]["name"] for tool in result["tools_schema"]]
    dup = next(tool for tool in result["tools_schema"] if tool["function"]["name"] == "dup")

    assert names == ["alpha", "beta", "gamma", "dup"]
    assert dup["function"]["description"] == "mcp"
