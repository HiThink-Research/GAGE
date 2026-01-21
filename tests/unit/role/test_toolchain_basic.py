import asyncio

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
