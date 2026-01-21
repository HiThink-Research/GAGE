from __future__ import annotations

import asyncio
import json

import pytest

from gage_eval.mcp import McpClient
from gage_eval.role.adapters.base import RoleAdapterState
from gage_eval.role.adapters.toolchain import ToolchainAdapter
from gage_eval.role.agent.loop import AgentLoop
from gage_eval.role.agent.tool_router import ToolRouter


class MetaToolBackend:
    def __init__(self) -> None:
        self.calls = 0

    def invoke(self, payload):
        self.calls += 1
        if self.calls == 1:
            return {
                "tool_calls": [
                    {
                        "id": "call-1",
                        "function": {
                            "name": "call_spotify",
                            "arguments": json.dumps(
                                {"endpoint": "search", "params": {"query": "lofi"}}
                            ),
                        },
                    }
                ]
            }
        return {"answer": "done"}


@pytest.mark.fast
def test_meta_tool_end_to_end() -> None:
    executed = {}

    def executor(name, arguments):
        executed["name"] = name
        executed["arguments"] = arguments
        return {"status": "ok"}

    mcp_tools = [
        {
            "name": "spotify__search",
            "description": "Search Spotify",
            "inputSchema": {"type": "object", "properties": {"query": {"type": "string"}}},
        }
    ]
    client = McpClient(
        mcp_client_id="appworld_env",
        transport="stub",
        endpoint="http://example.com",
        params={"executor": executor, "tools": mcp_tools},
    )
    adapter = ToolchainAdapter(
        adapter_id="toolchain_main",
        mcp_client=client,
        meta_tool_mode=True,
    )
    support_output = asyncio.run(adapter.ainvoke({"sample": {}}, RoleAdapterState()))
    tools_schema = support_output["tools_schema"]

    router = ToolRouter(mcp_clients={"appworld_env": client})
    loop = AgentLoop(
        backend=MetaToolBackend(),
        tool_router=router,
        max_turns=3,
    )
    result = loop.run(messages=[{"role": "user", "content": "go"}], tools=tools_schema)

    assert executed["name"] == "spotify__search"
    assert executed["arguments"] == {"query": "lofi"}
    assert result["answer"] == "done"
