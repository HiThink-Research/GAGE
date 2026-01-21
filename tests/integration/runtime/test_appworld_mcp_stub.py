from __future__ import annotations

import pytest

from gage_eval.mcp import McpClient
from tests.integration.runtime.mcp_stub import AppWorldMcpStub


@pytest.mark.fast
def test_appworld_mcp_stub_roundtrip() -> None:
    stub = AppWorldMcpStub()
    client = McpClient(
        mcp_client_id="appworld_env",
        endpoint="http://stub",
        params={"requester": stub.requester},
    )

    tools = client.list_tools()
    names = {tool.get("name") for tool in tools}
    assert {"get_state", "step"}.issubset(names)

    result = client.call_tool("step", {"action": "noop"})
    assert result["observation"]["status"] == "ok"
    assert stub.tool_calls[0]["name"] == "step"
