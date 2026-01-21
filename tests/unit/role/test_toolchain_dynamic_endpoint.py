from __future__ import annotations

import asyncio
from typing import Any, Dict, List

import pytest

from gage_eval.role.adapters.base import RoleAdapterState
from gage_eval.role.adapters.toolchain import ToolchainAdapter
from gage_eval.sandbox.manager import SandboxManager
from gage_eval.sandbox.provider import SandboxProvider, SandboxScope


class FakeSandbox:
    def __init__(self, runtime_configs: Dict[str, Any] | None = None, resources: Dict[str, Any] | None = None):
        self.runtime_configs = runtime_configs or {}
        self.resources = resources or {}

    def start(self, config: Dict[str, Any]) -> Dict[str, Any]:
        return {"mcp_endpoint": "http://dynamic-mcp"}

    def teardown(self) -> None:
        return None


class FakeMcpClient:
    def __init__(self, endpoint: str) -> None:
        self.mcp_client_id = "mcp-test"
        self.endpoint = endpoint
        self.list_calls: List[str] = []

    def list_tools(self) -> List[Dict[str, Any]]:
        self.list_calls.append(self.endpoint)
        return [{"name": "tool_a", "description": "A tool", "inputSchema": {"type": "object"}}]


@pytest.mark.fast
def test_toolchain_updates_mcp_endpoint_from_provider() -> None:
    manager = SandboxManager()
    manager.register_runtime("fake", FakeSandbox)
    provider = SandboxProvider(
        manager,
        {"runtime": "fake"},
        SandboxScope(run_id="run", task_id="task", sample_id="sample"),
    )
    mcp_client = FakeMcpClient(endpoint="http://static-mcp")
    adapter = ToolchainAdapter(adapter_id="toolchain_main", mcp_client=mcp_client)
    payload = {"sample": {}, "sandbox_provider": provider}
    result = asyncio.run(adapter.ainvoke(payload, RoleAdapterState()))

    assert mcp_client.endpoint == "http://dynamic-mcp"
    assert mcp_client.list_calls == ["http://dynamic-mcp"]
    assert result["tools_schema"][0]["function"]["name"] == "tool_a"
    provider.release()
