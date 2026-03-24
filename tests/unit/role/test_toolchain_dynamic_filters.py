from __future__ import annotations

import asyncio
from typing import Any, Dict, List

import pytest

from gage_eval.evaluation.support_artifacts import record_support_output
from gage_eval.role.adapters.base import RoleAdapterState
from gage_eval.role.toolchain import ToolchainAdapter


class FakeMcpClient:
    def __init__(self) -> None:
        self.mcp_client_id = "mcp-test"

    def list_tools(self) -> List[Dict[str, Any]]:
        return [
            {
                "name": "spotify__login",
                "description": "Login",
                "inputSchema": {"type": "object", "properties": {}},
            },
            {
                "name": "venmo__pay",
                "description": "Pay",
                "inputSchema": {"type": "object", "properties": {}},
            },
            {
                "name": "supervisor__complete_task",
                "description": "Complete",
                "inputSchema": {"type": "object", "properties": {}},
            },
        ]


@pytest.mark.fast
def test_toolchain_uses_support_filters() -> None:
    adapter = ToolchainAdapter(adapter_id="toolchain_main", mcp_client=FakeMcpClient())
    payload = {
        "sample": {
            "support_outputs": [
                {"tool_allowlist": ["spotify__login"], "tool_prefixes": ["supervisor"]}
            ]
        }
    }

    result = asyncio.run(adapter.ainvoke(payload, RoleAdapterState()))
    names = [tool["function"]["name"] for tool in result["tools_schema"]]

    assert names == ["spotify__login", "supervisor__complete_task"]


@pytest.mark.fast
def test_toolchain_uses_support_artifact_filters() -> None:
    adapter = ToolchainAdapter(adapter_id="toolchain_main", mcp_client=FakeMcpClient())
    sample: Dict[str, Any] = {}
    record_support_output(
        sample,
        slot_id="support:00:api_predictor",
        adapter_id="api_predictor",
        output={"tool_allowlist": ["spotify__login"], "tool_prefixes": ["supervisor"]},
    )
    sample.pop("support_outputs", None)

    result = asyncio.run(adapter.ainvoke({"sample": sample}, RoleAdapterState()))
    names = [tool["function"]["name"] for tool in result["tools_schema"]]

    assert names == ["spotify__login", "supervisor__complete_task"]
