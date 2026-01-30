from __future__ import annotations

from gage_eval.role.adapters.base import RoleAdapterState
from gage_eval.role.toolchain.toolchain import ToolchainAdapter


def test_tau2_toolchain_merges_respond_and_env_tools() -> None:
    respond_tool = {
        "type": "function",
        "function": {
            "name": "respond",
            "description": "Send message to user",
            "parameters": {
                "type": "object",
                "properties": {"message": {"type": "string"}},
                "required": ["message"],
            },
        },
        "x-gage": {"final_answer_from": "final_answer"},
    }
    env_tool = {
        "type": "function",
        "function": {
            "name": "lookup",
            "description": "Lookup tool",
            "parameters": {"type": "object", "properties": {}},
        },
    }
    adapter = ToolchainAdapter(adapter_id="tau2_toolchain", tools=[respond_tool])
    state = RoleAdapterState()

    payload = {"sample": {"tools": [env_tool]}}
    result = adapter.invoke(payload, state)

    tool_names = {tool["function"]["name"] for tool in result["tools_schema"]}
    assert tool_names == {"respond", "lookup"}
