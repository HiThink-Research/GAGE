import pytest

from gage_eval.mcp import McpClient
from gage_eval.role.agent.loop import AgentLoop
from gage_eval.role.agent.tool_router import ToolRouter
from gage_eval.sandbox.base import ExecResult
from gage_eval.sandbox.manager import SandboxManager
from gage_eval.sandbox.provider import SandboxProvider, SandboxScope


class FakeBackend:
    def __init__(self):
        self.calls = 0

    def invoke(self, payload):
        self.calls += 1
        if self.calls == 1:
            return {
                "tool_calls": [
                    {
                        "id": "c1",
                        "function": {"name": "run_shell", "arguments": "{\"command\": \"ls\"}"},
                    }
                ]
            }
        return {"answer": "done"}


class FakeSandbox:
    def __init__(self, runtime_configs=None, resources=None):
        self.runtime_configs = runtime_configs or {}
        self.resources = resources or {}

    def start(self, config):
        return {"sandbox_id": "fake"}

    def exec(self, command, timeout=30):
        return ExecResult(exit_code=0, stdout="ok", stderr="", duration_ms=1.0)

    def teardown(self):
        return None

    def is_alive(self, timeout_s=None):
        return True


@pytest.mark.fast
def test_agent_loop_tool_call_flow():
    manager = SandboxManager()
    manager.register_runtime("fake", FakeSandbox)
    provider = SandboxProvider(
        manager,
        {"runtime": "fake"},
        SandboxScope(run_id="run", task_id="task", sample_id="sample"),
    )
    loop = AgentLoop(
        backend=FakeBackend(),
        tool_router=ToolRouter(),
        max_turns=3,
    )
    result = loop.run(
        messages=[{"role": "user", "content": "hi"}],
        tools=[{"type": "function", "function": {"name": "run_shell", "parameters": {}}}],
        sandbox_config={"runtime": "fake"},
        sandbox_provider=provider,
    )
    provider.release()
    assert result["answer"] == "done"
    assert len(result["agent_trace"]) == 2
    assert result["agent_trace"][0]["trace_role"] == "tool"
    assert result["agent_trace"][1]["trace_role"] == "assistant"


@pytest.mark.fast
def test_agent_loop_records_resolved_tool_for_meta_calls() -> None:
    class MetaBackend:
        def __init__(self) -> None:
            self.calls = 0

        def invoke(self, payload):
            self.calls += 1
            if self.calls == 1:
                return {
                    "tool_calls": [
                        {
                            "id": "c1",
                            "function": {
                                "name": "call_spotify",
                                "arguments": {"endpoint": "search", "params": {"q": "hi"}},
                            },
                        }
                    ]
                }
            return {"answer": "done"}

    executed = {}

    def executor(name, arguments):
        executed["name"] = name
        executed["arguments"] = arguments
        return {"status": "ok"}

    client = McpClient(
        mcp_client_id="mcp_main",
        transport="stub",
        endpoint="http://example.com",
        params={"executor": executor, "tools": []},
    )
    router = ToolRouter(mcp_clients={"mcp_main": client})
    tools = [
        {
            "type": "function",
            "function": {
                "name": "call_spotify",
                "description": "Call spotify endpoints.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "endpoint": {"type": "string"},
                        "params": {"type": "object"},
                    },
                    "required": ["endpoint"],
                },
            },
            "x-gage": {
                "meta_tool": True,
                "app_name": "spotify",
                "allowed_endpoints": ["search"],
                "mcp_client_id": "mcp_main",
            },
        }
    ]

    loop = AgentLoop(
        backend=MetaBackend(),
        tool_router=router,
        max_turns=3,
    )
    result = loop.run(messages=[{"role": "user", "content": "hi"}], tools=tools)
    assert executed["name"] == "spotify__search"
    assert executed["arguments"] == {"q": "hi"}
    tool_step = result["agent_trace"][0]
    assert tool_step["trace_role"] == "tool"
    assert tool_step["resolved_tool"] == "spotify__search"
