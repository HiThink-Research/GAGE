from __future__ import annotations

import pytest

from gage_eval.role.agent.backends.base import AgentBackend
from gage_eval.role.agent.loop import AgentLoop
from gage_eval.role.agent.tool_router import ToolRouter
from gage_eval.sandbox.base import ExecResult


class FakeBackend(AgentBackend):
    def __init__(self) -> None:
        self.calls = 0

    def invoke(self, payload):  # type: ignore[override]
        self.calls += 1
        return {
            "answer": "",
            "tool_calls": [
                {
                    "id": "call-1",
                    "name": "submit_patch_tool",
                    "arguments": {},
                }
            ],
        }


class FakeSandbox:
    def __init__(self, stdout: str) -> None:
        self.stdout = stdout
        self.exec_calls: list[str] = []

    def exec(self, command: str, timeout: int = 30):
        self.exec_calls.append(command)
        return ExecResult(exit_code=0, stdout=self.stdout, stderr="")


class FakeHandle:
    def __init__(self, sandbox: FakeSandbox) -> None:
        self.sandbox = sandbox
        self.runtime_handle = {}


class FakeProvider:
    def __init__(self, sandbox: FakeSandbox) -> None:
        self._handle = FakeHandle(sandbox)

    def get_handle(self) -> FakeHandle:
        return self._handle


@pytest.mark.fast
def test_agent_loop_stops_on_final_answer_from_tool() -> None:
    diff_text = "diff --git a/foo b/foo\n"
    backend = FakeBackend()
    router = ToolRouter()
    loop = AgentLoop(backend=backend, tool_router=router, max_turns=5)
    tools = [
        {
            "type": "function",
            "function": {
                "name": "submit_patch_tool",
                "description": "Run git diff",
                "parameters": {"type": "object", "properties": {}},
            },
            "x-gage": {"final_answer_from": "stdout"},
        }
    ]
    sandbox = FakeSandbox(stdout=diff_text)
    provider = FakeProvider(sandbox)
    result = loop.run(messages=[{"role": "user", "content": "fix"}], tools=tools, sandbox_provider=provider, sandbox_config={})

    assert backend.calls == 1
    assert result["answer"] == diff_text
