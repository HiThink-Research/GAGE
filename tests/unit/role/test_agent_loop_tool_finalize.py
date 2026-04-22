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


class GemmaBackfillBackend(AgentBackend):
    def __init__(self) -> None:
        self.calls = 0
        self.payloads: list[dict] = []
        self._tool_result_format = "gemma4"

    def invoke(self, payload):  # type: ignore[override]
        self.calls += 1
        self.payloads.append(payload)
        if self.calls == 1:
            return {
                "answer": "",
                "tool_calls": [
                    {
                        "id": "call-1",
                        "type": "function",
                        "function": {
                            "name": "run_shell",
                            "arguments": {"command": "echo ok"},
                        },
                    }
                ],
            }
        return {"answer": "done"}


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


@pytest.mark.fast
def test_agent_loop_backfills_gemma_tool_responses_on_assistant_message() -> None:
    backend = GemmaBackfillBackend()
    router = ToolRouter()
    loop = AgentLoop(backend=backend, tool_router=router, max_turns=3)
    tools = [
        {
            "type": "function",
            "function": {
                "name": "run_shell",
                "description": "Run a shell command",
                "parameters": {"type": "object", "properties": {"command": {"type": "string"}}},
            },
        }
    ]
    sandbox = FakeSandbox(stdout="ok\n")
    provider = FakeProvider(sandbox)
    messages = [{"role": "user", "content": "run it"}]

    result = loop.run(messages=messages, tools=tools, sandbox_provider=provider, sandbox_config={})

    assert result["answer"] == "done"
    assert backend.calls == 2
    assistant_message = messages[1]
    assert assistant_message["role"] == "assistant"
    assert assistant_message["tool_calls"][0]["function"]["name"] == "run_shell"
    assert assistant_message["tool_responses"][0]["name"] == "run_shell"
    assert assistant_message["tool_responses"][0]["response"]["stdout"] == "ok\n"
    assert all(message["role"] != "tool" for message in messages)
