from __future__ import annotations

import pytest

from gage_eval.role.agent.tool_router import ToolRouter
from gage_eval.sandbox.base import ExecResult


class FakeSandbox:
    def __init__(self) -> None:
        self.calls: list[tuple[str, int]] = []

    def exec(self, command: str, timeout: int = 30) -> ExecResult:
        self.calls.append((command, timeout))
        return ExecResult(exit_code=0, stdout="ok", stderr="", duration_ms=1.0)


class ExecToolSandbox:
    def __init__(self) -> None:
        self.exec_tool_calls: list[tuple[str, object]] = []

    def exec_tool(self, name: str, arguments: object) -> dict[str, str]:
        self.exec_tool_calls.append((name, arguments))
        return {"stdout": "ok", "stderr": "", "exit_code": 0}


@pytest.mark.fast
def test_tool_router_maps_run_shell_cmd_alias() -> None:
    sandbox = FakeSandbox()
    router = ToolRouter()
    tool_call = {"function": {"name": "run_shell", "arguments": {"cmd": "echo ok", "timeout_s": "12"}}}

    result = router.execute(tool_call, sandbox)

    assert sandbox.calls == [("echo ok", 12)]
    assert result["status"] == "success"
    assert result["output"]["stdout"] == "ok"


@pytest.mark.fast
def test_tool_router_rejects_non_string_run_shell_command_without_exec() -> None:
    sandbox = FakeSandbox()
    router = ToolRouter()
    tool_call = {"function": {"name": "run_shell", "arguments": {"command": ["echo", "ok"]}}}

    result = router.execute(tool_call, sandbox)

    assert sandbox.calls == []
    assert result["status"] == "failed"
    assert result["output"]["error"] == "tool_argument_invalid"
    assert result["output"]["error_code"] == "tool_argument_invalid"
    assert result["output"]["argument"] == "command"
    assert result["output"]["expected_type"] == "string"


@pytest.mark.fast
def test_tool_router_rejects_non_string_run_shell_command_before_exec_tool() -> None:
    sandbox = ExecToolSandbox()
    router = ToolRouter()
    tool_call = {"function": {"name": "run_shell", "arguments": {"command": ["echo", "ok"]}}}

    result = router.execute(tool_call, sandbox)

    assert sandbox.exec_tool_calls == []
    assert result["status"] == "failed"
    assert result["output"]["error_code"] == "tool_argument_invalid"


@pytest.mark.fast
def test_tool_router_rejects_mixed_alias_when_any_run_shell_alias_is_non_string() -> None:
    sandbox = FakeSandbox()
    router = ToolRouter()
    tool_call = {
        "function": {
            "name": "run_shell",
            "arguments": {"command": "echo ok", "cmd": ["echo", "bad"]},
        }
    }

    result = router.execute(tool_call, sandbox)

    assert sandbox.calls == []
    assert result["status"] == "failed"
    assert result["output"]["error_code"] == "tool_argument_invalid"
    assert result["output"]["argument"] == "cmd"
