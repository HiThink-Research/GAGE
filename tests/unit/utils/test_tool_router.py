import pytest

from gage_eval.role.agent.tool_router import ToolRouter
from gage_eval.sandbox.base import ExecResult


class FakeSandbox:
    def __init__(self):
        self.last_command = None

    def exec(self, command, timeout=30):
        self.last_command = command
        return ExecResult(exit_code=0, stdout="ok", stderr="", duration_ms=0.5)


class FailingSandbox:
    def __init__(self):
        self.last_command = None

    def exec(self, command, timeout=30):
        self.last_command = command
        return ExecResult(exit_code=127, stdout="", stderr="/bin/sh: badcmd: not found\n", duration_ms=0.5)


@pytest.mark.fast
def test_tool_router_execs_command_payload():
    router = ToolRouter()
    sandbox = FakeSandbox()
    tool_call = {"function": {"name": "run_shell", "arguments": "{\"command\": \"ls\"}"}}
    result = router.execute(tool_call, sandbox)
    assert result["name"] == "run_shell"
    assert sandbox.last_command == "ls"
    assert result["output"]["exit_code"] == 0
    assert result["status"] == "success"


@pytest.mark.fast
def test_tool_router_marks_nonzero_exit_code_as_failed():
    router = ToolRouter()
    sandbox = FailingSandbox()
    tool_call = {"function": {"name": "run_shell", "arguments": "{\"command\": \"badcmd\"}"}}
    result = router.execute(tool_call, sandbox)
    assert result["name"] == "run_shell"
    assert sandbox.last_command == "badcmd"
    assert result["output"]["exit_code"] == 127
    assert result["status"] == "failed"
