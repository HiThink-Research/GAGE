import json

import pytest

from gage_eval.role.agent.tool_router import ToolRouter
from gage_eval.sandbox.base import ExecResult


class FakeSandbox:
    def __init__(self):
        self.last_command = None

    def exec(self, command, timeout=30):
        self.last_command = command
        return ExecResult(exit_code=0, stdout="ok", stderr="", duration_ms=0.5)


@pytest.mark.fast
def test_tool_router_execs_command_payload():
    router = ToolRouter()
    sandbox = FakeSandbox()
    tool_call = {"function": {"name": "run_shell", "arguments": "{\"command\": \"ls\"}"}}
    result = router.execute(tool_call, sandbox)
    assert result["name"] == "run_shell"
    payload = json.loads(sandbox.last_command)
    assert payload["tool"] == "run_shell"
    assert payload["arguments"]["command"] == "ls"
    assert result["output"]["exit_code"] == 0
