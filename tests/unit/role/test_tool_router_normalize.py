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


@pytest.mark.fast
def test_tool_router_unwraps_input_arguments() -> None:
    sandbox = FakeSandbox()
    router = ToolRouter()
    tool_call = {"function": {"name": "run_shell", "arguments": {"input": {"command": "echo hi"}}}}

    result = router.execute(tool_call, sandbox)

    assert sandbox.calls == [("echo hi", 30)]
    assert result["status"] == "success"
