from __future__ import annotations

import pytest

from gage_eval.role.agent.tool_router import ToolRouter
from gage_eval.sandbox.base import ExecResult


class FakeSandbox:
    def __init__(self) -> None:
        self.exec_calls: list[str] = []
        self.writes: list[tuple[str, bytes]] = []

    def exec(self, command: str, timeout: int = 30) -> ExecResult:
        self.exec_calls.append(command)
        return ExecResult(exit_code=0, stdout="diff --git a/foo b/foo\n", stderr="", duration_ms=1.0)

    def write_file(self, path: str, content: bytes) -> None:
        if isinstance(content, str):
            content = content.encode("utf-8")
        self.writes.append((path, content))


@pytest.mark.fast
def test_tool_router_submit_patch_writes_submission_file() -> None:
    sandbox = FakeSandbox()
    router = ToolRouter()
    tool_call = {"function": {"name": "submit_patch_tool", "arguments": {"timeout_s": 15}}}

    result = router.execute(tool_call, sandbox)

    assert result["status"] == "success"
    assert sandbox.exec_calls == ["git diff"]
    assert sandbox.writes[0][0] == "/workspace/submission.patch"


@pytest.mark.fast
def test_tool_router_submit_patch_stages_untracked_files() -> None:
    sandbox = FakeSandbox()
    router = ToolRouter()
    tool_call = {"function": {"name": "submit_patch_tool", "arguments": {"stage_untracked": True}}}

    result = router.execute(tool_call, sandbox)

    assert result["status"] == "success"
    assert sandbox.exec_calls == ["git add -N -- .", "git diff"]
