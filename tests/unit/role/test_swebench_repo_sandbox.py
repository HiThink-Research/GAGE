from __future__ import annotations

import pytest

from gage_eval.role.context.swebench_repo import SwebenchRepoContext
from gage_eval.sandbox.base import ExecResult


class FakeSandbox:
    def __init__(self) -> None:
        self.commands: list[str] = []

    def exec(self, command: str, timeout: int = 30) -> ExecResult:
        self.commands.append(command)
        if "find . -maxdepth" in command:
            return ExecResult(
                exit_code=0,
                stdout="\n".join(["./", "./src", "./src/foo.py", "./tests", "./tests/test_bar.py"]),
                stderr="",
            )
        if "find . -type f" in command:
            return ExecResult(exit_code=0, stdout="./src/foo.py\n./tests/test_bar.py\n", stderr="")
        if "sed -n" in command and "src/foo.py" in command:
            return ExecResult(exit_code=0, stdout="print('ok')", stderr="")
        if "sed -n" in command and "tests/test_bar.py" in command:
            return ExecResult(exit_code=0, stdout="def test_bar():\n    assert True", stderr="")
        return ExecResult(exit_code=0, stdout="", stderr="")


class FakeHandle:
    def __init__(self, sandbox: FakeSandbox) -> None:
        self.sandbox = sandbox
        self.config = {"runtime_configs": {}}


class FakeProvider:
    def __init__(self, sandbox: FakeSandbox) -> None:
        self._handle = FakeHandle(sandbox)

    def get_handle(self) -> FakeHandle:
        return self._handle


@pytest.mark.fast
def test_swebench_repo_uses_sandbox_exec() -> None:
    sandbox = FakeSandbox()
    provider = FakeProvider(sandbox)
    context = SwebenchRepoContext(repo_source="docker_image", repo_root="/app", topk_files=2)
    sample = {
        "id": "s1",
        "messages": [{"role": "user", "content": "Fix the bug"}],
        "metadata": {"selected_test_files_to_run": ["tests/test_bar.py::test_bar"]},
    }
    payload = {"sample": sample, "params": {}, "sandbox_provider": provider}

    result = context.provide(payload)

    assert result["repo_tree"]
    assert "tests/test_bar.py" in result["selected_files"]
    assert "Repository Tree" in sample["messages"][0]["content"]
    assert any(cmd.startswith("cd /app") for cmd in sandbox.commands)
