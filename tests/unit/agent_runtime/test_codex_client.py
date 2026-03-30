from __future__ import annotations

import pytest

from gage_eval.agent_runtime.clients import ClientRunRequest
from gage_eval.agent_runtime.clients.codex import CodexClient
from gage_eval.sandbox.base import ExecResult


class _FakeEnvironment:
    def __init__(self) -> None:
        self.commands: list[str] = []
        self.files: dict[str, bytes] = {}
        self.fail_first = False

    def exec(self, command: str, *, cwd: str | None = None, env=None, timeout_sec: int = 30):
        self.commands.append(command)
        if self.fail_first and len(self.commands) == 1:
            return ExecResult(exit_code=1, stdout="", stderr="usage limit")
        if command == "git diff --binary -- .":
            return ExecResult(exit_code=0, stdout="diff --git a/a b/a\n", stderr="")
        return ExecResult(exit_code=0, stdout="raw-cli-output", stderr="")

    def read_file(self, path: str) -> bytes:
        return self.files[path]

    def write_file(self, path: str, content: bytes) -> None:
        self.files[path] = content


@pytest.mark.fast
def test_codex_client_runs_codex_exec_and_collects_artifacts() -> None:
    environment = _FakeEnvironment()
    environment.files["/tmp/stdout.log"] = b"final message"
    client = CodexClient()

    result = client.run(
        ClientRunRequest(
            instruction="Fix the test and stop.",
            cwd="/workspace/repo",
            env={"OPENAI_API_KEY": "test-key"},
            metadata={
                "stdout_path": "/tmp/stdout.log",
                "patch_path": "/tmp/submission.patch",
                "trajectory_path": "/tmp/trajectory.log",
            },
        ),
        environment,
    )

    assert environment.commands
    assert environment.commands[0].startswith("codex exec --skip-git-repo-check --full-auto")
    assert "--output-last-message /tmp/stdout.log" in environment.commands[0]
    assert result.stdout == "final message"
    assert result.patch_path == "/tmp/submission.patch"
    assert result.patch_content == "diff --git a/a b/a\n"
    assert result.trajectory_path == "/tmp/trajectory.log"
    assert result.artifacts["stdout_path"] == "/tmp/stdout.log"
    assert environment.files["/tmp/submission.patch"].decode("utf-8").startswith("diff --git")
    assert "$ codex exec" in environment.files["/tmp/trajectory.log"].decode("utf-8")


@pytest.mark.fast
def test_codex_client_uses_fallback_command_when_primary_fails() -> None:
    environment = _FakeEnvironment()
    environment.fail_first = True
    client = CodexClient()

    result = client.run(
        ClientRunRequest(
            instruction="Fix the test and stop.",
            cwd="/workspace/repo",
            metadata={
                "patch_path": "/tmp/submission.patch",
                "fallback_command": "python3 -c \"print('fallback ok')\"",
            },
        ),
        environment,
    )

    assert result.exit_code == 0
    assert environment.commands[0].startswith("codex exec --skip-git-repo-check --full-auto")
    assert environment.commands[1] == "python3 -c \"print('fallback ok')\""
    assert result.patch_content == "diff --git a/a b/a\n"
