from __future__ import annotations

from pathlib import Path

import pytest

from gage_eval.agent_runtime.clients import ClientRunRequest
from gage_eval.agent_runtime.clients.codex import CodexClient, _resolve_command
from gage_eval.sandbox.base import ExecResult


class _FakeEnvironment:
    def __init__(self) -> None:
        self.commands: list[str] = []
        self.files: dict[str, bytes] = {}
        self.fail_first = False
        self.tracked_diff: str = "diff --git a/a b/a\n"
        self.untracked_files: list[str] = []
        self.untracked_diff: str = ""
        self.path_resolver = None

    def exec(self, command: str, *, cwd: str | None = None, env=None, timeout_sec: int = 30):
        self.commands.append(command)
        if self.fail_first and len(self.commands) == 1:
            return ExecResult(exit_code=1, stdout="", stderr="usage limit")
        if command == "git diff --binary -- .":
            return ExecResult(exit_code=0, stdout=self.tracked_diff, stderr="")
        if command == "git ls-files --others --exclude-standard -- .":
            return ExecResult(exit_code=0, stdout="\n".join(self.untracked_files), stderr="")
        if command.startswith("git diff --no-index --binary /dev/null "):
            return ExecResult(exit_code=1, stdout=self.untracked_diff, stderr="")
        return ExecResult(exit_code=0, stdout="raw-cli-output", stderr="")

    def read_file(self, path: str) -> bytes:
        return self.files[path]

    def write_file(self, path: str, content: bytes) -> None:
        self.files[path] = content

    def resolve_execution_path(self, path: str) -> str:
        if callable(self.path_resolver):
            return self.path_resolver(path)
        return path


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

    codex_command = next(command for command in environment.commands if command.startswith("codex exec "))
    assert environment.commands[0] == "mkdir -p /tmp"
    assert codex_command.startswith("codex exec --skip-git-repo-check --full-auto")
    assert "--output-last-message /tmp/stdout.log" in codex_command
    assert " --cd " not in codex_command
    assert result.stdout == "final message"
    assert result.patch_path == "/tmp/submission.patch"
    assert result.patch_content == "diff --git a/a b/a\n"
    assert result.trajectory_path == "/tmp/trajectory.log"
    assert result.artifacts["stdout_path"] == "/tmp/stdout.log"
    assert environment.files["/tmp/submission.patch"].decode("utf-8").startswith("diff --git")
    assert Path("/tmp/submission.patch").read_text(encoding="utf-8").startswith("diff --git")
    assert "$ codex exec" in environment.files["/tmp/trajectory.log"].decode("utf-8")
    assert Path("/tmp/stdout.log").read_text(encoding="utf-8") == "final message"


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
    assert " --cd " not in environment.commands[0]
    assert environment.commands[1] == "python3 -c \"print('fallback ok')\""
    assert result.patch_content == "diff --git a/a b/a\n"


@pytest.mark.fast
def test_codex_client_persists_local_artifacts_when_environment_is_remote(tmp_path) -> None:
    environment = _FakeEnvironment()
    stdout_path = tmp_path / "stdout.log"
    patch_path = tmp_path / "submission.patch"
    trajectory_path = tmp_path / "trajectory.log"
    environment.files[str(stdout_path)] = b"final message"
    client = CodexClient()

    result = client.run(
        ClientRunRequest(
            instruction="Fix the test and stop.",
            cwd="/workspace/repo",
            metadata={
                "stdout_path": str(stdout_path),
                "patch_path": str(patch_path),
                "trajectory_path": str(trajectory_path),
            },
        ),
        environment,
    )

    assert result.exit_code == 0
    assert stdout_path.read_text(encoding="utf-8") == "final message"
    assert patch_path.read_text(encoding="utf-8").startswith("diff --git")
    assert trajectory_path.read_text(encoding="utf-8").startswith("$ codex exec")


@pytest.mark.fast
def test_codex_client_collects_patch_from_untracked_files() -> None:
    environment = _FakeEnvironment()
    environment.tracked_diff = ""
    environment.untracked_files = ["answer.py"]
    environment.untracked_diff = (
        "diff --git a/answer.py b/answer.py\n"
        "new file mode 100644\n"
        "--- /dev/null\n"
        "+++ b/answer.py\n"
        "@@\n"
        "+def compute_answer() -> int:\n"
        "+    return 42\n"
    )
    client = CodexClient()

    result = client.run(
        ClientRunRequest(
            instruction="Fix the test and stop.",
            cwd="/workspace/repo",
            metadata={
                "patch_path": "/tmp/submission.patch",
            },
        ),
        environment,
    )

    assert result.patch_content is not None
    assert "return 42" in result.patch_content
    assert environment.commands.count("git diff --binary -- .") == 1
    assert "git ls-files --others --exclude-standard -- ." in environment.commands


@pytest.mark.fast
def test_resolve_command_keeps_cd_for_local_execution() -> None:
    command = _resolve_command(
        ClientRunRequest(
            instruction="Fix the test and stop.",
            cwd="/workspace/repo",
        ),
        executable="codex",
        default_args=(),
        output_path=None,
        include_cwd=True,
    )

    assert " --cd /workspace/repo " in f" {command} "


@pytest.mark.fast
def test_resolve_command_skips_full_auto_for_dangerous_bypass() -> None:
    command = _resolve_command(
        ClientRunRequest(
            instruction="Fix the test and stop.",
            cwd="/workspace/repo",
        ),
        executable="codex",
        default_args=("--dangerously-bypass-approvals-and-sandbox",),
        output_path=None,
        include_cwd=False,
    )

    assert "--dangerously-bypass-approvals-and-sandbox" in command
    assert "--full-auto" not in command


@pytest.mark.fast
def test_codex_client_maps_output_last_message_path_for_environment(monkeypatch, tmp_path) -> None:
    environment = _FakeEnvironment()
    monkeypatch.chdir(tmp_path)

    def _resolve(path: str) -> str:
        return f"/sandbox/{path}"

    environment.path_resolver = _resolve
    environment.files["/sandbox/runs/out.log"] = b"final message"
    client = CodexClient()

    result = client.run(
        ClientRunRequest(
            instruction="Fix the test and stop.",
            cwd="/workspace/repo",
            metadata={"stdout_path": "runs/out.log"},
        ),
        environment,
    )

    assert environment.commands[0] == "mkdir -p /sandbox/runs"
    assert "--output-last-message /sandbox/runs/out.log" in environment.commands[1]
    assert result.stdout == "final message"
    assert (tmp_path / "runs/out.log").read_text(encoding="utf-8") == "final message"
