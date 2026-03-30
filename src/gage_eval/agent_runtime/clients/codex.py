"""Codex CLI client driver."""

from __future__ import annotations

import os
import shlex
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

from gage_eval.agent_runtime.clients import ClientRunRequest, ClientRunResult


class CodexClient:
    """Drive a Codex-style CLI against an execution environment."""

    def __init__(
        self,
        *,
        executable: str = "codex",
        default_args: Sequence[str] | None = None,
    ) -> None:
        self._executable = executable
        self._default_args = tuple(default_args or ())

    def setup(self, environment: Any, session: Any) -> None:
        """Prepare the environment before execution."""
        return None

    def run(self, request: ClientRunRequest, environment: Any) -> ClientRunResult:
        """Execute the client command and collect terminal output."""
        command = _resolve_command(request, executable=self._executable, default_args=self._default_args)
        timeout_sec = _coerce_timeout(request.metadata.get("timeout_sec"), default=1800)
        env = dict(request.env)
        result = None
        if environment is not None and hasattr(environment, "exec"):
            result = environment.exec(command, cwd=request.cwd, env=env, timeout_sec=timeout_sec)
        else:
            result = _run_local(command, request.cwd, env, timeout_sec)

        stdout = _result_text(result, "stdout")
        stderr = _result_text(result, "stderr")
        exit_code = _result_exit_code(result)
        patch_path = _resolve_optional_path(request.metadata, ("patch_path", "submission_patch_path"))
        trajectory_path = _resolve_optional_path(request.metadata, ("trajectory_path", "trajectory_log_path"))
        stdout_path = _resolve_optional_path(request.metadata, ("stdout_path", "stdout_file", "output_path"))
        artifacts = _collect_artifacts(request.metadata)
        if patch_path:
            artifacts.setdefault("patch_path", patch_path)
        if trajectory_path:
            artifacts.setdefault("trajectory_path", trajectory_path)
        if stdout_path:
            artifacts.setdefault("stdout_path", stdout_path)
            _write_optional_file(environment, stdout_path, stdout)
        return ClientRunResult(
            exit_code=exit_code,
            stdout=stdout,
            stderr=stderr,
            patch_path=patch_path,
            trajectory_path=trajectory_path,
            artifacts=artifacts,
        )

    def cleanup(self, environment: Any, session: Any) -> None:
        """Release environment resources without raising on failure."""
        return None


def _resolve_command(
    request: ClientRunRequest,
    *,
    executable: str,
    default_args: Sequence[str],
) -> str:
    metadata = request.metadata or {}
    command = metadata.get("command") or metadata.get("cli_command")
    if isinstance(command, str) and command.strip():
        return command
    argv = metadata.get("argv")
    if isinstance(argv, (list, tuple)) and argv:
        return " ".join(shlex.quote(str(arg)) for arg in argv)
    args = [executable, *default_args]
    instruction = request.instruction.strip()
    if instruction:
        args.extend(["--instruction", instruction])
    if request.cwd:
        args.extend(["--cwd", request.cwd])
    return " ".join(shlex.quote(str(arg)) for arg in args)


def _coerce_timeout(value: Any, *, default: int) -> int:
    try:
        return max(1, int(value))
    except (TypeError, ValueError):
        return int(default)


def _result_text(result: Any, key: str) -> str:
    value = getattr(result, key, "")
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return str(value or "")


def _result_exit_code(result: Any) -> int:
    try:
        return int(getattr(result, "exit_code", 0) or 0)
    except (TypeError, ValueError):
        return 1


def _resolve_optional_path(metadata: Mapping[str, Any], keys: Sequence[str]) -> Optional[str]:
    for key in keys:
        value = metadata.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None


def _collect_artifacts(metadata: Mapping[str, Any]) -> dict[str, str]:
    artifacts = metadata.get("artifacts")
    if isinstance(artifacts, dict):
        return {str(key): str(value) for key, value in artifacts.items() if value is not None}
    return {}


def _write_optional_file(environment: Any, path: str, content: str) -> None:
    if not path:
        return
    writer = getattr(environment, "write_file", None)
    if callable(writer):
        try:
            writer(path, content.encode("utf-8"))
            return
        except Exception:
            pass
    target = Path(path)
    try:
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")
    except Exception:
        return


def _run_local(command: str, cwd: str, env: Mapping[str, str], timeout_sec: int) -> Any:
    import subprocess

    completed = subprocess.run(  # noqa: S603,S607 - command is intentionally shell-form
        command,
        cwd=cwd or None,
        env={**os.environ, **dict(env)} if env else None,
        shell=True,
        capture_output=True,
        text=True,
        timeout=timeout_sec,
        check=False,
    )
    return completed
