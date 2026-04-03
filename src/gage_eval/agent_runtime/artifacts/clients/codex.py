"""Codex CLI client driver."""

from __future__ import annotations

import os
import shlex
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

from repo.src.gage_eval.agent_runtime.artifacts.clients import ClientRunRequest, ClientRunResult


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
        stdout_path = _resolve_optional_path(
            request.metadata,
            ("stdout_path", "stdout_file", "output_path", "output_last_message_path"),
        )
        stdout_execution_path = _resolve_execution_path(environment, stdout_path)
        run_in_environment = environment is not None and hasattr(environment, "exec")
        timeout_sec = _coerce_timeout(request.metadata.get("timeout_sec"), default=1800)
        env = dict(request.env)
        if run_in_environment:
            _ensure_parent_directory(
                environment=environment,
                path=stdout_execution_path,
                cwd=request.cwd,
                env=env,
                timeout_sec=timeout_sec,
            )
        command = _resolve_command(
            request,
            executable=self._executable,
            default_args=self._default_args,
            output_path=stdout_execution_path,
            include_cwd=not run_in_environment,
        )
        result = None
        if run_in_environment:
            result = environment.exec(command, cwd=request.cwd, env=env, timeout_sec=timeout_sec)
        else:
            result = _run_local(command, request.cwd, env, timeout_sec)
        result, command = _apply_fallback_if_needed(
            result=result,
            command=command,
            metadata=request.metadata,
            environment=environment,
            cwd=request.cwd,
            env=env,
            timeout_sec=timeout_sec,
        )

        stdout = _result_text(result, "stdout")
        stderr = _result_text(result, "stderr")
        exit_code = _result_exit_code(result)
        patch_path = _resolve_optional_path(request.metadata, ("patch_path", "submission_patch_path"))
        trajectory_path = _resolve_optional_path(request.metadata, ("trajectory_path", "trajectory_log_path"))
        patch_execution_path = _resolve_execution_path(environment, patch_path)
        trajectory_execution_path = _resolve_execution_path(environment, trajectory_path)
        artifacts = _collect_artifacts(request.metadata)
        patch_content: Optional[str] = None
        stdout_capture = _read_optional_file(
            environment=environment,
            environment_path=stdout_execution_path,
            local_path=stdout_path,
        )
        if stdout_capture:
            stdout = stdout_capture
        if patch_path:
            patch_content = _collect_patch(environment, cwd=request.cwd, timeout_sec=timeout_sec)
            if not patch_content:
                patch_content = _collect_patch_from_transcript(stderr, stdout)
            if patch_content:
                _write_optional_file(
                    environment=environment,
                    environment_path=patch_execution_path,
                    local_path=patch_path,
                    content=patch_content,
                )
            artifacts.setdefault("patch_path", patch_path)
        if trajectory_path:
            transcript = _build_transcript(command, stdout, stderr)
            _write_optional_file(
                environment=environment,
                environment_path=trajectory_execution_path,
                local_path=trajectory_path,
                content=transcript,
            )
            artifacts.setdefault("trajectory_path", trajectory_path)
        if stdout_path:
            artifacts.setdefault("stdout_path", stdout_path)
            _write_optional_file(
                environment=environment,
                environment_path=stdout_execution_path,
                local_path=stdout_path,
                content=stdout,
            )
        return ClientRunResult(
            exit_code=exit_code,
            stdout=stdout,
            stderr=stderr,
            patch_path=patch_path,
            patch_content=patch_content,
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
    output_path: Optional[str],
    include_cwd: bool,
) -> str:
    metadata = request.metadata or {}
    command = metadata.get("command") or metadata.get("cli_command")
    if isinstance(command, str) and command.strip():
        return command
    argv = metadata.get("argv")
    if isinstance(argv, (list, tuple)) and argv:
        return " ".join(shlex.quote(str(arg)) for arg in argv)
    prompt = request.instruction.strip()
    args = [
        executable,
        "exec",
        "--skip-git-repo-check",
    ]
    if _should_use_full_auto(default_args):
        args.append("--full-auto")
    args.extend(default_args)
    if include_cwd and request.cwd:
        args.extend(["--cd", request.cwd])
    if output_path:
        args.extend(["--output-last-message", output_path])
    if prompt:
        args.append(prompt)
    return " ".join(shlex.quote(str(arg)) for arg in args)


def _coerce_timeout(value: Any, *, default: int) -> int:
    try:
        return max(1, int(value))
    except (TypeError, ValueError):
        return int(default)


def _should_use_full_auto(default_args: Sequence[str]) -> bool:
    disallow = {"--dangerously-bypass-approvals-and-sandbox"}
    return not any(str(arg) in disallow for arg in default_args)


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


def _resolve_execution_path(environment: Any, path: Optional[str]) -> Optional[str]:
    if not path:
        return None
    resolver = getattr(environment, "resolve_execution_path", None)
    if callable(resolver):
        try:
            resolved = resolver(path)
        except Exception:
            resolved = None
        if isinstance(resolved, str) and resolved.strip():
            return resolved
    return path


def _ensure_parent_directory(
    *,
    environment: Any,
    path: Optional[str],
    cwd: str,
    env: Mapping[str, str],
    timeout_sec: int,
) -> None:
    if not path or environment is None or not hasattr(environment, "exec"):
        return
    parent = Path(path).parent
    if not str(parent) or str(parent) == ".":
        return
    command = f"mkdir -p {shlex.quote(str(parent))}"
    try:
        environment.exec(command, cwd=cwd, env=env, timeout_sec=timeout_sec)
    except Exception:
        return


def _write_optional_file(
    *,
    environment: Any,
    environment_path: Optional[str],
    local_path: Optional[str],
    content: str,
) -> None:
    if not environment_path and not local_path:
        return
    writer = getattr(environment, "write_file", None)
    if callable(writer) and environment_path:
        try:
            writer(environment_path, content.encode("utf-8"))
        except Exception:
            pass
    if not local_path:
        return
    target = Path(local_path)
    try:
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")
    except Exception:
        return


def _read_optional_file(
    *,
    environment: Any,
    environment_path: Optional[str],
    local_path: Optional[str],
) -> Optional[str]:
    if not environment_path and not local_path:
        return None
    reader = getattr(environment, "read_file", None)
    if callable(reader) and environment_path:
        try:
            content = reader(environment_path)
            if isinstance(content, bytes):
                return content.decode("utf-8", errors="replace")
            return str(content)
        except Exception:
            pass
    if not local_path:
        return None
    target = Path(local_path)
    try:
        return target.read_text(encoding="utf-8")
    except Exception:
        return None


def _collect_patch(environment: Any, *, cwd: str, timeout_sec: int) -> Optional[str]:
    git_diff = "git diff --binary -- ."
    if environment is not None and hasattr(environment, "exec"):
        try:
            result = environment.exec(git_diff, cwd=cwd, env=None, timeout_sec=timeout_sec)
            if _result_exit_code(result) == 0:
                stdout = _result_text(result, "stdout")
                if stdout:
                    return stdout
                return _collect_untracked_patch_from_environment(
                    environment,
                    cwd=cwd,
                    timeout_sec=timeout_sec,
                )
        except Exception:
            return None
        return None
    tracked_patch = _run_local_capture(git_diff, cwd, timeout_sec)
    if tracked_patch:
        return tracked_patch
    return _collect_untracked_patch_from_local(cwd, timeout_sec)


def _collect_patch_from_transcript(*texts: str) -> Optional[str]:
    for text in texts:
        patch = _extract_last_diff_block(text)
        if patch:
            return patch
    return None


def _extract_last_diff_block(text: str) -> Optional[str]:
    if not text:
        return None
    lines = text.splitlines()
    blocks: list[str] = []
    current: list[str] = []
    in_block = False
    for line in lines:
        if line.startswith("diff --git "):
            if current:
                blocks.append("\n".join(current))
            current = [line]
            in_block = True
            continue
        if not in_block:
            continue
        if _is_diff_line(line) or not line.strip():
            current.append(line)
            continue
        if current:
            blocks.append("\n".join(current))
        current = []
        in_block = False
    if current:
        blocks.append("\n".join(current))
    for block in reversed(blocks):
        normalized = _normalize_diff_block(block)
        if normalized:
            return normalized
    return None


def _normalize_diff_block(block: str) -> Optional[str]:
    trimmed = _trim_diff_tail(block).strip()
    if not trimmed or not trimmed.startswith("diff --git "):
        return None
    has_headers = False
    has_hunk = False
    for line in trimmed.splitlines():
        if line.startswith(("--- ", "+++ ")):
            has_headers = True
        if line.startswith("@@"):
            has_hunk = True
    if not has_headers or not has_hunk:
        return None
    if not trimmed.endswith("\n"):
        trimmed += "\n"
    return trimmed


def _trim_diff_tail(text: str) -> str:
    lines = text.splitlines()
    if not lines:
        return text
    last_idx = None
    for idx in range(len(lines) - 1, -1, -1):
        if _is_diff_line(lines[idx]):
            last_idx = idx
            break
    if last_idx is None:
        return text
    return "\n".join(lines[: last_idx + 1])


def _is_diff_line(line: str) -> bool:
    return line.startswith(_DIFF_PREFIXES) or line[:1] in (" ", "+", "-", "\\")


def _run_local_capture(command: str, cwd: str, timeout_sec: int) -> Optional[str]:
    import subprocess

    try:
        completed = subprocess.run(  # noqa: S603,S607 - command is intentionally shell-form
            command,
            cwd=cwd or None,
            shell=True,
            capture_output=True,
            text=True,
            timeout=timeout_sec,
            check=False,
        )
    except Exception:
        return None
    if completed.returncode != 0:
        return None
    return completed.stdout or None


def _collect_untracked_patch_from_environment(
    environment: Any,
    *,
    cwd: str,
    timeout_sec: int,
) -> Optional[str]:
    files = _list_untracked_files_via_environment(environment, cwd=cwd, timeout_sec=timeout_sec)
    if not files:
        return None
    chunks: list[str] = []
    for file_path in files:
        command = f"git diff --no-index --binary /dev/null {shlex.quote(file_path)}"
        result = environment.exec(command, cwd=cwd, env=None, timeout_sec=timeout_sec)
        exit_code = _result_exit_code(result)
        if exit_code not in {0, 1}:
            continue
        stdout = _result_text(result, "stdout")
        if stdout:
            chunks.append(stdout)
    return "".join(chunks) or None


def _collect_untracked_patch_from_local(cwd: str, timeout_sec: int) -> Optional[str]:
    files = _list_untracked_files_local(cwd, timeout_sec)
    if not files:
        return None
    chunks: list[str] = []
    for file_path in files:
        import subprocess

        try:
            completed = subprocess.run(  # noqa: S603,S607 - shell command is intentional
                f"git diff --no-index --binary /dev/null {shlex.quote(file_path)}",
                cwd=cwd or None,
                shell=True,
                capture_output=True,
                text=True,
                timeout=timeout_sec,
                check=False,
            )
        except Exception:
            continue
        if completed.returncode not in {0, 1}:
            continue
        if completed.stdout:
            chunks.append(completed.stdout)
    return "".join(chunks) or None


def _list_untracked_files_via_environment(environment: Any, *, cwd: str, timeout_sec: int) -> list[str]:
    result = environment.exec(
        "git ls-files --others --exclude-standard -- .",
        cwd=cwd,
        env=None,
        timeout_sec=timeout_sec,
    )
    if _result_exit_code(result) != 0:
        return []
    return [line.strip() for line in _result_text(result, "stdout").splitlines() if line.strip()]


def _list_untracked_files_local(cwd: str, timeout_sec: int) -> list[str]:
    import subprocess

    try:
        completed = subprocess.run(  # noqa: S603,S607 - shell command is intentional
            "git ls-files --others --exclude-standard -- .",
            cwd=cwd or None,
            shell=True,
            capture_output=True,
            text=True,
            timeout=timeout_sec,
            check=False,
        )
    except Exception:
        return []
    if completed.returncode != 0:
        return []
    return [line.strip() for line in completed.stdout.splitlines() if line.strip()]


_DIFF_PREFIXES = (
    "diff --git ",
    "index ",
    "--- ",
    "+++ ",
    "@@ ",
    "@@@ ",
    "new file mode ",
    "deleted file mode ",
    "rename from ",
    "rename to ",
    "similarity index ",
    "dissimilarity index ",
    "old mode ",
    "new mode ",
    "GIT binary patch",
    "Binary files ",
)


def _build_transcript(command: str, stdout: str, stderr: str) -> str:
    lines = [f"$ {command}"]
    if stdout:
        lines.append(stdout.rstrip())
    if stderr:
        lines.append("[stderr]")
        lines.append(stderr.rstrip())
    return "\n".join(line for line in lines if line).rstrip() + "\n"


def _apply_fallback_if_needed(
    *,
    result: Any,
    command: str,
    metadata: Mapping[str, Any],
    environment: Any,
    cwd: str,
    env: Mapping[str, str],
    timeout_sec: int,
) -> tuple[Any, str]:
    fallback_command = metadata.get("fallback_command")
    if _result_exit_code(result) == 0:
        return result, command
    if not isinstance(fallback_command, str) or not fallback_command.strip():
        return result, command
    if environment is not None and hasattr(environment, "exec"):
        fallback_result = environment.exec(fallback_command, cwd=cwd, env=env, timeout_sec=timeout_sec)
    else:
        fallback_result = _run_local(fallback_command, cwd, env, timeout_sec)
    combined_stderr = _combine_streams(
        _result_text(result, "stderr"),
        f"[fallback command]\\n{_result_text(fallback_result, 'stderr')}",
    )
    merged = _MutableExecResult(
        exit_code=_result_exit_code(fallback_result),
        stdout=_result_text(fallback_result, "stdout"),
        stderr=combined_stderr,
    )
    return merged, f"{command}\n# fallback\n{fallback_command}"


def _combine_streams(primary: str, secondary: str) -> str:
    chunks = [chunk for chunk in (primary.strip(), secondary.strip()) if chunk]
    return "\n".join(chunks)


class _MutableExecResult:
    def __init__(self, *, exit_code: int, stdout: str, stderr: str) -> None:
        self.exit_code = exit_code
        self.stdout = stdout
        self.stderr = stderr


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
