from __future__ import annotations

import json
import shlex
from pathlib import Path
from typing import Any, Mapping

from gage_eval.agent_eval_kits.common import resolve_sample_artifact_target


def capture_terminal_workspace_state(
    sandbox_provider,
    *,
    cwd: str,
) -> dict[str, Any]:
    """Capture a stable snapshot of the workspace file manifest.

    Args:
        sandbox_provider: Runtime sandbox provider, when available.
        cwd: Workspace root inside the sandbox.

    Returns:
        A JSON-serializable payload describing the current workspace snapshot.
    """

    sandbox = _resolve_sandbox(sandbox_provider)
    if sandbox is None:
        return {
            "available": False,
            "cwd": cwd,
            "files": {},
            "error": "sandbox_unavailable",
        }
    try:
        return {
            "available": True,
            "cwd": cwd,
            "files": _capture_workspace_manifest(sandbox, cwd=cwd),
        }
    except Exception as exc:
        return {
            "available": False,
            "cwd": cwd,
            "files": {},
            "error": str(exc),
        }


def persist_terminal_artifacts(
    *,
    session,
    scheduler_output: Mapping[str, Any] | None,
    sandbox_provider,
) -> dict[str, str]:
    """Persist terminal benchmark artifacts under the sample artifact root.

    Args:
        session: Runtime session owning the artifact layout.
        scheduler_output: Raw scheduler output to normalize.
        sandbox_provider: Optional sandbox provider for workspace inspection.

    Returns:
        Sample-root-relative artifact paths keyed by evidence type.
    """

    output = dict(scheduler_output or {})
    agent_trace = output.get("agent_trace")
    if not isinstance(agent_trace, list):
        agent_trace = []

    stdout_text, stderr_text = _collect_streams(agent_trace)
    artifact_paths = {
        "tool_trace": _write_json_artifact(session, "tool_trace.json", agent_trace),
        "stdout": _write_text_artifact(session, "stdout.log", stdout_text),
        "stderr": _write_text_artifact(session, "stderr.log", stderr_text),
    }

    initial_workspace_state = session.benchmark_state.get("workspace_state")
    current_workspace_state = capture_terminal_workspace_state(
        sandbox_provider,
        cwd=str(session.runtime_context.get("cwd") or "/workspace"),
    )
    workspace_diff = _build_workspace_diff_payload(
        initial_state=initial_workspace_state,
        current_state=current_workspace_state,
    )
    artifact_paths["workspace_diff"] = _write_json_artifact(
        session,
        "workspace_diff.json",
        workspace_diff,
    )
    return artifact_paths


def _resolve_sandbox(sandbox_provider) -> Any | None:
    if sandbox_provider is None:
        return None
    handle = sandbox_provider.get_handle()
    return handle.sandbox if handle is not None else None


def _capture_workspace_manifest(sandbox: Any, *, cwd: str) -> dict[str, dict[str, Any]]:
    command = (
        f"cd {shlex.quote(cwd)} && "
        "find . -type f | LC_ALL=C sort | while IFS= read -r path; do "
        "clean=${path#./}; "
        "size=$(wc -c < \"$path\" 2>/dev/null || echo 0); "
        "sha=$(sha256sum \"$path\" 2>/dev/null | awk '{print $1}'); "
        "printf '%s\\t%s\\t%s\\n' \"$clean\" \"$size\" \"$sha\"; "
        "done"
    )
    result = sandbox.exec(command, timeout=20)
    if int(getattr(result, "exit_code", 1)) != 0:
        raise RuntimeError(str(getattr(result, "stderr", "") or "workspace_snapshot_failed"))

    manifest: dict[str, dict[str, Any]] = {}
    for line in str(getattr(result, "stdout", "") or "").splitlines():
        path, size, sha = _parse_manifest_line(line)
        if not path:
            continue
        manifest[path] = {
            "size_bytes": size,
            "sha256": sha,
        }
    return manifest


def _parse_manifest_line(line: str) -> tuple[str, int, str]:
    parts = line.split("\t", 2)
    if len(parts) != 3:
        return "", 0, ""
    raw_path, raw_size, raw_sha = parts
    try:
        size = int(raw_size)
    except (TypeError, ValueError):
        size = 0
    return raw_path.strip(), size, raw_sha.strip()


def _collect_streams(agent_trace: list[dict[str, Any]]) -> tuple[str, str]:
    stdout_chunks: list[str] = []
    stderr_chunks: list[str] = []
    for step in agent_trace:
        if not isinstance(step, dict):
            continue
        output = step.get("output")
        if not isinstance(output, Mapping):
            continue
        stdout = output.get("stdout")
        stderr = output.get("stderr")
        if stdout not in (None, ""):
            stdout_chunks.append(str(stdout))
        if stderr not in (None, ""):
            stderr_chunks.append(str(stderr))
    stdout_text = "\n".join(chunk.rstrip("\n") for chunk in stdout_chunks if chunk)
    stderr_text = "\n".join(chunk.rstrip("\n") for chunk in stderr_chunks if chunk)
    return stdout_text, stderr_text


def _build_workspace_diff_payload(
    *,
    initial_state: Mapping[str, Any] | None,
    current_state: Mapping[str, Any] | None,
) -> dict[str, Any]:
    before_available = bool(isinstance(initial_state, Mapping) and initial_state.get("available"))
    after_available = bool(isinstance(current_state, Mapping) and current_state.get("available"))
    before_files = dict(initial_state.get("files") or {}) if isinstance(initial_state, Mapping) else {}
    after_files = dict(current_state.get("files") or {}) if isinstance(current_state, Mapping) else {}

    added = sorted(path for path in after_files if path not in before_files)
    removed = sorted(path for path in before_files if path not in after_files)
    modified = sorted(
        path
        for path in after_files
        if path in before_files and after_files[path] != before_files[path]
    )

    return {
        "available": before_available and after_available,
        "cwd": (
            str(current_state.get("cwd"))
            if isinstance(current_state, Mapping) and current_state.get("cwd")
            else str(initial_state.get("cwd"))
            if isinstance(initial_state, Mapping) and initial_state.get("cwd")
            else "/workspace"
        ),
        "initial_file_count": len(before_files),
        "final_file_count": len(after_files),
        "added": added,
        "removed": removed,
        "modified": modified,
        "initial_error": initial_state.get("error") if isinstance(initial_state, Mapping) else None,
        "final_error": current_state.get("error") if isinstance(current_state, Mapping) else None,
    }


def _write_json_artifact(session, filename: str, payload: Any) -> str:
    target, relative_path = resolve_sample_artifact_target(session, filename)
    target.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )
    return relative_path


def _write_text_artifact(session, filename: str, content: str) -> str:
    target, relative_path = resolve_sample_artifact_target(session, filename)
    target.write_text(content or "", encoding="utf-8")
    return relative_path
