from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Mapping

from gage_eval.agent_eval_kits.common import resolve_sample_artifact_target


def persist_swebench_artifacts(
    *,
    session: Any,
    scheduler_output: Mapping[str, Any] | None,
    sandbox_provider: Any = None,
) -> dict[str, str]:
    """Persist SWE-bench runtime artifacts under the sample artifact root.

    Args:
        session: Runtime session owning the sample artifact layout.
        scheduler_output: Raw scheduler output to normalize.
        sandbox_provider: Optional sandbox provider for patch extraction.

    Returns:
        A mapping of artifact keys to sample-root-relative paths.
    """

    output = dict(scheduler_output or {})
    persisted: dict[str, str] = {}

    # STEP 1: Preserve already-materialized artifacts when they exist on disk.
    existing_paths = output.get("artifact_paths")
    if isinstance(existing_paths, Mapping):
        persisted.update(_materialized_artifact_paths(session, existing_paths))

    # STEP 2: Persist patch evidence from the strongest available source.
    patch_content = _resolve_patch_content(output, sandbox_provider=sandbox_provider)
    if patch_content:
        persisted["submission_patch"] = _write_text_artifact(
            session,
            "submission.patch",
            patch_content,
        )

    # STEP 3: Persist lightweight diagnostics so the artifact directory is auditable.
    agent_trace = output.get("agent_trace")
    trace_diagnostics = _build_trace_diagnostics(agent_trace)
    if isinstance(agent_trace, list) and agent_trace:
        persisted["agent_trace"] = _write_json_artifact(
            session,
            "agent_trace.json",
            agent_trace,
        )
    answer = output.get("answer")
    if isinstance(answer, str) and answer.strip():
        persisted["final_response"] = _write_text_artifact(
            session,
            "final_response.txt",
            answer,
        )
    persisted["swebench_diagnostics"] = _write_json_artifact(
        session,
        "swebench_diagnostics.json",
        {
            "submission_patch_present": "submission_patch" in persisted,
            "agent_trace_step_count": len(agent_trace) if isinstance(agent_trace, list) else 0,
            "answer_present": isinstance(answer, str) and bool(answer.strip()),
            "required_artifact": "submission.patch",
            "missing_commands": trace_diagnostics["missing_commands"],
            "recent_errors": trace_diagnostics["recent_errors"],
        },
    )

    return persisted


def _materialized_artifact_paths(
    session: Any,
    artifact_paths: Mapping[str, Any],
) -> dict[str, str]:
    sample_root = Path(str(getattr(session, "artifact_layout", {}).get("sample_root") or "."))
    resolved: dict[str, str] = {}
    for key, raw_path in artifact_paths.items():
        if not isinstance(raw_path, str) or not raw_path.strip():
            continue
        path = Path(raw_path)
        absolute = path if path.is_absolute() else sample_root / path
        if absolute.exists():
            resolved[str(key)] = (
                absolute.relative_to(sample_root).as_posix()
                if absolute.is_relative_to(sample_root)
                else str(absolute)
            )
    return resolved


def _resolve_patch_content(
    output: Mapping[str, Any],
    *,
    sandbox_provider: Any,
) -> str:
    patch_fields = ("patch_content", "patch", "diff")
    for field in patch_fields:
        value = output.get(field)
        if isinstance(value, str) and value.strip():
            return _normalize_text_block(value)
    agent_trace = output.get("agent_trace")
    if isinstance(agent_trace, list):
        patch_from_trace = _resolve_patch_from_agent_trace(agent_trace)
        if patch_from_trace:
            return patch_from_trace
    sandbox_patch = _read_submission_patch_from_sandbox(sandbox_provider)
    if sandbox_patch:
        return sandbox_patch
    return _read_git_diff_from_sandbox(sandbox_provider)


def _resolve_patch_from_agent_trace(agent_trace: list[Any]) -> str:
    for entry in reversed(agent_trace):
        if not isinstance(entry, Mapping):
            continue
        name = str(entry.get("name") or entry.get("tool") or "")
        if name not in {"submit_patch_tool", "codex_exec"}:
            continue
        output = entry.get("output")
        patch = _extract_text_payload(output)
        if "diff --git" in patch or patch.startswith("*** Begin Patch"):
            return _normalize_text_block(patch)
    return ""


def _read_submission_patch_from_sandbox(sandbox_provider: Any) -> str:
    if sandbox_provider is None:
        return ""
    handle = getattr(sandbox_provider, "get_handle", lambda: None)()
    sandbox = getattr(handle, "sandbox", None) if handle is not None else None
    if sandbox is None:
        return ""
    reader = getattr(sandbox, "read_file", None)
    if callable(reader):
        try:
            payload = reader("/workspace/submission.patch")
        except Exception:
            payload = ""
        return _normalize_text_block(_decode_payload(payload))
    executor = getattr(sandbox, "exec", None)
    if not callable(executor):
        return ""
    try:
        result = executor("cat /workspace/submission.patch", timeout=5)
    except Exception:
        return ""
    stdout = getattr(result, "stdout", "")
    return _normalize_text_block(_decode_payload(stdout))


def _read_git_diff_from_sandbox(sandbox_provider: Any) -> str:
    if sandbox_provider is None:
        return ""
    handle = getattr(sandbox_provider, "get_handle", lambda: None)()
    sandbox = getattr(handle, "sandbox", None) if handle is not None else None
    executor = getattr(sandbox, "exec", None)
    if not callable(executor):
        return ""
    commands = (
        "cd /workspace && git diff --binary --no-color",
        "cd /workspace && git diff --binary --no-color HEAD --",
    )
    for command in commands:
        try:
            result = executor(command, timeout=10)
        except Exception:
            continue
        stdout = _normalize_text_block(_decode_payload(getattr(result, "stdout", "")))
        if "diff --git" in stdout or stdout.startswith("*** Begin Patch"):
            return stdout
    return ""


def _extract_text_payload(payload: Any) -> str:
    if isinstance(payload, str):
        return payload
    if isinstance(payload, Mapping):
        for key in ("patch", "diff", "stdout", "text", "content", "answer"):
            value = payload.get(key)
            if isinstance(value, str) and value.strip():
                return value
        nested = payload.get("output")
        if nested is not None and nested is not payload:
            return _extract_text_payload(nested)
    return ""


def _decode_payload(payload: Any) -> str:
    if isinstance(payload, (bytes, bytearray)):
        return payload.decode("utf-8", errors="replace")
    if payload is None:
        return ""
    return str(payload)


def _normalize_text_block(value: str) -> str:
    text = value.strip()
    if not text:
        return ""
    if not text.endswith("\n"):
        text += "\n"
    return text


def _build_trace_diagnostics(agent_trace: Any) -> dict[str, list[str]]:
    if not isinstance(agent_trace, list):
        return {"missing_commands": [], "recent_errors": []}

    missing_commands: list[str] = []
    recent_errors: list[str] = []
    for entry in agent_trace[-5:]:
        if not isinstance(entry, Mapping):
            continue
        output = entry.get("output")
        stderr = ""
        if isinstance(output, Mapping):
            stderr = _decode_payload(output.get("stderr"))
        elif isinstance(entry.get("stderr"), str):
            stderr = str(entry.get("stderr") or "")
        for line in stderr.splitlines():
            stripped = line.strip()
            if not stripped:
                continue
            recent_errors.append(stripped)
            match = re.search(r": ([A-Za-z0-9_.-]+): not found$", stripped)
            if match:
                command = match.group(1)
                if command not in missing_commands:
                    missing_commands.append(command)
    return {
        "missing_commands": missing_commands,
        "recent_errors": recent_errors[-5:],
    }


def _write_json_artifact(session: Any, filename: str, payload: Any) -> str:
    target, relative_path = resolve_sample_artifact_target(session, filename)
    target.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )
    return relative_path


def _write_text_artifact(session: Any, filename: str, content: str) -> str:
    target, relative_path = resolve_sample_artifact_target(session, filename)
    target.write_text(content, encoding="utf-8")
    return relative_path
