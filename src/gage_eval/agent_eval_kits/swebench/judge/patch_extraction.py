from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Mapping


@dataclass(frozen=True)
class PatchResolution:
    source: str
    patch: str
    failure_code: str | None = None


async def resolve_patch(
    *,
    model_output: Mapping[str, Any],
    sample: Mapping[str, Any],
    environment: Any,
    trace: Any | None = None,
) -> PatchResolution:
    patch = clean_patch_content(_resolve_patch(model_output))
    if patch:
        _emit_patch_source_event(trace, sample, source="model_output")
        return PatchResolution(source="model_output", patch=patch)

    trace_patch = clean_patch_content(_resolve_patch_from_agent_trace(model_output))
    if trace_patch:
        _emit_patch_source_event(trace, sample, source="submit_patch_stdout")
        return PatchResolution(source="submit_patch_stdout", patch=trace_patch)

    submission_patch = clean_patch_content(await _read_submission_patch(environment))
    if submission_patch:
        _emit_patch_source_event(
            trace,
            sample,
            source="workspace_submission_patch",
            path="/workspace/submission.patch",
        )
        _emit_patch_fallback_event(
            trace,
            sample,
            source="workspace_submission_patch",
            path="/workspace/submission.patch",
        )
        return PatchResolution(source="workspace_submission_patch", patch=submission_patch)

    git_diff = clean_patch_content(await _read_git_diff(environment))
    if git_diff:
        _emit_patch_source_event(
            trace,
            sample,
            source="git_diff_fallback",
            path="/workspace",
        )
        _emit_patch_fallback_event(
            trace,
            sample,
            source="git_diff_fallback",
            path="/workspace",
        )
        return PatchResolution(source="git_diff_fallback", patch=git_diff)

    return PatchResolution(source="missing", patch="", failure_code="artifact_capture.patch_missing")


def clean_patch_content(raw: str) -> str:
    if not raw:
        return ""
    cleaned = raw.strip()
    fenced = _extract_code_block(cleaned)
    if fenced:
        cleaned = fenced.strip()
    cleaned = _strip_apply_patch_markers(cleaned)
    diff_index = cleaned.find("diff --git")
    if diff_index != -1:
        cleaned = cleaned[diff_index:]
    cleaned = _strip_binary_hunks(cleaned)
    if _has_diff_markers(cleaned):
        cleaned = _trim_diff_tail(cleaned)
        cleaned = _normalize_hunk_context_lines(cleaned)
    cleaned = cleaned.strip()
    if cleaned and not _has_diff_markers(cleaned):
        return ""
    if cleaned and not cleaned.endswith("\n"):
        cleaned += "\n"
    return cleaned


def _resolve_patch(model_output: Mapping[str, Any]) -> str:
    for key in ("patch_content", "patch", "diff", "answer", "text", "content"):
        value = model_output.get(key)
        if isinstance(value, str) and value.strip():
            return value
    message = model_output.get("message")
    if isinstance(message, Mapping):
        content = message.get("content")
        if isinstance(content, str) and content.strip():
            return content
    return ""


def _resolve_patch_from_agent_trace(model_output: Mapping[str, Any]) -> str:
    agent_trace = model_output.get("agent_trace")
    if not isinstance(agent_trace, list):
        return ""
    for entry in reversed(agent_trace):
        if not isinstance(entry, Mapping):
            continue
        name = str(entry.get("name") or entry.get("tool") or "")
        if name != "submit_patch_tool":
            continue
        patch = _extract_tool_stdout(entry.get("output"))
        if patch.strip():
            return patch
    return ""


def _extract_tool_stdout(output: Any) -> str:
    if isinstance(output, Mapping):
        for key in ("stdout", "patch_content", "content", "text", "answer", "patch", "diff"):
            value = output.get(key)
            if isinstance(value, str) and value.strip():
                return value
        nested = output.get("output")
        if nested is not None and nested is not output:
            return _extract_tool_stdout(nested)
    if isinstance(output, str):
        return output
    return ""


async def _read_submission_patch(environment: Any) -> str:
    try:
        payload = await environment.read_file("/workspace/submission.patch")
    except Exception:
        return ""
    return _decode_payload(payload)


async def _read_git_diff(environment: Any) -> str:
    for command in (
        "cd /workspace && git diff --binary --no-color",
        "cd /workspace && git diff --binary --no-color HEAD --",
    ):
        try:
            result = await environment.exec(command, timeout_s=10)
        except Exception:
            continue
        if getattr(result, "exit_code", 1) not in (0, None):
            continue
        stdout = _decode_payload(getattr(result, "stdout", ""))
        if "diff --git" in stdout:
            return stdout
    return ""


def _emit_patch_fallback_event(
    trace: Any | None,
    sample: Mapping[str, Any],
    *,
    source: str,
    path: str,
) -> None:
    emit = getattr(trace, "emit", None)
    if not callable(emit):
        return
    instance_id = _resolve_instance_id(sample)
    payload = {"source": source, "path": path}
    if instance_id:
        payload["instance_id"] = instance_id
    emit(
        "swebench_patch_fallback",
        payload,
        sample_id=str(sample.get("id") or instance_id or "sample"),
    )


def _emit_patch_source_event(
    trace: Any | None,
    sample: Mapping[str, Any],
    *,
    source: str,
    path: str | None = None,
) -> None:
    emit = getattr(trace, "emit", None)
    if not callable(emit):
        return
    instance_id = _resolve_instance_id(sample)
    payload = {"source": source}
    if path:
        payload["path"] = path
    if instance_id:
        payload["instance_id"] = instance_id
    emit(
        "patch.source.resolved",
        payload,
        sample_id=str(sample.get("id") or instance_id or "sample"),
    )


_CODE_FENCE_RE = re.compile(r"```(?P<lang>[a-zA-Z0-9_-]*)\s*\n(?P<body>.*?)(?:\n)?```", re.DOTALL)
_UNCLOSED_CODE_FENCE_RE = re.compile(r"```(?P<lang>[a-zA-Z0-9_-]*)\s*\n(?P<body>.*)\Z", re.DOTALL)


def _extract_code_block(text: str) -> str | None:
    matches = list(_CODE_FENCE_RE.finditer(text))
    if matches:
        for match in matches:
            if match.group("lang").strip().lower() in {"diff", "patch"}:
                return match.group("body")
        return matches[0].group("body")
    match = _UNCLOSED_CODE_FENCE_RE.search(text)
    if match:
        return match.group("body")
    return None


def _strip_apply_patch_markers(text: str) -> str:
    lines = text.splitlines()
    begin_idx = None
    end_idx = None
    for idx, line in enumerate(lines):
        if line.strip().startswith("*** Begin Patch"):
            begin_idx = idx
            break
    if begin_idx is not None:
        for idx in range(begin_idx + 1, len(lines)):
            if lines[idx].strip().startswith("*** End Patch"):
                end_idx = idx
                break
    if begin_idx is not None and end_idx is not None and end_idx > begin_idx:
        return "\n".join(lines[begin_idx + 1 : end_idx])
    return "\n".join(
        line
        for line in lines
        if not line.strip().startswith("*** Begin Patch")
        and not line.strip().startswith("*** End Patch")
    )


def _strip_binary_hunks(text: str) -> str:
    lines = text.splitlines()
    output: list[str] = []
    idx = 0
    while idx < len(lines):
        line = lines[idx]
        if line.startswith("diff --git "):
            block = [line]
            idx += 1
            is_binary = False
            while idx < len(lines) and not lines[idx].startswith("diff --git "):
                if lines[idx].startswith(("GIT binary patch", "Binary files ")):
                    is_binary = True
                block.append(lines[idx])
                idx += 1
            if not is_binary:
                output.extend(block)
            continue
        output.append(line)
        idx += 1
    return "\n".join(output)


def _normalize_hunk_context_lines(text: str) -> str:
    lines = text.splitlines()
    output: list[str] = []
    in_hunk = False
    for line in lines:
        if line.startswith("diff --git "):
            in_hunk = False
            output.append(line)
            continue
        if line.startswith("@@"):
            in_hunk = True
            output.append(line)
            continue
        if in_hunk and line and not line.startswith((" ", "+", "-", "\\")):
            output.append(f" {line}")
            continue
        output.append(line)
    return "\n".join(output)


def _has_diff_markers(text: str) -> bool:
    return any(line.startswith(("diff --git ", "--- ", "+++ ", "@@ ")) for line in text.splitlines())


_DIFF_PREFIXES = (
    "diff --git ",
    "index ",
    "--- ",
    "+++ ",
    "@@ ",
    "new file mode ",
    "deleted file mode ",
    "rename from ",
    "rename to ",
    "similarity index ",
    "dissimilarity index ",
    "old mode ",
    "new mode ",
)


def _trim_diff_tail(text: str) -> str:
    lines = text.splitlines()
    last_idx = None
    for idx in range(len(lines) - 1, -1, -1):
        line = lines[idx]
        if line.startswith(_DIFF_PREFIXES) or line[:1] in (" ", "+", "-", "\\"):
            last_idx = idx
            break
    if last_idx is None:
        return text
    return "\n".join(lines[: last_idx + 1])


def _decode_payload(payload: Any) -> str:
    if isinstance(payload, (bytes, bytearray)):
        return payload.decode("utf-8", errors="replace")
    if payload is None:
        return ""
    return str(payload)


def _resolve_instance_id(sample: Mapping[str, Any]) -> str:
    metadata = sample.get("metadata") or {}
    if isinstance(metadata, Mapping):
        value = metadata.get("instance_id")
        if value:
            return str(value)
    return str(sample.get("instance_id") or sample.get("id") or "")
