"""Shared helpers for the SkillsBench benchmark kit."""

from __future__ import annotations

from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Dict, Mapping, Optional


def resolve_sample_id(sample: Mapping[str, Any]) -> str:
    """Resolve a stable sample identifier."""

    metadata = sample.get("metadata") or {}
    for key in ("sample_id", "instance_id", "id"):
        value = sample.get(key) or metadata.get(key)
        if value is not None and str(value).strip():
            return str(value)
    skillsbench_meta = _skillsbench_meta(sample)
    task_id = skillsbench_meta.get("task_id")
    return str(task_id or "unknown")


def extract_artifact_paths(artifacts: Any) -> Dict[str, str]:
    """Normalize artifact layout inputs into a string mapping."""

    if artifacts is None:
        return {}
    if isinstance(artifacts, dict):
        payload = dict(artifacts)
    elif is_dataclass(artifacts):
        payload = asdict(artifacts)
    else:
        payload = {
            key: getattr(artifacts, key)
            for key in (
                "run_dir",
                "sample_dir",
                "agent_dir",
                "verifier_dir",
                "patch_file",
                "trajectory_file",
                "stdout_file",
                "stderr_file",
                "metadata_file",
                "verifier_result_file",
                "verifier_stdout_file",
                "verifier_stderr_file",
            )
            if getattr(artifacts, key, None) is not None
        }
    result = {str(key): str(value) for key, value in payload.items() if value is not None}
    for source_key, alias_key in (
        ("patch_file", "patch_path"),
        ("trajectory_file", "trajectory_path"),
        ("stdout_file", "stdout_path"),
        ("stderr_file", "stderr_path"),
        ("metadata_file", "metadata_path"),
        ("verifier_result_file", "verifier_result_path"),
    ):
        if source_key in result and alias_key not in result:
            result[alias_key] = result[source_key]
    return result


def resolve_instruction(sample: Mapping[str, Any]) -> str:
    """Resolve the user instruction shown to the client."""

    for key in ("instruction", "prompt"):
        value = sample.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    messages = sample.get("messages")
    if isinstance(messages, list):
        parts: list[str] = []
        for message in messages:
            if not isinstance(message, dict):
                continue
            content = message.get("content")
            if isinstance(content, str) and content.strip():
                parts.append(content.strip())
                continue
            if isinstance(content, list):
                for item in content:
                    if isinstance(item, dict) and isinstance(item.get("text"), str) and item["text"].strip():
                        parts.append(item["text"].strip())
        if parts:
            return "\n".join(parts).strip()
    return ""


def resolve_cwd(sample: Mapping[str, Any], session: Any) -> str:
    """Resolve the client working directory."""

    for key in ("cwd", "workspace_root", "workdir", "working_dir"):
        value = sample.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    metadata = sample.get("metadata") if isinstance(sample.get("metadata"), dict) else {}
    for key in ("cwd", "workspace_root", "workdir", "working_dir"):
        value = metadata.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    skillsbench_meta = _skillsbench_meta(sample)
    for key in ("workdir", "workspace_root"):
        value = skillsbench_meta.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    resource_metadata = getattr(getattr(session, "resources", None), "metadata", {}) if session is not None else {}
    workspace_root = resource_metadata.get("workspace_root") if isinstance(resource_metadata, dict) else None
    if isinstance(workspace_root, str) and workspace_root.strip():
        return workspace_root.strip()
    return "/app"


def resolve_env(sample: Mapping[str, Any], session: Any) -> Dict[str, str]:
    """Resolve runtime env for the client request."""

    env: Dict[str, str] = {}
    for source in (
        sample.get("env"),
        (sample.get("metadata") or {}).get("env"),
        getattr(getattr(session, "plan", None), "params", {}).get("env")
        if isinstance(getattr(getattr(session, "plan", None), "params", {}), dict)
        else None,
    ):
        if not isinstance(source, dict):
            continue
        for key, value in source.items():
            if value is not None:
                env[str(key)] = str(value)
    return env


def serialize_scheduler_result(scheduler_result: Any) -> Dict[str, Any]:
    """Convert a scheduler result into a JSON-friendly mapping."""

    if scheduler_result is None:
        return {}
    if isinstance(scheduler_result, dict):
        return dict(scheduler_result)
    payload: Dict[str, Any] = {}
    for key in ("status", "answer", "patch_path", "stdout_path", "trajectory_path", "artifacts", "metrics", "raw_output"):
        value = getattr(scheduler_result, key, None)
        if value is not None:
            payload[key] = value
    return payload


def resolve_skillsbench_meta(sample: Mapping[str, Any]) -> Dict[str, Any]:
    """Resolve nested SkillsBench metadata."""

    return _skillsbench_meta(sample)


def resolve_agent_workspace_dir(artifacts: Any) -> Optional[Path]:
    """Return the canonical agent workspace directory when present."""

    paths = extract_artifact_paths(artifacts)
    agent_dir = paths.get("agent_dir")
    if not agent_dir:
        return None
    return Path(agent_dir) / "workspace"


def _skillsbench_meta(sample: Mapping[str, Any]) -> Dict[str, Any]:
    metadata = sample.get("metadata")
    if not isinstance(metadata, dict):
        return {}
    skillsbench_meta = metadata.get("skillsbench")
    if not isinstance(skillsbench_meta, dict):
        return {}
    return dict(skillsbench_meta)


__all__ = [
    "extract_artifact_paths",
    "resolve_agent_workspace_dir",
    "resolve_cwd",
    "resolve_env",
    "resolve_instruction",
    "resolve_sample_id",
    "resolve_skillsbench_meta",
    "serialize_scheduler_result",
]
