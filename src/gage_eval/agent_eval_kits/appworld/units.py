"""Shared helpers for the AppWorld benchmark kit."""

from __future__ import annotations

from dataclasses import asdict, is_dataclass
from typing import Any, Dict


def extract_artifact_paths(artifacts: Any) -> Dict[str, str]:
    """Normalize artifact layout objects into a serializable path mapping."""

    if artifacts is None:
        return {}
    if is_dataclass(artifacts):
        payload = asdict(artifacts)
    elif isinstance(artifacts, dict):
        payload = dict(artifacts)
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
                "metadata_file",
            )
            if getattr(artifacts, key, None) is not None
        }
    paths = {str(key): str(value) for key, value in payload.items() if value is not None}
    aliases = (
        ("patch_file", "patch_path"),
        ("trajectory_file", "trajectory_path"),
        ("stdout_file", "stdout_path"),
        ("metadata_file", "metadata_path"),
    )
    for source_key, alias_key in aliases:
        value = paths.get(source_key)
        if value is not None:
            paths.setdefault(alias_key, value)
    return paths


def resolve_sample_id(sample: dict) -> str:
    """Resolve the canonical AppWorld sample id."""

    appworld_meta = _appworld_metadata(sample)
    sample_id = appworld_meta.get("task_id") or sample.get("task_id") or sample.get("id")
    return str(sample_id or "unknown")


def resolve_instruction(sample: dict) -> str:
    """Resolve the instruction text shown to the client."""

    for key in ("instruction", "prompt"):
        value = sample.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    messages = sample.get("messages")
    if not isinstance(messages, list):
        return ""
    parts: list[str] = []
    for message in messages:
        if not isinstance(message, dict):
            continue
        content = message.get("content")
        if isinstance(content, str) and content.strip():
            parts.append(content.strip())
            continue
        if not isinstance(content, list):
            continue
        for item in content:
            if isinstance(item, dict) and isinstance(item.get("text"), str) and item["text"].strip():
                parts.append(item["text"].strip())
    return "\n".join(parts).strip()


def resolve_cwd(sample: dict, session: Any) -> str:
    """Resolve the working directory for AppWorld agent runs."""

    metadata = sample.get("metadata") if isinstance(sample.get("metadata"), dict) else {}
    resources = getattr(session, "resources", None)
    resource_metadata = getattr(resources, "metadata", {}) if resources is not None else {}
    candidates = (
        sample.get("cwd"),
        sample.get("workspace_root"),
        metadata.get("workspace_root"),
        resource_metadata.get("workspace_root"),
    )
    for value in candidates:
        if isinstance(value, str) and value.strip():
            return value.strip()
    return "."


def resolve_env(sample: dict, session: Any) -> Dict[str, str]:
    """Resolve environment variables for AppWorld agent runs."""

    env: Dict[str, str] = {}
    for source in (
        sample.get("env"),
        getattr(session, "metadata", {}).get("env") if session is not None else None,
    ):
        if not isinstance(source, dict):
            continue
        for key, value in source.items():
            if value is None:
                continue
            env[str(key)] = str(value)
    return env


def serialize_scheduler_result(scheduler_result: Any) -> Dict[str, Any]:
    """Serialize scheduler result objects into plain dicts."""

    return {
        "status": getattr(scheduler_result, "status", None),
        "answer": getattr(scheduler_result, "answer", None),
        "patch_path": getattr(scheduler_result, "patch_path", None),
        "stdout_path": getattr(scheduler_result, "stdout_path", None),
        "trajectory_path": getattr(scheduler_result, "trajectory_path", None),
        "artifacts": dict(getattr(scheduler_result, "artifacts", {}) or {}),
        "metrics": dict(getattr(scheduler_result, "metrics", {}) or {}),
        "raw_output": dict(getattr(scheduler_result, "raw_output", {}) or {}),
    }


def resolve_runtime_handle(resources: Any, scheduler_result: Any) -> Dict[str, Any]:
    """Resolve runtime handle metadata from resources or scheduler output."""

    resource_metadata = dict(getattr(resources, "metadata", {}) or {}) if resources is not None else {}
    runtime_handle = resource_metadata.get("runtime_handle")
    if isinstance(runtime_handle, dict) and runtime_handle:
        return dict(runtime_handle)
    raw_output = dict(getattr(scheduler_result, "raw_output", {}) or {})
    runtime_handle = raw_output.get("runtime_handle")
    if isinstance(runtime_handle, dict):
        return dict(runtime_handle)
    return {}


def _appworld_metadata(sample: dict) -> Dict[str, Any]:
    metadata = sample.get("metadata") if isinstance(sample.get("metadata"), dict) else {}
    appworld_meta = metadata.get("appworld")
    if isinstance(appworld_meta, dict):
        return dict(appworld_meta)
    return {}
