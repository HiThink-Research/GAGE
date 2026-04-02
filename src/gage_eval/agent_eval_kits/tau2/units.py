"""Shared helpers for the Tau2 benchmark kit."""

from __future__ import annotations

from dataclasses import asdict, is_dataclass
import json
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
    """Resolve the canonical Tau2 sample id."""

    tau2_meta = _tau2_metadata(sample)
    sample_id = sample.get("id") or tau2_meta.get("task_id")
    return str(sample_id or "unknown")


def resolve_instruction(sample: dict) -> str:
    """Build the tau2 instruction shown to the client."""

    for key in ("instruction", "prompt"):
        value = sample.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    explicit_message = _messages_text(sample)
    if explicit_message:
        return explicit_message

    tau2_meta = _tau2_metadata(sample)
    sections: list[str] = []
    user_scenario = tau2_meta.get("user_scenario")
    if user_scenario:
        sections.append(_stringify_block(user_scenario))
    agent_instruction = tau2_meta.get("agent_instruction")
    if agent_instruction:
        sections.append(str(agent_instruction).strip())
    gage_instruction = tau2_meta.get("gage_instruction")
    if gage_instruction:
        sections.append(str(gage_instruction).strip())
    policy = tau2_meta.get("policy")
    if policy:
        sections.append(f"Policy:\n{_stringify_block(policy)}")
    if sections:
        return "\n\n".join(section for section in sections if section).strip()

    raw_assets = sample.get("raw_assets") if isinstance(sample.get("raw_assets"), dict) else {}
    tau2_assets = raw_assets.get("tau2") if isinstance(raw_assets.get("tau2"), dict) else {}
    task_payload = tau2_assets.get("task") if isinstance(tau2_assets.get("task"), dict) else {}
    if task_payload.get("user_scenario"):
        return _stringify_block(task_payload["user_scenario"])
    return ""


def resolve_cwd(sample: dict, session: Any) -> str:
    """Resolve the working directory for Tau2 runs."""

    metadata = sample.get("metadata") if isinstance(sample.get("metadata"), dict) else {}
    tau2_meta = metadata.get("tau2") if isinstance(metadata.get("tau2"), dict) else {}
    resources = getattr(session, "resources", None)
    resource_metadata = getattr(resources, "metadata", {}) if resources is not None else {}
    candidates = (
        sample.get("cwd"),
        sample.get("workspace_root"),
        tau2_meta.get("workspace_root"),
        resource_metadata.get("workspace_root"),
    )
    for value in candidates:
        if isinstance(value, str) and value.strip():
            return value.strip()
    return "."


def resolve_env(sample: dict, session: Any) -> Dict[str, str]:
    """Resolve environment variables for Tau2 runs."""

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


def resolve_runtime_state(resources: Any, scheduler_result: Any) -> Dict[str, Any]:
    """Resolve tau2 runtime state from resources or scheduler output."""

    resource_metadata = dict(getattr(resources, "metadata", {}) or {}) if resources is not None else {}
    runtime_state = resource_metadata.get("runtime_state")
    if isinstance(runtime_state, dict) and runtime_state:
        return dict(runtime_state)
    raw_output = dict(getattr(scheduler_result, "raw_output", {}) or {})
    runtime_state = raw_output.get("runtime_state")
    if isinstance(runtime_state, dict):
        return dict(runtime_state)
    return {}


def _messages_text(sample: dict) -> str:
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


def _tau2_metadata(sample: dict) -> Dict[str, Any]:
    metadata = sample.get("metadata") if isinstance(sample.get("metadata"), dict) else {}
    tau2_meta = metadata.get("tau2")
    if isinstance(tau2_meta, dict):
        return dict(tau2_meta)
    return {}


def _stringify_block(value: Any) -> str:
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(value, ensure_ascii=False, indent=2)
    return str(value)
