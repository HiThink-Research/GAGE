"""Shared helpers for the SWE-bench benchmark kit."""

from __future__ import annotations

from dataclasses import asdict, is_dataclass
from typing import Any, Dict, Mapping, Optional


def resolve_sample_id(sample: Mapping[str, Any]) -> str:
    """Resolve a stable sample identifier."""
    metadata = sample.get("metadata") or {}
    for key in ("sample_id", "instance_id", "id"):
        value = sample.get(key) or metadata.get(key)
        if value is not None and str(value).strip():
            return str(value)
    return "unknown"


def extract_artifact_paths(artifacts: Any) -> Dict[str, str]:
    """Normalize artifact path inputs into a string mapping."""
    if artifacts is None:
        return {}
    if isinstance(artifacts, dict):
        payload = dict(artifacts)
    elif is_dataclass(artifacts):
        payload = asdict(artifacts)
    else:
        payload = {}
        for key in ("patch_file", "trajectory_file", "stdout_file", "metadata_file"):
            value = getattr(artifacts, key, None)
            if value is not None:
                payload[key] = value
    result: Dict[str, str] = {}
    for key, value in payload.items():
        if value is None:
            continue
        result[str(key)] = str(value)
    for source_key, alias_key in (
        ("patch_file", "patch_path"),
        ("trajectory_file", "trajectory_path"),
        ("stdout_file", "stdout_path"),
        ("metadata_file", "metadata_path"),
    ):
        if source_key in result and alias_key not in result:
            result[alias_key] = result[source_key]
    return result


def resolve_model_output(scheduler_result: Any) -> Any:
    """Resolve the model output consumed by the judge."""
    if scheduler_result is None:
        return {}
    if isinstance(scheduler_result, dict):
        if "answer" in scheduler_result:
            return scheduler_result["answer"]
        if "model_output" in scheduler_result:
            return scheduler_result["model_output"]
        if "raw_output" in scheduler_result:
            return scheduler_result["raw_output"]
        return scheduler_result
    for key in ("answer", "model_output", "raw_output"):
        value = getattr(scheduler_result, key, None)
        if value is not None and value != {}:
            return value
    return {}


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


def resolve_cwd(sample: Mapping[str, Any], session: Any) -> str:
    """Resolve the client working directory."""
    for key in ("cwd", "working_dir", "workdir"):
        value = sample.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
        metadata = sample.get("metadata") or {}
        value = metadata.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    metadata = sample.get("metadata") or {}
    for key in ("repo_root", "root", "workspace"):
        value = metadata.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    plan = getattr(session, "plan", None)
    plan_params = getattr(plan, "params", None)
    if isinstance(plan_params, dict):
        for key in ("cwd", "working_dir", "workdir", "repo_root"):
            value = plan_params.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
    return "/app"


def resolve_env(sample: Mapping[str, Any], session: Any) -> Dict[str, str]:
    """Resolve environment variables for the client request."""
    env: Dict[str, str] = {}
    for source in (
        sample.get("env"),
        (sample.get("metadata") or {}).get("env"),
        getattr(getattr(session, "plan", None), "params", {}).get("env") if isinstance(getattr(getattr(session, "plan", None), "params", {}), dict) else None,
    ):
        if isinstance(source, dict):
            for key, value in source.items():
                if value is not None:
                    env[str(key)] = str(value)
    return env
