"""SWE-bench benchmark-specific workflow helpers."""

from __future__ import annotations

from typing import Any, Dict

from gage_eval.agent_eval_kits.swebench.units import (
    extract_artifact_paths,
    resolve_cwd,
    resolve_env,
    resolve_sample_id,
    serialize_scheduler_result,
)


def prepare_inputs(sample: dict, session: Any) -> Dict[str, Any]:
    """Prepare benchmark-specific inputs for the scheduler."""
    metadata = dict(sample.get("metadata") or {})
    artifact_paths = extract_artifact_paths(getattr(session, "artifacts", None))
    metadata.setdefault("artifact_paths", artifact_paths)
    metadata.setdefault("benchmark_kit_id", getattr(getattr(session, "plan", None), "benchmark_kit_id", "swebench"))
    payload = {
        "sample_id": resolve_sample_id(sample),
        "sample": sample,
        "instruction": _resolve_instruction(sample),
        "cwd": resolve_cwd(sample, session),
        "env": resolve_env(sample, session),
        "metadata": metadata,
        "artifacts": artifact_paths,
        "artifact_paths": artifact_paths,
        "session": {
            "run_id": getattr(getattr(session, "trace", None), "run_id", None),
            "benchmark_kit_id": getattr(getattr(session, "plan", None), "benchmark_kit_id", None),
        },
    }
    return payload


def finalize_result(sample: dict, scheduler_result: Any, artifacts: Any) -> Dict[str, Any]:
    """Post-process scheduler output for reporting."""
    artifact_paths = extract_artifact_paths(artifacts)
    payload = serialize_scheduler_result(scheduler_result)
    if isinstance(payload.get("artifacts"), dict):
        payload["artifacts"] = dict(payload["artifacts"])
    if isinstance(payload.get("metrics"), dict):
        payload["metrics"] = dict(payload["metrics"])
    if isinstance(payload.get("raw_output"), dict):
        payload["raw_output"] = dict(payload["raw_output"])
    payload.update(
        {
            "sample_id": resolve_sample_id(sample),
            "sample_metadata": dict(sample.get("metadata") or {}),
            "artifacts": artifact_paths,
            "artifact_paths": artifact_paths,
            "patch_path": payload.get("patch_path") or artifact_paths.get("patch_path") or artifact_paths.get("patch_file"),
            "stdout_path": payload.get("stdout_path") or artifact_paths.get("stdout_path") or artifact_paths.get("stdout_file"),
            "trajectory_path": payload.get("trajectory_path") or artifact_paths.get("trajectory_path") or artifact_paths.get("trajectory_file"),
        }
    )
    return payload


def _resolve_instruction(sample: dict) -> str:
    if isinstance(sample.get("instruction"), str) and sample["instruction"].strip():
        return sample["instruction"].strip()
    if isinstance(sample.get("prompt"), str) and sample["prompt"].strip():
        return sample["prompt"].strip()
    messages = sample.get("messages")
    if isinstance(messages, list) and messages:
        first = messages[0]
        if isinstance(first, dict):
            content = first.get("content")
            if isinstance(content, str):
                return content.strip()
            if isinstance(content, list):
                parts = []
                for item in content:
                    if isinstance(item, dict) and isinstance(item.get("text"), str):
                        parts.append(item["text"])
                return "\n".join(parts).strip()
    return ""
