"""Tau2 benchmark-specific workflow helpers."""

from __future__ import annotations

from gage_eval.agent_eval_kits.tau2.units import (
    extract_artifact_paths,
    resolve_cwd,
    resolve_env,
    resolve_instruction,
    resolve_sample_id,
    serialize_scheduler_result,
)


def prepare_inputs(sample: dict, session) -> dict:
    """Prepare Tau2 benchmark inputs for the scheduler."""

    metadata = dict(sample.get("metadata") or {})
    artifact_paths = extract_artifact_paths(getattr(session, "artifacts", None))
    metadata.setdefault("artifact_paths", artifact_paths)
    metadata.setdefault(
        "benchmark_kit_id",
        getattr(getattr(session, "plan", None), "benchmark_kit_id", "tau2"),
    )
    return {
        "sample_id": resolve_sample_id(sample),
        "sample": sample,
        "instruction": resolve_instruction(sample),
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


def finalize_result(sample: dict, scheduler_result, artifacts) -> dict:
    """Normalize Tau2 scheduler results for reporting."""

    artifact_paths = extract_artifact_paths(artifacts)
    payload = serialize_scheduler_result(scheduler_result)
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
