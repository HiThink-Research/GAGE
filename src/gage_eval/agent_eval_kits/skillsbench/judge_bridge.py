"""Build verifier inputs for SkillsBench."""

from __future__ import annotations

from typing import Any, Dict

from gage_eval.agent_eval_kits.skillsbench.units import (
    extract_artifact_paths,
    resolve_sample_id,
    serialize_scheduler_result,
)
from gage_eval.agent_runtime.verifier.base import VerifierInput


def build_verifier_input(sample: dict, scheduler_result: Any, artifacts: Any) -> VerifierInput:
    """Convert scheduler output into a verifier input."""

    artifact_paths = extract_artifact_paths(artifacts)
    scheduler_payload = serialize_scheduler_result(scheduler_result)
    scheduler_artifacts = scheduler_payload.get("artifacts") if isinstance(scheduler_payload.get("artifacts"), dict) else {}
    merged_artifacts = dict(artifact_paths)
    for key, value in dict(scheduler_artifacts or {}).items():
        if value is not None:
            merged_artifacts.setdefault(str(key), str(value))
    payload: Dict[str, Any] = {
        "sample": sample,
        "scheduler_result": scheduler_payload,
        "artifact_paths": merged_artifacts,
        "stdout_path": merged_artifacts.get("stdout_path"),
        "patch_path": merged_artifacts.get("patch_path"),
        "trajectory_path": merged_artifacts.get("trajectory_path"),
        "agent_workspace_dir": merged_artifacts.get("agent_workspace_dir"),
    }
    metadata = dict(sample.get("metadata") or {})
    metadata.setdefault("benchmark_kit_id", "skillsbench")
    return VerifierInput(
        benchmark_kit_id="skillsbench",
        sample_id=resolve_sample_id(sample),
        payload=payload,
        artifact_paths=merged_artifacts,
        metadata=metadata,
    )
