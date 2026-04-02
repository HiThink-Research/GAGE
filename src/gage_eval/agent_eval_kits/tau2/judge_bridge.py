"""Build verifier inputs for Tau2."""

from __future__ import annotations

from typing import Any, Dict

from gage_eval.agent_eval_kits.tau2.units import (
    extract_artifact_paths,
    resolve_runtime_state,
    resolve_sample_id,
    serialize_scheduler_result,
)
from gage_eval.agent_runtime.verifier.base import VerifierInput


def build_verifier_input(sample: dict, scheduler_result, artifacts, resources=None) -> VerifierInput:
    """Convert scheduler output into a Tau2 verifier input."""

    artifact_paths = extract_artifact_paths(artifacts)
    runtime_state = resolve_runtime_state(resources, scheduler_result)
    payload: Dict[str, Any] = {
        "sample": sample,
        "scheduler_result": serialize_scheduler_result(scheduler_result),
        "artifact_paths": artifact_paths,
        "runtime_state": runtime_state,
        "stdout_path": artifact_paths.get("stdout_path"),
        "patch_path": artifact_paths.get("patch_path"),
        "trajectory_path": artifact_paths.get("trajectory_path"),
    }
    metadata = dict(sample.get("metadata") or {})
    metadata.setdefault("benchmark_kit_id", "tau2")
    return VerifierInput(
        benchmark_kit_id="tau2",
        sample_id=resolve_sample_id(sample),
        payload=payload,
        artifact_paths=artifact_paths,
        metadata=metadata,
    )
