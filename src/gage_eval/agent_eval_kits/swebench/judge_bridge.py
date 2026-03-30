"""Build verifier inputs for SWE-bench."""

from __future__ import annotations

from typing import Any, Dict

from gage_eval.agent_eval_kits.swebench.units import (
    extract_artifact_paths,
    resolve_model_output,
    resolve_sample_id,
    serialize_scheduler_result,
)
from gage_eval.agent_runtime.verifier.base import VerifierInput


def build_verifier_input(sample: dict, scheduler_result: Any, artifacts: Any) -> VerifierInput:
    """Convert scheduler output into a verifier input."""
    artifact_paths = extract_artifact_paths(artifacts)
    payload: Dict[str, Any] = {
        "sample": sample,
        "model_output": resolve_model_output(scheduler_result),
        "scheduler_result": serialize_scheduler_result(scheduler_result),
        "artifact_paths": artifact_paths,
        "stdout_path": artifact_paths.get("stdout_path"),
        "patch_path": artifact_paths.get("patch_path"),
        "trajectory_path": artifact_paths.get("trajectory_path"),
    }
    metadata = dict(sample.get("metadata") or {})
    metadata.setdefault("benchmark_kit_id", "swebench")
    return VerifierInput(
        benchmark_kit_id="swebench",
        sample_id=resolve_sample_id(sample),
        payload=payload,
        artifact_paths=artifact_paths,
        metadata=metadata,
    )
