"""Terminal benchmark verifier bridge."""

from __future__ import annotations

from dataclasses import asdict
from typing import Any

from gage_eval.agent_eval_kits.terminal_bench.contracts import (
    TERMINAL_BENCH_KIT_ID,
    TERMINAL_BENCH_REQUIRED_SURFACES,
)
from gage_eval.agent_eval_kits.terminal_bench.trace_mapping import normalize_trace_events
from gage_eval.agent_eval_kits.terminal_bench.units import build_task_context, get_sample_id, get_instruction
from gage_eval.agent_runtime.verifier.base import VerifierInput


def _artifact_paths(artifacts: Any) -> dict[str, str]:
    paths: dict[str, str] = {}
    for field_name in ("patch_file", "trajectory_file", "stdout_file", "metadata_file", "verifier_dir"):
        value = getattr(artifacts, field_name, None)
        if value:
            paths[field_name] = str(value)
    return paths


def _scheduler_result_payload(scheduler_result: Any) -> dict[str, Any]:
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


def _trace_payload(sample: dict, scheduler_result: Any) -> tuple[dict[str, Any], ...]:
    sample_payload = sample.get("payload") if isinstance(sample.get("payload"), dict) else {}
    trace_source = sample.get("trace")
    if trace_source is None and isinstance(sample_payload, dict):
        trace_source = sample_payload.get("trace")
    if trace_source is None and isinstance(sample.get("metadata"), dict):
        trace_source = sample["metadata"].get("trace")
    if trace_source is None:
        raw_output = getattr(scheduler_result, "raw_output", {}) or {}
        trace_source = raw_output.get("trace")
        if trace_source is None:
            trace_source = raw_output.get("events")
    return normalize_trace_events(trace_source)


def build_verifier_input(sample: dict, scheduler_result, artifacts):
    """Build a verifier input for the terminal benchmark native verifier."""

    task_context = build_task_context(sample)
    artifact_paths = _artifact_paths(artifacts)
    scheduler_payload = _scheduler_result_payload(scheduler_result)
    payload = {
        "kit_id": TERMINAL_BENCH_KIT_ID,
        "sample": dict(sample),
        "sample_id": get_sample_id(sample),
        "instruction": get_instruction(sample),
        "task_context": asdict(task_context),
        "required_surfaces": TERMINAL_BENCH_REQUIRED_SURFACES,
        "surface_names": TERMINAL_BENCH_REQUIRED_SURFACES,
        "surface_status": {surface: "required" for surface in TERMINAL_BENCH_REQUIRED_SURFACES},
        "scheduler_result": scheduler_payload,
        "trace": _trace_payload(sample, scheduler_result),
        "artifact_paths": artifact_paths,
    }
    metadata = {
        "kit_id": TERMINAL_BENCH_KIT_ID,
        "required_surfaces": TERMINAL_BENCH_REQUIRED_SURFACES,
    }
    return VerifierInput(
        benchmark_kit_id=TERMINAL_BENCH_KIT_ID,
        sample_id=payload["sample_id"],
        payload=payload,
        artifact_paths=artifact_paths,
        metadata=metadata,
    )
