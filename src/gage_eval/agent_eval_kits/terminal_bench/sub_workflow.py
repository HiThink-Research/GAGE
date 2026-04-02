"""Terminal benchmark sub-workflow helpers."""

from __future__ import annotations

from dataclasses import asdict, is_dataclass
from typing import Any

from gage_eval.agent_eval_kits.terminal_bench.contracts import (
    TERMINAL_BENCH_KIT_ID,
    TERMINAL_BENCH_REQUIRED_SURFACES,
)
from gage_eval.agent_eval_kits.terminal_bench.judge_bridge import build_verifier_input
from gage_eval.agent_eval_kits.terminal_bench.resources import build_resource_requirements
from gage_eval.agent_eval_kits.terminal_bench.trace_mapping import normalize_trace_events
from gage_eval.agent_eval_kits.terminal_bench.units import build_task_context, get_sample_id


def _artifact_paths(artifacts: Any) -> dict[str, str]:
    if artifacts is None:
        return {}
    if is_dataclass(artifacts):
        payload = asdict(artifacts)
    elif isinstance(artifacts, dict):
        payload = dict(artifacts)
    else:
        payload = {
            key: value
            for key in dir(artifacts)
            if not key.startswith("_")
            and (value := getattr(artifacts, key, None)) is not None
            and not callable(value)
        }
    return {str(key): str(value) for key, value in payload.items() if value is not None}


def _trace_source(sample: dict, session: Any) -> Any:
    trace_source = sample.get("trace")
    if trace_source is None and isinstance(sample.get("payload"), dict):
        trace_source = sample["payload"].get("trace")
    if trace_source is None and isinstance(sample.get("metadata"), dict):
        trace_source = sample["metadata"].get("trace")
    if trace_source is None and session is not None:
        session_payload = getattr(session, "payload", None)
        if isinstance(session_payload, dict):
            trace_source = session_payload.get("trace")
        if trace_source is None:
            session_metadata = getattr(session, "metadata", None)
            if isinstance(session_metadata, dict):
                trace_source = session_metadata.get("trace")
    return trace_source


def prepare_inputs(sample: dict, session) -> dict:
    """Prepare a terminal benchmark sample for runtime execution."""

    plan = getattr(session, "plan", None)
    task_context = build_task_context(sample, session=session)
    surface_requirements = tuple(task_context.required_surfaces)
    resource_requirements = build_resource_requirements(sample, plan)
    metadata = dict(sample.get("metadata") or {})
    metadata.update(dict(getattr(session, "metadata", {}) or {}))
    metadata.update(
        {
            "benchmark_kit_id": TERMINAL_BENCH_KIT_ID,
            "surface_requirements": surface_requirements,
            "resource_requirements": resource_requirements,
        }
    )
    env = dict(sample.get("env") or {})
    return {
        "kit_id": TERMINAL_BENCH_KIT_ID,
        "sample_id": get_sample_id(sample),
        "instruction": task_context.instruction,
        "cwd": task_context.workspace_root or ".",
        "env": env,
        "task_context": asdict(task_context),
        "surface_requirements": surface_requirements,
        "resource_requirements": resource_requirements,
        "artifact_paths": _artifact_paths(getattr(session, "artifacts", None)),
        "trace": normalize_trace_events(_trace_source(sample, session)),
        "metadata": metadata,
    }


def finalize_result(sample: dict, scheduler_result, artifacts) -> dict:
    """Normalize scheduler results into a terminal benchmark summary."""

    verifier_input = build_verifier_input(sample, scheduler_result, artifacts)
    result_payload = {
        "kit_id": TERMINAL_BENCH_KIT_ID,
        "sample_id": get_sample_id(sample),
        "surface_requirements": TERMINAL_BENCH_REQUIRED_SURFACES,
        "status": getattr(scheduler_result, "status", None),
        "answer": getattr(scheduler_result, "answer", None),
        "patch_path": getattr(scheduler_result, "patch_path", None),
        "stdout_path": getattr(scheduler_result, "stdout_path", None),
        "trajectory_path": getattr(scheduler_result, "trajectory_path", None),
        "artifact_paths": dict(getattr(scheduler_result, "artifacts", {}) or {}),
        "metrics": dict(getattr(scheduler_result, "metrics", {}) or {}),
        "raw_output": dict(getattr(scheduler_result, "raw_output", {}) or {}),
        "verifier_input": verifier_input,
    }
    result_payload["artifact_layout"] = _artifact_paths(artifacts)
    return result_payload
