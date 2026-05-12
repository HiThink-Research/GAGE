"""AppWorld verifier scoring helpers."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Mapping


def normalize_appworld_payload(
    *,
    sample: Mapping[str, Any],
    scheduler_result: Mapping[str, Any],
    runtime_context: Mapping[str, Any],
) -> dict[str, Any]:
    """Resolve the best AppWorld verifier payload from runtime-owned outputs."""

    for candidate in (
        scheduler_result.get("appworld"),
        scheduler_result.get("appworld_output"),
        scheduler_result.get("saved_payload"),
        runtime_context.get("appworld"),
    ):
        if isinstance(candidate, Mapping):
            return _normalize_eval_payload(dict(candidate), task_id=_resolve_task_id(sample))

    appworld_save = runtime_context.get("appworld_save")
    if isinstance(appworld_save, Mapping):
        output = appworld_save.get("output") or appworld_save.get("outputs")
        if isinstance(output, Mapping):
            return _normalize_eval_payload(dict(output), task_id=_resolve_task_id(sample))

    return _build_error_payload(_resolve_task_id(sample), None, "default", "missing_appworld_success_signal")


def build_appworld_diagnostics(appworld_output: Mapping[str, Any]) -> tuple[str, dict[str, Any]]:
    """Build stable failure diagnostics for AppWorld verifier output."""

    if appworld_output.get("status") == "error":
        reason = str(appworld_output.get("failure_reason") or "appworld_error")
        return reason, {"appworld": dict(appworld_output)}

    tests = appworld_output.get("tests")
    if isinstance(tests, Mapping):
        failures = tests.get("fails") or tests.get("failures") or []
        if isinstance(failures, list) and failures:
            return (
                "verifier_assertion_failed",
                {"verifier_failures": [_summarize_failure(item) for item in failures if isinstance(item, Mapping)]},
            )

    tgc = _coerce_float(appworld_output.get("tgc"))
    if tgc is not None and tgc < 1.0:
        return "task_incomplete", {"tgc": tgc}

    return "appworld_evaluation_completed", {}


def resolve_output_dir(
    *,
    appworld_root: str,
    experiment_name: str,
    task_id: str,
    output_dir_template: str | None,
) -> str:
    template = output_dir_template or "{root}/experiments/outputs/{experiment_name}/tasks/{task_id}"
    return template.format(root=appworld_root.rstrip("/"), experiment_name=experiment_name, task_id=task_id)


def resolve_export_dir(
    *,
    base_dir: str | None,
    payload: Mapping[str, Any],
    task_id: str,
    experiment_name: str,
    default_enabled: bool,
) -> str | None:
    if not default_enabled and not base_dir:
        return None
    root = Path(base_dir or os.environ.get("GAGE_EVAL_SAVE_DIR") or ".").expanduser()
    trace = payload.get("trace")
    run_id = str(getattr(trace, "run_id", "") or "run")
    return str(root / run_id / "appworld_artifacts")


def _normalize_eval_payload(payload: Mapping[str, Any], *, task_id: str) -> dict[str, Any]:
    aggregate = payload.get("aggregate") if isinstance(payload.get("aggregate"), Mapping) else {}
    individual = payload.get("individual") if isinstance(payload.get("individual"), Mapping) else {}
    task_payload = individual.get(task_id) if isinstance(individual.get(task_id), Mapping) else {}
    tests = payload.get("tests") if isinstance(payload.get("tests"), Mapping) else {}
    passes = tests.get("passes") or task_payload.get("passes") or []
    fails = tests.get("fails") or tests.get("failures") or task_payload.get("failures") or []
    normalized = {
        "task_id": task_id,
        "status": str(payload.get("status") or "completed"),
        "tgc": payload.get("tgc", aggregate.get("task_goal_completion")),
        "sgc": payload.get("sgc", aggregate.get("scenario_goal_completion")),
        "tests": {"passes": list(passes or []), "fails": list(fails or [])},
    }
    if task_payload.get("difficulty") is not None:
        normalized["difficulty"] = task_payload.get("difficulty")
    if payload.get("failure_reason") is not None:
        normalized["failure_reason"] = str(payload.get("failure_reason"))
    return normalized


def _build_error_payload(
    task_id: str,
    subset: str | None,
    experiment_name: str,
    reason: str,
) -> dict[str, Any]:
    return {
        "task_id": task_id,
        "subset": subset,
        "experiment_name": experiment_name,
        "status": "error",
        "failure_reason": reason,
    }


def _resolve_task_id(sample: Mapping[str, Any]) -> str:
    metadata = sample.get("metadata") if isinstance(sample.get("metadata"), Mapping) else {}
    appworld = metadata.get("appworld") if isinstance(metadata.get("appworld"), Mapping) else {}
    return str(appworld.get("task_id") or sample.get("task_id") or sample.get("id") or "appworld-sample")


def _summarize_failure(item: Mapping[str, Any]) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    for key in ("label", "requirement"):
        if item.get(key) is not None:
            summary[key] = item.get(key)
    trace = item.get("trace")
    if trace is not None:
        summary["trace_excerpt"] = str(trace)[:1000]
    return summary


def _coerce_float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None
