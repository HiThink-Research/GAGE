"""Derive user-facing progress from Harbor result files."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Mapping

from gage_eval.observability.progress_sink import ProgressSink
from gage_eval.observability.trace import ObservabilityTrace

DEFAULT_PROGRESS_POLL_INTERVAL_S = 5.0
MIN_PROGRESS_POLL_INTERVAL_S = 1.0


@dataclass(frozen=True)
class ExternalHarnessProgress:
    task_id: str
    total_trials: int
    completed_trials: int
    failed_trials: int
    running_trials: int
    current_sample_ids: list[str]
    phase: str = "pending"


def resolve_progress_poll_interval_s(value: Any | None = None) -> float:
    """Resolves a Harbor progress poll interval with a non-busy-loop floor."""

    if value is None:
        return DEFAULT_PROGRESS_POLL_INTERVAL_S
    try:
        interval = float(value)
    except (TypeError, ValueError):
        return DEFAULT_PROGRESS_POLL_INTERVAL_S
    if interval != interval:
        return DEFAULT_PROGRESS_POLL_INTERVAL_S
    return max(interval, MIN_PROGRESS_POLL_INTERVAL_S)


def derive_harbor_progress(
    *,
    jobs_dir: Path | str,
    job_name: str,
    task_id: str,
    total_trials: int | None = None,
    job_config: Mapping[str, Any] | None = None,
    launcher_result: Mapping[str, Any] | None = None,
    launcher_result_path: Path | str | None = None,
) -> ExternalHarnessProgress:
    """Build a GAGE progress snapshot from Harbor job/trial result files.

    Harbor 0.6.6 does not write a stable state file, so this function only
    reads job and trial ``result.json`` files.
    """

    job_dir = Path(jobs_dir) / str(job_name)
    total = _expected_total_trials(total_trials=total_trials, job_config=job_config)
    launcher = _mapping(launcher_result) or _mapping(
        _read_json_if_exists(Path(launcher_result_path)) if launcher_result_path is not None else None
    )
    if _launcher_failed(launcher):
        failed = max(1, total) if total else 1
        return ExternalHarnessProgress(
            task_id=str(task_id),
            total_trials=max(total, failed),
            completed_trials=0,
            failed_trials=failed,
            running_trials=0,
            current_sample_ids=[],
            phase="failed",
        )

    job_result = _read_json_if_exists(job_dir / "result.json")
    if job_result:
        return _progress_from_job_result(
            task_id=str(task_id),
            total_trials=total,
            job_result=job_result,
            job_dir=job_dir,
        )

    return _progress_from_trial_results(
        task_id=str(task_id),
        total_trials=total,
        job_dir=job_dir,
    )


def emit_external_harness_job_submitted(
    trace: ObservabilityTrace,
    *,
    kit_id: str,
    job_name: str,
    total_trials: int,
    dataset_ref: str,
) -> None:
    trace.emit(
        "external_harness_job_submitted",
        {
            "kit_id": str(kit_id),
            "job_name": str(job_name),
            "total_trials": int(total_trials),
            "dataset_ref": str(dataset_ref),
        },
    )


def emit_external_harness_job_completed(
    trace: ObservabilityTrace,
    *,
    job_name: str,
    harbor_job_uuid: str | None,
    exit_code: int,
    total_trials: int | None = None,
    completed_trials: int | None = None,
    failed_trials: int | None = None,
    progress: ExternalHarnessProgress | None = None,
) -> None:
    if progress is not None:
        total_trials = progress.total_trials
        completed_trials = progress.completed_trials
        failed_trials = progress.failed_trials
    trace.emit(
        "external_harness_job_completed",
        {
            "job_name": str(job_name),
            "harbor_job_uuid": str(harbor_job_uuid or ""),
            "exit_code": int(exit_code),
            "total_trials": int(total_trials or 0),
            "completed_trials": int(completed_trials or 0),
            "failed_trials": int(failed_trials or 0),
        },
    )


def emit_external_harness_progress(
    trace: ObservabilityTrace,
    *,
    job_name: str,
    progress: ExternalHarnessProgress,
    phase: str,
    elapsed_s: float,
) -> None:
    trace.emit(
        "external_harness_progress",
        {
            "job_name": str(job_name),
            "completed": int(progress.completed_trials),
            "total": int(progress.total_trials),
            "phase": str(phase),
            "elapsed_s": float(elapsed_s),
        },
    )


def _progress_from_job_result(
    *,
    task_id: str,
    total_trials: int,
    job_result: Mapping[str, Any],
    job_dir: Path,
) -> ExternalHarnessProgress:
    stats = _mapping(job_result.get("stats"))
    total = _non_negative_int(job_result.get("n_total_trials"), default=total_trials)
    completed = _non_negative_int(stats.get("n_completed_trials"), default=0)
    failed = _non_negative_int(stats.get("n_errored_trials"), default=0) + _non_negative_int(
        stats.get("n_cancelled_trials"),
        default=0,
    )
    running = _non_negative_int(stats.get("n_running_trials"), default=0)
    if not running:
        running = max(total - completed - failed, 0)
    phase = "completed" if total and completed + failed >= total else ("running" if running else "pending")
    return ExternalHarnessProgress(
        task_id=task_id,
        total_trials=total,
        completed_trials=completed,
        failed_trials=failed,
        running_trials=running,
        current_sample_ids=_sample_ids_from_trial_results(job_dir),
        phase=phase,
    )


def _progress_from_trial_results(
    *,
    task_id: str,
    total_trials: int,
    job_dir: Path,
) -> ExternalHarnessProgress:
    completed = 0
    failed = 0
    sample_ids: set[str] = set()
    for path in sorted(job_dir.glob("*/result.json")):
        payload = _read_json_if_exists(path)
        if not payload:
            continue
        task_name = payload.get("task_name")
        if task_name:
            sample_ids.add(str(task_name))
        if payload.get("exception_info") or payload.get("verifier_result") is None:
            failed += 1
        else:
            completed += 1
    total = max(total_trials, completed + failed)
    if completed + failed == 0:
        running = 0
        phase = "pending"
    else:
        running = max(total - completed - failed, 0)
    if completed + failed == 0:
        phase = "pending"
    elif running:
        phase = "running"
    elif failed and not completed:
        phase = "failed"
    else:
        phase = "completed"
    return ExternalHarnessProgress(
        task_id=task_id,
        total_trials=total,
        completed_trials=completed,
        failed_trials=failed,
        running_trials=running,
        current_sample_ids=sorted(sample_ids),
        phase=phase,
    )


def _sample_ids_from_trial_results(job_dir: Path) -> list[str]:
    sample_ids: set[str] = set()
    for path in sorted(job_dir.glob("*/result.json")):
        payload = _read_json_if_exists(path)
        if not payload:
            continue
        task_name = payload.get("task_name")
        if task_name:
            sample_ids.add(str(task_name))
    return sorted(sample_ids)


def _expected_total_trials(
    *,
    total_trials: int | None,
    job_config: Mapping[str, Any] | None,
) -> int:
    if total_trials is not None:
        return max(int(total_trials), 0)
    config = _mapping(job_config)
    attempts = _non_negative_int(config.get("n_attempts"), default=1)
    if isinstance(config.get("tasks"), list):
        return attempts * len(config["tasks"])
    if isinstance(config.get("datasets"), list):
        total = 0
        for dataset in config["datasets"]:
            total += _non_negative_int(_mapping(dataset).get("n_tasks"), default=0)
        if total:
            return attempts * total
    return 0


def _launcher_failed(payload: Mapping[str, Any]) -> bool:
    if not payload:
        return False
    return int(payload.get("exit_code") or 0) != 0 or bool(payload.get("launcher_error"))


def _read_json_if_exists(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    try:
        with path.open(encoding="utf-8") as handle:
            payload = json.load(handle)
    except (OSError, json.JSONDecodeError):
        return None
    return payload if isinstance(payload, dict) else None


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _non_negative_int(value: Any, *, default: int) -> int:
    try:
        result = int(value)
    except (TypeError, ValueError):
        return default
    return result if result >= 0 else default


__all__ = [
    "ExternalHarnessProgress",
    "DEFAULT_PROGRESS_POLL_INTERVAL_S",
    "MIN_PROGRESS_POLL_INTERVAL_S",
    "ProgressSink",
    "derive_harbor_progress",
    "emit_external_harness_job_completed",
    "emit_external_harness_progress",
    "emit_external_harness_job_submitted",
    "resolve_progress_poll_interval_s",
]
