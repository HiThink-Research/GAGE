from __future__ import annotations

from typing import Any


def augment_runtime_health_from_tasks(
    runtime_health: dict[str, Any],
    tasks: list[dict[str, Any]],
) -> dict[str, Any]:
    health = dict(runtime_health)
    task_failed_count = 0
    task_aborted_count = 0
    for task in tasks:
        if not isinstance(task, dict):
            continue
        execution = task.get("execution") if isinstance(task.get("execution"), dict) else {}
        status = str(task.get("status") or execution.get("status") or "").lower()
        if status in {"failed", "error", "errored"}:
            task_failed_count += 1
        elif status == "aborted":
            task_aborted_count += 1
    if task_failed_count:
        health["task_failed_count"] = task_failed_count
    if task_aborted_count:
        health["task_aborted_count"] = task_aborted_count
    return health
