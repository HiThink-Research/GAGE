"""CLI intent types shared by config loading and smart defaults."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class CLIIntent:
    """Read-only CLI options that influence config materialization."""

    backend_id: str | None = None
    max_samples: int | None = None
    skip_judge: bool = False
    metric_ids: tuple[str, ...] | None = None


def parse_metric_ids_csv(metric_ids_csv: str | None) -> tuple[str, ...] | None:
    """Parse a comma-separated metric id list from CLI input."""

    if not metric_ids_csv:
        return None
    parsed = tuple(item.strip() for item in metric_ids_csv.split(",") if item.strip())
    return parsed or None


def apply_cli_final_overrides(payload: dict[str, Any], intent: CLIIntent) -> None:
    """Apply CLI overrides to an already materialized PipelineConfig payload."""

    if intent.max_samples is not None:
        _apply_max_samples(payload, intent.max_samples)
    if intent.metric_ids:
        _apply_metric_filter(payload, set(intent.metric_ids))
    if intent.skip_judge:
        _apply_skip_judge(payload)


def _apply_max_samples(payload: dict[str, Any], max_samples: int) -> None:
    for task in payload.get("tasks") or []:
        if isinstance(task, dict):
            task["max_samples"] = max_samples

    for dataset in payload.get("datasets") or []:
        if not isinstance(dataset, dict):
            continue
        params = dataset.get("params")
        if isinstance(params, dict):
            params["limit"] = max_samples
        hub_params = dataset.get("hub_params")
        if isinstance(hub_params, dict):
            hub_params["limit"] = max_samples


def _apply_metric_filter(payload: dict[str, Any], wanted: set[str]) -> None:
    metrics = payload.get("metrics")
    if metrics is None:
        return
    if not isinstance(metrics, list):
        return

    filtered = []
    for metric in metrics:
        metric_id = _metric_id_from_entry(metric)
        if metric_id is None or metric_id in wanted:
            filtered.append(metric)
    payload["metrics"] = filtered


def _metric_id_from_entry(entry: Any) -> str | None:
    """Extract a metric id from documented PipelineConfig metric entry forms."""

    if isinstance(entry, str):
        metric_id = entry.split("(", 1)[0].strip()
        return metric_id or None

    if isinstance(entry, dict):
        if len(entry) == 1 and "metric_id" not in entry and "implementation" not in entry:
            metric_id = next(iter(entry.keys()))
            return str(metric_id) if metric_id else None

        # Compatibility fallback: single-key {"implementation": "..."} resolves
        # to implementation, but this is not a documented CLI filter shortcut.
        metric_id = entry.get("metric_id") or entry.get("implementation")
        return str(metric_id) if metric_id else None

    return None


def _apply_skip_judge(payload: dict[str, Any]) -> None:
    custom = payload.get("custom")
    if isinstance(custom, dict) and "steps" in custom:
        custom["steps"] = _remove_judge_steps(custom.get("steps"))

    for task in payload.get("tasks") or []:
        if isinstance(task, dict) and "steps" in task:
            task["steps"] = _remove_judge_steps(task.get("steps"))


def _remove_judge_steps(steps: Any) -> Any:
    if not isinstance(steps, list):
        return steps
    return [step for step in steps if not (isinstance(step, dict) and step.get("step") == "judge")]
