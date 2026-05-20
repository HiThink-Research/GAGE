from __future__ import annotations

from typing import Any


def merge_metric_entries(
    metrics: list[dict[str, Any]],
    additional: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    merged = list(metrics)
    seen = {
        str(metric.get("metric_id") or metric.get("id") or metric.get("name"))
        for metric in merged
        if isinstance(metric, dict)
    }
    for metric in additional:
        metric_id = str(metric.get("metric_id") or metric.get("id") or metric.get("name"))
        if metric_id in seen:
            continue
        merged.append(metric)
        seen.add(metric_id)
    return merged


def external_harness_metric_entries(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    harbor_scores: list[float] = []
    harbor_resolves: list[float] = []
    for record in records:
        if not isinstance(record, dict) or not is_harbor_record(record):
            continue
        eval_result = harbor_eval_result(record)
        score = first_float(
            eval_result.get("harbor_score_mean"),
            eval_result.get("score_mean"),
            eval_result.get("score"),
            eval_result.get("reward"),
        )
        if score is None:
            score = trial_score_mean(record)
        if score is not None:
            harbor_scores.append(score)
        resolve_rate = first_float(
            eval_result.get("harbor_resolve_rate"),
            eval_result.get("resolve_rate"),
            eval_result.get("pass_rate"),
        )
        if resolve_rate is None:
            resolved = first_bool(eval_result.get("resolved"), eval_result.get("passed"))
            if resolved is not None:
                resolve_rate = 1.0 if resolved else 0.0
        if resolve_rate is None:
            resolve_rate = trial_resolve_rate(record)
        if resolve_rate is not None:
            harbor_resolves.append(resolve_rate)

    entries: list[dict[str, Any]] = []
    if harbor_scores:
        entries.append(
            {
                "metric_id": "harbor_score_mean",
                "name": "Harbor score mean",
                "values": {"mean": sum(harbor_scores) / len(harbor_scores)},
                "scope": "run",
                "source": "external_harness.harbor",
                "unit": "score",
                "primary": True,
            }
        )
    if harbor_resolves:
        entries.append(
            {
                "metric_id": "harbor_resolve_rate",
                "name": "Harbor resolve rate",
                "values": {"rate": sum(harbor_resolves) / len(harbor_resolves)},
                "scope": "run",
                "source": "external_harness.harbor",
                "unit": "ratio",
            }
        )
    return entries


def external_harness_task_metric_entries(records: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    by_task: dict[str, list[dict[str, Any]]] = {}
    for record in records:
        if not isinstance(record, dict) or not is_harbor_record(record):
            continue
        task_id = record_task_id(record)
        if not task_id:
            continue
        by_task.setdefault(task_id, []).append(record)
    metrics_by_task: dict[str, list[dict[str, Any]]] = {}
    for task_id, task_records in by_task.items():
        task_metrics = []
        for metric in external_harness_metric_entries(task_records):
            item = dict(metric)
            item["scope"] = "task"
            item["task_id"] = task_id
            item.pop("primary", None)
            task_metrics.append(item)
        if task_metrics:
            metrics_by_task[task_id] = task_metrics
    return metrics_by_task


def is_harbor_record(record: dict[str, Any]) -> bool:
    sample = record.get("sample") if isinstance(record.get("sample"), dict) else {}
    metadata = sample.get("metadata") if isinstance(sample.get("metadata"), dict) else {}
    harness = metadata.get("_harness") if isinstance(metadata.get("_harness"), dict) else {}
    task_type = str(sample.get("task_type") or record.get("task_type") or "")
    eval_result = harbor_eval_result(record)
    return (
        task_type == "external_harness.harbor"
        or harness.get("kit_id") == "harbor"
        or any(key in eval_result for key in ("harbor_score_mean", "harbor_resolve_rate", "external_trial_pass_values"))
    )


def harbor_eval_result(record: dict[str, Any]) -> dict[str, Any]:
    sample = record.get("sample") if isinstance(record.get("sample"), dict) else {}
    eval_result = sample.get("eval_result") if isinstance(sample.get("eval_result"), dict) else {}
    if eval_result:
        return dict(eval_result)
    judge_output = record.get("judge_output") if isinstance(record.get("judge_output"), dict) else {}
    return dict(judge_output)


def record_task_id(record: dict[str, Any]) -> str | None:
    namespace = str(record.get("namespace") or "")
    if namespace in {"task_global", "task_root"}:
        return None
    if namespace.startswith("task/"):
        return namespace.removeprefix("task/")
    if namespace.startswith("task_"):
        return namespace.removeprefix("task_")
    sample = record.get("sample") if isinstance(record.get("sample"), dict) else {}
    metadata = sample.get("metadata") if isinstance(sample.get("metadata"), dict) else {}
    value = record.get("task_id") or metadata.get("task_id")
    return str(value) if value else None


def trial_score_mean(record: dict[str, Any]) -> float | None:
    scores: list[float] = []
    for trial in record.get("trial_results") or []:
        if not isinstance(trial, dict):
            continue
        verifier = trial.get("verifier_result") if isinstance(trial.get("verifier_result"), dict) else {}
        score = float_or_none(verifier.get("score"))
        if score is None:
            score = float_or_none(verifier.get("reward"))
        if score is not None:
            scores.append(score)
    return (sum(scores) / len(scores)) if scores else None


def trial_resolve_rate(record: dict[str, Any]) -> float | None:
    values: list[bool] = []
    eval_result = harbor_eval_result(record)
    pass_values = eval_result.get("external_trial_pass_values")
    if isinstance(pass_values, list):
        values.extend(value for value in pass_values if isinstance(value, bool))
    for trial in record.get("trial_results") or []:
        if not isinstance(trial, dict):
            continue
        verifier = trial.get("verifier_result") if isinstance(trial.get("verifier_result"), dict) else {}
        value = verifier.get("resolved")
        if not isinstance(value, bool):
            value = verifier.get("passed")
        if isinstance(value, bool):
            values.append(value)
    return (sum(1 for value in values if value) / len(values)) if values else None


def first_float(*values: Any) -> float | None:
    for value in values:
        result = float_or_none(value)
        if result is not None:
            return result
    return None


def first_bool(*values: Any) -> bool | None:
    for value in values:
        result = bool_or_none(value)
        if result is not None:
            return result
    return None


def bool_or_none(value: Any) -> bool | None:
    if isinstance(value, dict):
        return bool_or_none(value.get("value"))
    if isinstance(value, bool):
        return value
    return None


def float_or_none(value: Any) -> float | None:
    if isinstance(value, dict):
        return float_or_none(value.get("value"))
    if value is None or isinstance(value, bool):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None
