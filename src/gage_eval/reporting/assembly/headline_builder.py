from __future__ import annotations

from typing import Any


_QUALITY_KEYWORDS = (
    "acc",
    "accuracy",
    "reward",
    "score",
    "resolve_rate",
    "pass_rate",
    "pass_hat",
    "success_rate",
    "f1",
    "win_rate",
)
_OPERATIONAL_KEYWORDS = (
    "latency",
    "cost",
    "token",
    "tokens",
    "duration",
    "wall_runtime",
    "reason",
    "failure_reason",
)
_PREFERRED_VALUE_KEYS = ("score", "mean", "accuracy", "acc", "reward", "rate", "pass_rate", "resolve_rate")


class HeadlineBuilder:
    def build(
        self,
        *,
        metrics: list[dict[str, Any]],
        runtime_health: dict[str, Any],
        attention_cases: list[Any],
        outliers: list[Any],
        failure_clusters: list[Any],
        diagnostics: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        diagnostics = diagnostics or {}
        verdict = self._verdict(runtime_health, diagnostics)
        primary_metric = _pick_primary_metric(metrics)
        score = _primary_score(primary_metric)
        return {
            "verdict": verdict,
            "verdict_reason": _verdict_reason(runtime_health),
            "score": score,
            "primary_metric": primary_metric,
            "key_metric_ids": _key_metric_ids(metrics),
            "top_attention_case_ids": _top_ids(attention_cases, "case_id"),
            "top_failure_cluster_ids": _top_ids(failure_clusters, "cluster_id"),
            "top_outlier_metric_ids": _top_ids(outliers, "metric_id"),
            "warnings": list(diagnostics.get("warnings", [])),
            "errors": list(diagnostics.get("errors", [])),
            "one_line_summary": _one_line_summary(verdict, runtime_health),
            "attention_case_count": len(attention_cases),
            "outlier_count": len(outliers),
            "failure_cluster_count": len(failure_clusters),
        }

    def _verdict(self, runtime_health: dict[str, Any], diagnostics: dict[str, Any]) -> str:
        if diagnostics.get("report_pack_status") == "degraded":
            return "degraded"
        completed = int(runtime_health.get("completed_count") or 0)
        failed = int(runtime_health.get("failed_count") or 0)
        aborted = int(runtime_health.get("aborted_count") or 0)
        task_failed = int(runtime_health.get("task_failed_count") or 0)
        task_aborted = int(runtime_health.get("task_aborted_count") or 0)
        if completed == 0 and (failed > 0 or aborted > 0 or task_failed > 0 or task_aborted > 0):
            return "failed"
        if failed > 0 or aborted > 0 or task_failed > 0 or task_aborted > 0:
            return "passed_with_warnings"
        return "passed"


def _primary_score(metric: dict[str, Any] | None) -> Any:
    if not metric:
        return None
    if "value" in metric:
        return metric.get("value")
    values = metric.get("raw_values") or metric.get("values") or {}
    for key in _PREFERRED_VALUE_KEYS:
        if key in values:
            return values[key]
    return None


def _one_line_summary(verdict: str, runtime_health: dict[str, Any]) -> str:
    completed = runtime_health.get("completed_count", 0)
    if verdict == "passed":
        return f"Run completed with {completed} completed samples."
    if verdict in {"passed_with_warnings", "passed-with-warnings"}:
        return f"Run completed with warnings and {completed} completed samples."
    return f"Run {verdict.replace('_', ' ')} with {completed} completed samples."


def _pick_primary_metric(metrics: list[dict[str, Any]]) -> dict[str, Any] | None:
    usable = [metric for metric in metrics if isinstance(metric, dict) and _has_metric_value(metric)]
    for metric in usable:
        if metric.get("primary") is True:
            return metric

    quality = [
        metric
        for metric in usable
        if _is_quality_metric(metric) and not _is_operational_metric(metric)
    ]
    if quality:
        return quality[0]

    fallback = [metric for metric in usable if not _is_operational_metric(metric)]
    return fallback[0] if fallback else None


def _has_metric_value(metric: dict[str, Any]) -> bool:
    if "value" in metric:
        value = metric.get("value")
        return value not in (None, "")
    values = metric.get("raw_values") if isinstance(metric.get("raw_values"), dict) else metric.get("values")
    if not isinstance(values, dict) or not values:
        return False
    return any(value not in (None, "") for value in values.values())


def _is_quality_metric(metric: dict[str, Any]) -> bool:
    metric_id = str(metric.get("metric_id") or metric.get("id") or metric.get("name") or "").lower()
    return any(keyword in metric_id for keyword in _QUALITY_KEYWORDS)


def _is_operational_metric(metric: dict[str, Any]) -> bool:
    metric_id = str(metric.get("metric_id") or metric.get("id") or metric.get("name") or "").lower()
    return any(keyword in metric_id for keyword in _OPERATIONAL_KEYWORDS)


def _verdict_reason(runtime_health: dict[str, Any]) -> str:
    sample_count = int(runtime_health.get("sample_count") or 0)
    if sample_count <= 0:
        sample_count = sum(
            int(runtime_health.get(key) or 0)
            for key in ("completed_count", "failed_count", "aborted_count")
        )
    completed = int(runtime_health.get("completed_count") or 0)
    parts = [f"{completed}/{sample_count} samples completed"]
    for key, label in (
        ("failed_count", "failed"),
        ("aborted_count", "aborted"),
        ("task_failed_count", "task failed"),
        ("task_aborted_count", "task aborted"),
        ("scheduler_failed_count", "scheduler failure"),
        ("verifier_skipped_count", "verifier skipped"),
    ):
        count = int(runtime_health.get(key) or 0)
        if count:
            suffix = label if count == 1 else f"{label}s"
            parts.append(f"{count} {suffix}")
    return "; ".join(parts)


def _key_metric_ids(metrics: list[dict[str, Any]], *, limit: int = 3) -> list[str]:
    primary = [metric for metric in metrics if metric.get("primary")]
    ordered = [*primary, *[metric for metric in metrics if metric not in primary]]
    return _dedupe(
        str(metric_id)
        for metric in ordered
        if (metric_id := metric.get("metric_id") or metric.get("id"))
    )[:limit]


def _top_ids(items: list[Any], field_name: str, *, limit: int = 3) -> list[str]:
    return _dedupe(
        str(value)
        for item in items
        if (value := _field_value(item, field_name))
    )[:limit]


def _field_value(item: Any, field_name: str) -> Any:
    if isinstance(item, dict):
        return item.get(field_name)
    return getattr(item, field_name, None)


def _dedupe(values: Any) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        result.append(value)
    return result
