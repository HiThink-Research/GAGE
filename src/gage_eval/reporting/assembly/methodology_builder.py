from __future__ import annotations

from typing import Any

from gage_eval.reporting.privacy import SecretFilter


_SECRET_FILTER = SecretFilter()


class MethodologyBuilder:
    def build(
        self,
        *,
        run_metadata: dict[str, Any] | None = None,
        metrics: list[dict[str, Any]] | None = None,
        runtime_health: dict[str, Any] | None = None,
        diagnostics: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        runtime_health = runtime_health or {}
        diagnostics = diagnostics or {}
        caveats: list[str] = []
        if int(runtime_health.get("sample_count") or 0) < 10:
            caveats.append("sample_size.small")
        if diagnostics.get("report_pack_status") == "degraded":
            caveats.append("report_pack.degraded")
        caveats.extend(item.get("code") for item in diagnostics.get("warnings", []) if item.get("code"))
        return {
            "methodology_version": "gage.methodology.v1",
            "run_metadata": _SECRET_FILTER.redact(run_metadata or {}).value,
            "metric_ids": _unique_metric_ids(metrics or []),
            "sample_count": runtime_health.get("sample_count", 0),
            "caveats": caveats,
        }


def _unique_metric_ids(metrics: list[dict[str, Any]]) -> list[str]:
    values: list[str] = []
    seen: set[str] = set()
    for metric in metrics:
        if not isinstance(metric, dict):
            continue
        metric_id = str(metric.get("metric_id") or "").strip()
        if not metric_id or metric_id in seen:
            continue
        values.append(metric_id)
        seen.add(metric_id)
    return values
