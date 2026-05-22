from __future__ import annotations

from copy import deepcopy
from typing import Any


class MetricSummaryCollector:
    """Formats metrics for reports while preserving raw numeric values."""

    def collect(self, metrics: list[dict[str, Any]] | dict[str, Any] | None) -> list[dict[str, Any]]:
        if metrics is None:
            return []
        if isinstance(metrics, dict):
            metrics = metrics.get("metrics", [])
        return [self._collect_one(metric) for metric in metrics]

    def _collect_one(self, metric: dict[str, Any]) -> dict[str, Any]:
        item = dict(metric)
        values = dict(item.get("values") or {})
        raw_values = dict(item.get("raw_values") or values)
        item["raw_values"] = deepcopy(raw_values)
        item["values"] = {key: _format_value(value) for key, value in values.items()}
        item.setdefault("scope", "run")
        item.setdefault("source", "summary")
        return item


def _format_value(value: Any) -> Any:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return f"{float(value):.5f}"
    return value
