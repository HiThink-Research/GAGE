"""Summary generator abstractions."""

from __future__ import annotations

from typing import Any, Iterable, Mapping

from gage_eval.reporting.contracts import SummaryGeneratorResult


class SummaryGenerator:
    """Base class for report-context summary generators."""

    name = "base"
    contract_version = "gage.summary_generator.v2"

    def generate(self, context: Mapping[str, Any]) -> SummaryGeneratorResult | None:  # pragma: no cover - abstract
        raise NotImplementedError


__all__ = ["SummaryGenerator"]


def records_from_context(context: Any) -> list[dict[str, Any]]:
    """Return sample records from a v2 report context, with test-safe legacy tolerance."""
    if isinstance(context, Mapping):
        raw = context.get("samples", [])
    elif hasattr(context, "iter_samples"):
        raw = context.iter_samples()
    else:
        raw = []
    records: list[dict[str, Any]] = []
    for item in raw if isinstance(raw, Iterable) else []:
        if isinstance(item, Mapping):
            records.append(dict(item))
    return records


def section(
    section_id: str,
    title: str,
    *,
    generator_id: str | None = None,
    severity: str = "info",
    metrics: dict[str, Any] | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "section_id": section_id,
        "title": title,
        "severity": severity,
    }
    if generator_id:
        payload["generator_id"] = generator_id
    if metrics:
        payload["metrics"] = dict(metrics)
    return payload
