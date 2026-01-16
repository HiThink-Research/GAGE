"""AppWorld summary generator."""

from __future__ import annotations

from typing import Any, Dict, Optional

from gage_eval.evaluation.cache import EvalCache
from gage_eval.registry import registry
from gage_eval.reporting.summary_generators import SummaryGenerator


@registry.asset(
    "summary_generators",
    "appworld_summary",
    desc="AppWorld summary generator",
    tags=("appworld", "report"),
    default_enabled=True,
)
class AppWorldSummaryGenerator(SummaryGenerator):
    def generate(self, cache: EvalCache) -> Optional[Dict[str, Any]]:
        summary = _build_appworld_summary(cache)
        if not summary:
            return None
        return {"appworld_summary": summary}


def _build_appworld_summary(cache: EvalCache) -> Optional[Dict[str, Any]]:
    overall_total = 0
    overall_tgc_sum = 0.0
    overall_tgc_count = 0
    overall_sgc_sum = 0.0
    overall_sgc_count = 0
    by_subset: Dict[str, Dict[str, float]] = {}

    for record in cache.iter_samples():
        if not isinstance(record, dict):
            continue
        sample = record.get("sample") if isinstance(record.get("sample"), dict) else None
        if not sample:
            continue
        metadata = sample.get("metadata") if isinstance(sample.get("metadata"), dict) else {}
        appworld_meta = metadata.get("appworld") if isinstance(metadata.get("appworld"), dict) else {}
        if not appworld_meta:
            continue
        subset = str(appworld_meta.get("subset") or "unknown")
        judge_output = record.get("judge_output") if isinstance(record.get("judge_output"), dict) else {}
        appworld_output = judge_output.get("appworld") if isinstance(judge_output.get("appworld"), dict) else {}

        tgc = _coerce_float(appworld_output.get("tgc"))
        sgc = _coerce_float(appworld_output.get("sgc"))

        overall_total += 1
        if tgc is not None:
            overall_tgc_sum += tgc
            overall_tgc_count += 1
        if sgc is not None:
            overall_sgc_sum += sgc
            overall_sgc_count += 1

        subset_entry = by_subset.setdefault(
            subset,
            {
                "total": 0.0,
                "tgc_sum": 0.0,
                "tgc_count": 0.0,
                "sgc_sum": 0.0,
                "sgc_count": 0.0,
            },
        )
        subset_entry["total"] += 1.0
        if tgc is not None:
            subset_entry["tgc_sum"] += tgc
            subset_entry["tgc_count"] += 1.0
        if sgc is not None:
            subset_entry["sgc_sum"] += sgc
            subset_entry["sgc_count"] += 1.0

    if overall_total == 0:
        return None
    summary: Dict[str, Any] = {
        "overall": {
            "total": overall_total,
            "tgc_mean": (overall_tgc_sum / overall_tgc_count) if overall_tgc_count else 0.0,
            "sgc_mean": (overall_sgc_sum / overall_sgc_count) if overall_sgc_count else 0.0,
        },
        "by_subset": {},
    }
    for subset, stats in by_subset.items():
        tgc_count = stats.get("tgc_count", 0.0) or 0.0
        sgc_count = stats.get("sgc_count", 0.0) or 0.0
        summary["by_subset"][subset] = {
            "total": int(stats.get("total", 0.0) or 0.0),
            "tgc_mean": (stats.get("tgc_sum", 0.0) / tgc_count) if tgc_count else 0.0,
            "sgc_mean": (stats.get("sgc_sum", 0.0) / sgc_count) if sgc_count else 0.0,
        }
    return summary


def _coerce_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


__all__ = ["AppWorldSummaryGenerator"]
