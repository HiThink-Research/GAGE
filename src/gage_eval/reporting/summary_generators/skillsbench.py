"""SkillsBench summary generator."""

from __future__ import annotations

from typing import Any, Dict, Optional

from gage_eval.evaluation.cache import EvalCache
from gage_eval.evaluation.sample_envelope import resolve_judge_output
from gage_eval.registry import registry
from gage_eval.reporting.summary_generators import SummaryGenerator


@registry.asset(
    "summary_generators",
    "skillsbench_summary",
    desc="SkillsBench summary generator",
    tags=("skillsbench", "report"),
    default_enabled=True,
)
class SkillsBenchSummaryGenerator(SummaryGenerator):
    def generate(self, cache: EvalCache) -> Optional[Dict[str, Any]]:
        summary = _build_skillsbench_summary(cache)
        if not summary:
            return None
        return {"skillsbench_summary": summary}


def _build_skillsbench_summary(cache: EvalCache) -> Optional[Dict[str, Any]]:
    total = 0
    resolved_total = 0
    by_category: Dict[str, Dict[str, int]] = {}
    by_difficulty: Dict[str, Dict[str, int]] = {}
    failure_reasons: Dict[str, int] = {}

    for record in cache.iter_samples():
        if not isinstance(record, dict):
            continue
        sample = record.get("sample") if isinstance(record.get("sample"), dict) else None
        if not sample:
            continue
        metadata = sample.get("metadata") if isinstance(sample.get("metadata"), dict) else {}
        skillsbench_meta = metadata.get("skillsbench") if isinstance(metadata.get("skillsbench"), dict) else {}
        if not _is_skillsbench_sample(sample, metadata, skillsbench_meta):
            continue
        judge_output = resolve_judge_output(
            sample,
            record.get("judge_output") if isinstance(record.get("judge_output"), dict) else {},
        )
        resolved = _is_resolved(judge_output)
        failure_reason = _failure_reason(judge_output)
        category = skillsbench_meta.get("category")
        difficulty = skillsbench_meta.get("difficulty")

        total += 1
        if resolved:
            resolved_total += 1
        if category:
            _accumulate_stats(by_category, str(category), resolved)
        if difficulty:
            _accumulate_stats(by_difficulty, str(difficulty), resolved)
        if not resolved:
            key = str(failure_reason or "unknown")
            failure_reasons[key] = failure_reasons.get(key, 0) + 1

    if total == 0:
        return None
    return {
        "overall": {
            "total": total,
            "resolved": resolved_total,
            "resolve_rate": resolved_total / total if total else 0.0,
        },
        "by_category": _finalize_stats(by_category),
        "by_difficulty": _finalize_stats(by_difficulty),
        "failure_reason": failure_reasons,
    }


def _is_skillsbench_sample(
    sample: Dict[str, Any],
    metadata: Dict[str, Any],
    skillsbench_meta: Dict[str, Any],
) -> bool:
    dataset_id = sample.get("_dataset_id")
    if isinstance(dataset_id, str) and "skillsbench" in dataset_id.lower():
        return True
    return bool(skillsbench_meta) or metadata.get("benchmark_kit_id") == "skillsbench"


def _is_resolved(judge_output: Dict[str, Any]) -> bool:
    if bool(judge_output.get("resolved")):
        return True
    status = str(judge_output.get("status") or "").strip().lower()
    if status in {"pass", "passed", "resolved", "success"}:
        return True
    raw_output = judge_output.get("raw_output") if isinstance(judge_output.get("raw_output"), dict) else {}
    return bool(raw_output.get("resolved"))


def _failure_reason(judge_output: Dict[str, Any]) -> Optional[str]:
    value = judge_output.get("failure_reason")
    if value:
        return str(value)
    raw_output = judge_output.get("raw_output") if isinstance(judge_output.get("raw_output"), dict) else {}
    value = raw_output.get("failure_reason")
    if value:
        return str(value)
    return None


def _accumulate_stats(bucket: Dict[str, Dict[str, int]], key: str, resolved: bool) -> None:
    stats = bucket.setdefault(key, {"total": 0, "resolved": 0})
    stats["total"] += 1
    if resolved:
        stats["resolved"] += 1


def _finalize_stats(bucket: Dict[str, Dict[str, int]]) -> Dict[str, Dict[str, float]]:
    finalized: Dict[str, Dict[str, float]] = {}
    for key, stats in bucket.items():
        total = stats.get("total", 0) or 0
        resolved = stats.get("resolved", 0) or 0
        finalized[key] = {
            "total": float(total),
            "resolved": float(resolved),
            "resolve_rate": (resolved / total) if total else 0.0,
        }
    return finalized


__all__ = ["SkillsBenchSummaryGenerator", "_build_skillsbench_summary"]
