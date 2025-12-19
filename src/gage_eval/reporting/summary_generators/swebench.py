"""SWE-bench Pro summary generator."""

from __future__ import annotations

from typing import Any, Dict, Optional

from gage_eval.evaluation.cache import EvalCache
from gage_eval.registry import registry
from gage_eval.reporting.summary_generators import SummaryGenerator


@registry.asset(
    "summary_generators",
    "swebench_summary",
    desc="SWE-bench Pro summary generator",
    tags=("swebench", "report"),
    default_enabled=True,
)
class SwebenchSummaryGenerator(SummaryGenerator):
    def generate(self, cache: EvalCache) -> Optional[Dict[str, Any]]:
        summary = _build_swebench_summary(cache)
        if not summary:
            return None
        return {"swebench_summary": summary}


def _build_swebench_summary(cache: EvalCache) -> Optional[Dict[str, Any]]:
    total = 0
    resolved_total = 0
    by_repo: Dict[str, Dict[str, int]] = {}
    by_language: Dict[str, Dict[str, int]] = {}
    failure_reasons: Dict[str, int] = {}

    for record in cache.iter_samples():
        if not isinstance(record, dict):
            continue
        sample = record.get("sample") if isinstance(record.get("sample"), dict) else None
        if not sample:
            continue
        metadata = sample.get("metadata") if isinstance(sample.get("metadata"), dict) else {}
        if not _is_swebench_sample(sample, metadata):
            continue
        judge_output = record.get("judge_output") if isinstance(record.get("judge_output"), dict) else {}
        resolved = bool((judge_output or {}).get("resolved"))
        failure_reason = (judge_output or {}).get("failure_reason")
        repo = metadata.get("repo") or sample.get("repo")
        language = metadata.get("repo_language") or sample.get("repo_language")

        total += 1
        if resolved:
            resolved_total += 1
        if repo:
            _accumulate_stats(by_repo, str(repo), resolved)
        if language:
            _accumulate_stats(by_language, str(language), resolved)
        if not resolved:
            reason_key = str(failure_reason or "unknown")
            failure_reasons[reason_key] = failure_reasons.get(reason_key, 0) + 1

    if total == 0:
        return None
    return {
        "overall": {
            "total": total,
            "resolved": resolved_total,
            "resolve_rate": resolved_total / total if total else 0.0,
        },
        "by_repo": _finalize_stats(by_repo),
        "by_language": _finalize_stats(by_language),
        "failure_reason": failure_reasons,
    }


def _is_swebench_sample(sample: Dict[str, Any], metadata: Dict[str, Any]) -> bool:
    dataset_id = sample.get("_dataset_id")
    if isinstance(dataset_id, str) and "swebench" in dataset_id.lower():
        return True
    for key in ("instance_id", "base_commit", "test_patch", "fail_to_pass", "pass_to_pass"):
        if metadata.get(key) or sample.get(key):
            return True
    return False


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


__all__ = ["SwebenchSummaryGenerator"]
