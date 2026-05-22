"""SWE-bench Pro summary generator."""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Dict, Iterable, Mapping, Optional

from gage_eval.registry import registry
from gage_eval.reporting.assembly.attention_detector import AttentionCaseDetector
from gage_eval.reporting.contracts import SummaryGeneratorResult
from gage_eval.reporting.summary_generators import SummaryGenerator
from gage_eval.reporting.summary_generators._reason_codes import (
    extract_attention_reason_codes,
    first_agentkit_trial_result,
)
from gage_eval.reporting.summary_generators.base import records_from_context, section


@registry.asset(
    "summary_generators",
    "swebench_summary",
    desc="SWE-bench Pro summary generator",
    tags=("swebench", "report"),
    default_enabled=True,
)
class SwebenchSummaryGenerator(SummaryGenerator):
    def generate(self, context: Any) -> SummaryGeneratorResult | None:
        records = records_from_context(context)
        summary = _build_swebench_summary(records)
        if not summary:
            return None
        attention_cases = _build_swebench_attention_cases(records)
        return SummaryGeneratorResult(
            generator_id="swebench_summary",
            summary_sections=[section("overview", "SWE-bench Summary", generator_id="swebench_summary")],
            attention_cases=attention_cases,
            legacy_payload={"swebench_summary": summary},
        )


def _build_swebench_summary(records: Iterable[dict[str, Any]]) -> Optional[Dict[str, Any]]:
    total = 0
    resolved_total = 0
    by_repo: Dict[str, Dict[str, int]] = {}
    by_language: Dict[str, Dict[str, int]] = {}
    failure_reasons: Dict[str, int] = {}

    for record in records:
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


def _build_swebench_attention_cases(records: list[dict[str, Any]]) -> list[Any]:
    swebench_records = [record for record in records if _record_is_swebench(record)]
    total_samples = max(len(swebench_records), 1)
    candidates: list[dict[str, Any]] = []
    reason_samples: dict[str, set[str]] = defaultdict(set)

    for record in swebench_records:
        judge_output = record.get("judge_output") if isinstance(record.get("judge_output"), Mapping) else {}
        if bool(judge_output.get("resolved")):
            continue
        sample_id = _sample_id(record)
        trial = _trial_payload(record)
        reason_codes = extract_attention_reason_codes(record, trial=trial)
        primary_reason = reason_codes[0]
        reason_samples[primary_reason].add(sample_id)
        candidates.append(
            {
                "case_id": f"swebench/{sample_id or 'unresolved'}",
                "reason_codes": reason_codes,
                "summary": f"SWE-bench sample is unresolved: {_humanize_reason_codes(reason_codes)}.",
                "evidence_ref_ids": [],
                "sample_id": sample_id,
                "trial_id": _trial_id(record, trial),
                "_primary_reason": primary_reason,
            }
        )

    for candidate in candidates:
        frequency = len(reason_samples[candidate.pop("_primary_reason")]) / total_samples
        candidate["frequency"] = frequency
    return AttentionCaseDetector().detect(candidates, total_samples=total_samples)


def _record_is_swebench(record: Mapping[str, Any]) -> bool:
    sample = record.get("sample") if isinstance(record.get("sample"), Mapping) else {}
    metadata = sample.get("metadata") if isinstance(sample.get("metadata"), Mapping) else {}
    return _is_swebench_sample(dict(sample), dict(metadata))


def _trial_payload(record: Mapping[str, Any]) -> Mapping[str, Any]:
    for key in ("trial", "trial_result"):
        trial = record.get(key)
        if isinstance(trial, Mapping):
            return trial
    trial_results = record.get("trial_results")
    if isinstance(trial_results, list):
        for trial in trial_results:
            if isinstance(trial, Mapping):
                return trial
    return first_agentkit_trial_result(record) or {}


def _sample_id(record: Mapping[str, Any]) -> str:
    sample = record.get("sample") if isinstance(record.get("sample"), Mapping) else {}
    return str(sample.get("id") or record.get("sample_id") or "sample")


def _trial_id(record: Mapping[str, Any], trial: Mapping[str, Any]) -> str | None:
    value = trial.get("trial_id") or record.get("trial_id")
    return str(value) if value not in (None, "") else None


def _humanize_reason_codes(reason_codes: list[str]) -> str:
    return ", ".join(_humanize_reason_code(code) for code in reason_codes)


def _humanize_reason_code(code: str) -> str:
    if code == "score.low":
        return code
    return code.replace("_", " ").replace(".", " ")


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
