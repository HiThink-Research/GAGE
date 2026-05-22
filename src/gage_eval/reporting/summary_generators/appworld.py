"""AppWorld summary generator."""

from __future__ import annotations

import json
from collections import defaultdict
from typing import Any, Dict, Iterable, Mapping, Optional

from gage_eval.registry import registry
from gage_eval.reporting.assembly.attention_detector import AttentionCaseDetector
from gage_eval.reporting.contracts import SummaryGeneratorResult
from gage_eval.reporting.summary_generators import SummaryGenerator
from gage_eval.reporting.summary_generators._reason_codes import (
    agentkit_trial_results,
    extract_attention_reason_codes,
)
from gage_eval.reporting.summary_generators.base import records_from_context, section


@registry.asset(
    "summary_generators",
    "appworld_summary",
    desc="AppWorld summary generator",
    tags=("appworld", "report"),
    default_enabled=True,
)
class AppWorldSummaryGenerator(SummaryGenerator):
    def generate(self, context: Any) -> SummaryGeneratorResult | None:
        records = records_from_context(context)
        summary = _build_appworld_summary(records)
        if not summary:
            return None
        attention_cases = _build_appworld_attention_cases(records)
        return SummaryGeneratorResult(
            generator_id="appworld_summary",
            summary_sections=[section("overview", "AppWorld Summary", generator_id="appworld_summary")],
            attention_cases=attention_cases,
            legacy_payload={"appworld_summary": summary},
        )


def _build_appworld_summary(records: Iterable[dict[str, Any]]) -> Optional[Dict[str, Any]]:
    overall_total = 0
    overall_tgc_sum = 0.0
    overall_tgc_count = 0
    overall_sgc_sum = 0.0
    overall_sgc_count = 0
    by_subset: Dict[str, Dict[str, float]] = {}

    for record in records:
        if not isinstance(record, dict):
            continue
        sample = record.get("sample") if isinstance(record.get("sample"), dict) else None
        if not sample:
            continue
        metadata = sample.get("metadata") if isinstance(sample.get("metadata"), dict) else {}
        appworld_meta = metadata.get("appworld") if isinstance(metadata.get("appworld"), dict) else {}
        judge_output = record.get("judge_output") if isinstance(record.get("judge_output"), dict) else {}
        appworld_output = judge_output.get("appworld") if isinstance(judge_output.get("appworld"), dict) else {}
        if not appworld_meta and not appworld_output:
            continue
        subset = str(appworld_meta.get("subset") or "unknown")

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


def _build_appworld_attention_cases(records: list[dict[str, Any]]) -> list[Any]:
    appworld_records = [record for record in records if _is_appworld_record(record)]
    total_samples = max(len(appworld_records), 1)
    candidates: list[dict[str, Any]] = []
    reason_samples: dict[str, set[str]] = defaultdict(set)

    for record in appworld_records:
        sample_id = _sample_id(record)
        for trial in _attention_trials(record):
            if not _trial_or_verifier_failed(record, trial):
                continue
            reason_codes = extract_attention_reason_codes(
                _record_with_appworld_failure_code(record),
                trial=trial,
            )
            primary_reason = reason_codes[0]
            reason_samples[primary_reason].add(sample_id)
            trial_id = _trial_id(record, trial)
            candidates.append(
                {
                    "case_id": f"appworld/{sample_id}/{trial_id}",
                    "reason_codes": reason_codes,
                    "summary": f"AppWorld sample failed: {_humanize_reason_codes(reason_codes)}.",
                    "evidence_ref_ids": [],
                    "sample_id": sample_id,
                    "trial_id": trial_id,
                    "_primary_reason": primary_reason,
                }
            )

    for candidate in candidates:
        frequency = len(reason_samples[candidate.pop("_primary_reason")]) / total_samples
        candidate["frequency"] = frequency
    return AttentionCaseDetector().detect(candidates, total_samples=total_samples)


def _is_appworld_record(record: Mapping[str, Any]) -> bool:
    sample = record.get("sample") if isinstance(record.get("sample"), Mapping) else None
    metadata = sample.get("metadata") if isinstance(sample, Mapping) and isinstance(sample.get("metadata"), Mapping) else {}
    if isinstance(metadata.get("appworld"), Mapping):
        return True
    judge_output = record.get("judge_output") if isinstance(record.get("judge_output"), Mapping) else {}
    return isinstance(judge_output.get("appworld"), Mapping)


def _attention_trials(record: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    trials: list[Mapping[str, Any]] = []
    trial_results = record.get("trial_results")
    if isinstance(trial_results, list):
        trials.extend(trial for trial in trial_results if isinstance(trial, Mapping))
    for key in ("trial", "trial_result"):
        trial = record.get(key)
        if isinstance(trial, Mapping):
            trials.append(trial)
    trials.extend(agentkit_trial_results(record))
    return _dedupe_trials(trials) or [{}]


def _trial_or_verifier_failed(record: Mapping[str, Any], trial: Mapping[str, Any]) -> bool:
    if _status_failed(trial.get("status")):
        return True
    scheduler_result = trial.get("scheduler_result") if isinstance(trial.get("scheduler_result"), Mapping) else {}
    if _status_failed(scheduler_result.get("status")):
        return True
    verifier_result = trial.get("verifier_result") if isinstance(trial.get("verifier_result"), Mapping) else {}
    if _status_failed(verifier_result.get("status")):
        return True
    record_scheduler_result = record.get("scheduler_result") if isinstance(record.get("scheduler_result"), Mapping) else {}
    if _status_failed(record_scheduler_result.get("status")):
        return True
    judge_output = record.get("judge_output") if isinstance(record.get("judge_output"), Mapping) else {}
    if _status_failed(judge_output.get("status")):
        return True
    return bool(
        _appworld_failure_reason(record)
        or (isinstance(judge_output.get("verifier_failure"), Mapping) and judge_output["verifier_failure"].get("failure_code"))
        or judge_output.get("failure_code")
        or verifier_result.get("failure_code")
    )


def _record_with_appworld_failure_code(record: Mapping[str, Any]) -> Mapping[str, Any]:
    failure_reason = _appworld_failure_reason(record)
    if not failure_reason:
        return record
    payload = dict(record)
    verifier_result = dict(payload.get("verifier_result") or {})
    verifier_result.setdefault("failure_code", failure_reason)
    payload["verifier_result"] = verifier_result
    return payload


def _appworld_failure_reason(record: Mapping[str, Any]) -> str | None:
    judge_output = record.get("judge_output") if isinstance(record.get("judge_output"), Mapping) else {}
    appworld_output = judge_output.get("appworld") if isinstance(judge_output.get("appworld"), Mapping) else {}
    for value in (
        judge_output.get("failure_code"),
        appworld_output.get("failure_code"),
        appworld_output.get("failure_reason"),
        judge_output.get("failure_reason"),
    ):
        if value not in (None, ""):
            return str(value)
    return None


def _sample_id(record: Mapping[str, Any]) -> str:
    sample = record.get("sample") if isinstance(record.get("sample"), Mapping) else {}
    return str(sample.get("id") or record.get("sample_id") or "sample")


def _trial_id(record: Mapping[str, Any], trial: Mapping[str, Any]) -> str:
    return str(trial.get("trial_id") or record.get("trial_id") or "trial")


def _dedupe_trials(trials: list[Mapping[str, Any]]) -> list[Mapping[str, Any]]:
    deduped: list[Mapping[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    for trial in trials:
        key = _trial_dedupe_key(trial)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(trial)
    return deduped


def _trial_dedupe_key(trial: Mapping[str, Any]) -> tuple[str, str]:
    trial_id = trial.get("trial_id")
    if trial_id not in (None, ""):
        return ("trial_id", str(trial_id))
    return ("content", json.dumps(dict(trial), sort_keys=True, default=str))


def _status_failed(value: Any) -> bool:
    if isinstance(value, Mapping):
        value = value.get("value")
    # AppWorld live outputs use skipped verifier states as failure evidence when
    # the scheduler/trial already failed; do not reuse this helper for generic skips.
    return str(value or "").lower() in {"failed", "error", "errored", "aborted", "skipped"}


def _humanize_reason_codes(reason_codes: list[str]) -> str:
    return ", ".join(reason_codes)


def _coerce_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


__all__ = ["AppWorldSummaryGenerator"]
