"""Summary generator for Harbor external harness imports."""

from __future__ import annotations

import math
from typing import Any, Dict, Iterable, Mapping, Optional

from gage_eval.registry import registry
from gage_eval.reporting.contracts import SummaryGeneratorResult
from gage_eval.reporting.summary_generators import SummaryGenerator
from gage_eval.reporting.summary_generators.base import records_from_context, section


@registry.asset(
    "summary_generators",
    "harbor_summary",
    desc="Harbor external harness summary generator",
    tags=("external_harness", "harbor", "report"),
    default_enabled=True,
)
class HarborSummaryGenerator(SummaryGenerator):
    def generate(self, context: Any) -> SummaryGeneratorResult | Dict[str, Any] | None:
        summary = _build_harbor_summary(records_from_context(context))
        if not summary:
            return None
        legacy_payload = {"external_harness": {"harbor": summary}}
        if hasattr(context, "iter_samples"):
            return legacy_payload
        return SummaryGeneratorResult(
            generator_id="harbor_summary",
            summary_sections=[
                section(
                    "overview",
                    "Harbor External Harness Summary",
                    generator_id="harbor_summary",
                )
            ],
            legacy_payload=legacy_payload,
        )


def _build_harbor_summary(records: Iterable[Mapping[str, Any]] | Any) -> Optional[Dict[str, Any]]:
    records = [_normalize_record(record) for record in records]
    records = [record for record in records if _is_harbor_record(record)]
    if not records:
        return None

    task_ids: set[str] = set()
    dataset_ids: set[str] = set()
    trial_count = 0
    completed_count = 0
    failed_count = 0
    aborted_count = 0
    skipped_count = 0
    resolve_values: list[float] = []
    score_values: list[float] = []
    pass_hat_values: list[float] = []
    raw_artifact_paths: list[str] = []
    failure_rollup: dict[str, dict[str, int]] = {
        "status_counts": {},
        "failure_codes": {},
        "failure_domains": {},
    }

    for record in records:
        sample = _mapping(record.get("sample"))
        eval_result = _mapping(sample.get("eval_result")) or _mapping(record.get("judge_output"))
        aggregate = _mapping(record.get("aggregate_result"))
        task_id = _task_id(record, sample)
        dataset_id = _dataset_id(sample)
        if task_id:
            task_ids.add(task_id)
        if dataset_id:
            dataset_ids.add(dataset_id)

        trials = _trial_results(record)
        trial_count += _int_or_default(aggregate.get("trial_count"), len(trials))
        completed_count += _int_or_default(aggregate.get("completed_trial_count"), _count_status(trials, "completed"))
        failed_count += _int_or_default(aggregate.get("failed_trial_count"), _count_non_completed(trials))
        aborted_count += _int_or_default(
            _mapping(_mapping(aggregate.get("failure_rollup")).get("status_counts")).get("aborted"),
            _count_status(trials, "aborted"),
        )
        skipped_count += len(_skipped_failed_trials(aggregate, eval_result))
        _merge_failure_rollup(failure_rollup, _mapping(aggregate.get("failure_rollup")))

        resolve = _float_or_none(eval_result.get("harbor_resolve_rate"))
        if resolve is not None:
            resolve_values.append(resolve)
        score = _float_or_none(eval_result.get("harbor_score_mean"))
        if score is not None:
            score_values.append(score)
        pass_hat = _pass_hat_k(_pass_values(eval_result), k=1)
        if pass_hat is not None:
            pass_hat_values.append(pass_hat)
        raw_artifact_paths.extend(_raw_artifact_paths(record))

    return {
        "task_ids": sorted(task_ids),
        "dataset_ids": sorted(dataset_ids),
        "sample_count": len(records),
        "trial_count": trial_count,
        "completed": completed_count,
        "failed": failed_count,
        "aborted": aborted_count,
        "skipped": skipped_count,
        "harbor_resolve_rate": _mean_or_none(resolve_values),
        "harbor_score_mean": _mean_or_none(score_values),
        "external_trial_pass_hat_k": {"pass_hat@1": _mean_or_none(pass_hat_values)},
        "raw_artifact_paths": sorted(set(raw_artifact_paths)),
        "failure_rollup": failure_rollup,
    }


def _normalize_record(record: Any) -> dict[str, Any]:
    return dict(record) if isinstance(record, Mapping) else {}


def _is_harbor_record(record: Mapping[str, Any]) -> bool:
    sample = _mapping(record.get("sample"))
    metadata = _mapping(sample.get("metadata"))
    harness = _mapping(metadata.get("_harness"))
    eval_result = _mapping(sample.get("eval_result")) or _mapping(record.get("judge_output"))
    if sample.get("task_type") == "external_harness.harbor":
        return True
    if harness.get("kit_id") == "harbor":
        return True
    return any(key in eval_result for key in ("harbor_resolve_rate", "harbor_score_mean", "external_trial_pass_values"))


def _task_id(record: Mapping[str, Any], sample: Mapping[str, Any]) -> str | None:
    namespace = str(record.get("namespace") or "")
    if namespace.startswith("task_"):
        return namespace.removeprefix("task_")
    if namespace.startswith("task/"):
        return namespace.removeprefix("task/")
    metadata = _mapping(sample.get("metadata"))
    value = metadata.get("task_id") or record.get("task_id")
    return str(value) if value else None


def _dataset_id(sample: Mapping[str, Any]) -> str | None:
    metadata = _mapping(sample.get("metadata"))
    dataset_source = _mapping(sample.get("dataset_source"))
    value = sample.get("dataset_id") or metadata.get("dataset_id") or dataset_source.get("benchmark")
    return str(value) if value else None


def _trial_results(record: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    raw = record.get("trial_results")
    if not isinstance(raw, list):
        return []
    return [item for item in raw if isinstance(item, Mapping)]


def _count_status(trials: Iterable[Mapping[str, Any]], status: str) -> int:
    return sum(1 for trial in trials if trial.get("status") == status)


def _count_non_completed(trials: Iterable[Mapping[str, Any]]) -> int:
    return sum(1 for trial in trials if trial.get("status") != "completed")


def _skipped_failed_trials(aggregate: Mapping[str, Any], eval_result: Mapping[str, Any]) -> list[Any]:
    projection = _mapping(eval_result.get("external_trial_metric_projection")) or _mapping(
        aggregate.get("metric_projection")
    )
    skipped = projection.get("skipped_failed_trials")
    return list(skipped) if isinstance(skipped, list) else []


def _merge_failure_rollup(target: dict[str, dict[str, int]], source: Mapping[str, Any]) -> None:
    for bucket in ("status_counts", "failure_codes", "failure_domains"):
        values = _mapping(source.get(bucket))
        for key, value in values.items():
            try:
                count = int(value)
            except (TypeError, ValueError):
                continue
            target[bucket][str(key)] = target[bucket].get(str(key), 0) + count


def _pass_values(eval_result: Mapping[str, Any]) -> list[bool | None]:
    raw = eval_result.get("external_trial_pass_values")
    if not isinstance(raw, list):
        return []
    values: list[bool | None] = []
    for value in raw:
        if isinstance(value, bool):
            values.append(value)
        elif value is None:
            values.append(None)
        else:
            values.append(bool(value))
    return values


def _pass_hat_k(values: list[bool | None], *, k: int) -> float | None:
    if not values:
        return None
    total = len(values)
    successes = sum(1 for value in values if value is True)
    if total < k or successes < k:
        return 0.0
    return math.comb(successes, k) / math.comb(total, k)


def _raw_artifact_paths(record: Mapping[str, Any]) -> list[str]:
    paths: list[str] = []
    for ref in record.get("artifact_refs") or []:
        if not isinstance(ref, Mapping):
            continue
        name = str(ref.get("name") or "")
        path = str(ref.get("path") or "")
        if name in {"harbor_raw_result.json", "harbor_job_result.json"} and path:
            paths.append(path)
    return paths


def _mean_or_none(values: list[float]) -> float | None:
    return (sum(values) / len(values)) if values else None


def _float_or_none(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _int_or_default(value: Any, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return int(default)


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


__all__ = ["HarborSummaryGenerator"]
