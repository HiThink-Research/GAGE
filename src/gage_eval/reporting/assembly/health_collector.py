from __future__ import annotations

from typing import Any


class RuntimeHealthCollector:
    """Collects deterministic runtime health counters from sample records."""

    def collect(self, records: list[dict[str, Any]] | None) -> dict[str, Any]:
        records = list(records or [])
        health = {
            "sample_count": len(records),
            "completed_count": 0,
            "failed_count": 0,
            "aborted_count": 0,
            "verifier_skipped_count": 0,
            "scheduler_failed_count": 0,
        }
        for record in records:
            judge_output = _record_judge_output(record)
            model_output = record.get("model_output")
            scheduler_result = _record_scheduler_result(record)
            runtime_failure = _record_runtime_failure(record)
            scheduler_failed = _scheduler_failed(scheduler_result, runtime_failure)
            verifier_skipped = _verifier_skipped(judge_output)
            has_successful_model_output = (
                isinstance(model_output, dict)
                and bool(model_output)
                and not runtime_failure
                and not scheduler_failed
                and not verifier_skipped
            )
            status = (
                _status_value(record.get("status"))
                or _status_value((record.get("sample") or {}).get("status"))
                or _status_value(judge_output.get("status"))
                or _status_from_trial_results(record)
            )
            if scheduler_failed:
                health["scheduler_failed_count"] += 1
            if verifier_skipped:
                health["verifier_skipped_count"] += 1

            if status == "aborted":
                health["aborted_count"] += 1
            elif status == "completed" or (not status and has_successful_model_output):
                health["completed_count"] += 1
            elif scheduler_failed or status == "failed" or status == "skipped" or not status:
                health["failed_count"] += 1
        return health


def _record_judge_output(record: dict[str, Any]) -> dict[str, Any]:
    judge_output = record.get("judge_output")
    if isinstance(judge_output, dict):
        return dict(judge_output)
    model_output = record.get("model_output")
    if not isinstance(model_output, dict):
        return {}
    runtime_outcome = model_output.get("runtime_judge_outcome")
    if isinstance(runtime_outcome, dict) and isinstance(runtime_outcome.get("judge_output"), dict):
        return dict(runtime_outcome["judge_output"])
    return {}


def _record_scheduler_result(record: dict[str, Any]) -> dict[str, Any]:
    model_output = record.get("model_output")
    if not isinstance(model_output, dict):
        return {}
    runtime_outcome = model_output.get("runtime_judge_outcome")
    if not isinstance(runtime_outcome, dict):
        return {}
    verifier_input = runtime_outcome.get("verifier_input")
    if isinstance(verifier_input, dict) and isinstance(verifier_input.get("scheduler_result"), dict):
        return dict(verifier_input["scheduler_result"])
    return {}


def _record_runtime_failure(record: dict[str, Any]) -> dict[str, Any]:
    model_output = record.get("model_output")
    if isinstance(model_output, dict) and isinstance(model_output.get("runtime_failure"), dict):
        return dict(model_output["runtime_failure"])
    return {}


def _status_from_trial_results(record: dict[str, Any]) -> str:
    trial_results = record.get("trial_results")
    if not isinstance(trial_results, list):
        return ""

    statuses = [
        str(trial.get("status") or "").lower()
        for trial in trial_results
        if isinstance(trial, dict)
    ]
    if any(status in {"completed", "passed"} for status in statuses):
        return "completed"
    if any(status == "aborted" for status in statuses):
        return "aborted"
    if any(status in {"failed", "error", "errored"} for status in statuses):
        return "failed"
    if any(status == "skipped" for status in statuses):
        return "skipped"
    return ""


def _status_value(value: Any) -> str:
    if isinstance(value, dict):
        value = value.get("value")
    return str(value or "").lower()


def _scheduler_failed(scheduler_result: dict[str, Any], runtime_failure: dict[str, Any]) -> bool:
    if _status_value(scheduler_result.get("status")) in {"failed", "aborted"}:
        return True
    failure_code = str(
        runtime_failure.get("failure_code")
        or scheduler_result.get("failure_code")
        or ""
    )
    return failure_code.startswith("client_execution.")


def _verifier_skipped(judge_output: dict[str, Any]) -> bool:
    if _status_value(judge_output.get("status")) == "skipped":
        return True
    return judge_output.get("failure_code") == "verifier.skipped_due_to_scheduler_failure"
