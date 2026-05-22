from __future__ import annotations

from typing import Any

from gage_eval.reporting.evidence.external_harness.base import (
    ExternalHarnessAdapter,
    ExternalHarnessJob,
    ExternalHarnessTrial,
)


class HarborHarnessAdapter(ExternalHarnessAdapter):
    harness_id = "harbor"
    adapter_id = "external_harness.harbor"

    def detect(self, evidence: Any) -> bool:
        return any(_is_harbor_sample(sample) for sample in _samples(evidence))

    def normalize_job(self, evidence: Any) -> ExternalHarnessJob:
        samples = _samples(evidence)
        first = samples[0] if samples else {}
        harness = _harness_metadata(first)
        aggregate = _aggregate(samples)
        return ExternalHarnessJob(
            harness_id=self.harness_id,
            job_id=_string(harness.get("job_name") or harness.get("job_id")),
            status=_job_status(samples),
            raw_ref_ids=_raw_ref_ids(evidence, samples),
            aggregate=aggregate,
        )

    def normalize_trials(self, evidence: Any) -> list[ExternalHarnessTrial]:
        trials: list[ExternalHarnessTrial] = []
        raw_by_trial = _raw_ref_ids_by_trial(evidence, _samples(evidence))
        for sample in _samples(evidence):
            for raw_trial in sample.get("trial_results") or []:
                if not isinstance(raw_trial, dict):
                    continue
                trial_id = _string(raw_trial.get("trial_id"))
                if not trial_id:
                    continue
                trials.append(
                    ExternalHarnessTrial(
                        trial_id=trial_id,
                        status=_string(raw_trial.get("status")),
                        raw_ref_ids=raw_by_trial.get(trial_id, []),
                        metrics=_trial_metrics(raw_trial),
                        failure=raw_trial.get("failure") if isinstance(raw_trial.get("failure"), dict) else None,
                    )
                )
        return sorted(trials, key=lambda trial: trial.trial_id)

    def project_metrics(self, evidence: Any) -> list[dict[str, Any]]:
        metrics: list[dict[str, Any]] = []
        for sample in _samples(evidence):
            eval_result = _mapping(_mapping(sample.get("sample")).get("eval_result"))
            for key in ("harbor_resolve_rate", "harbor_score_mean", "external_trial_pass_values"):
                if key in eval_result:
                    metrics.append({"name": key, "value": eval_result[key], "harness_id": self.harness_id})
        return metrics


def _samples(evidence: Any) -> list[dict[str, Any]]:
    samples = getattr(evidence, "samples", evidence)
    if isinstance(samples, list):
        return [sample for sample in samples if isinstance(sample, dict)]
    return []


def _is_harbor_sample(sample: dict[str, Any]) -> bool:
    sample_payload = _mapping(sample.get("sample"))
    task_type = _string(sample_payload.get("task_type") or sample.get("task_type"))
    harness = _harness_metadata(sample)
    if task_type == "external_harness.harbor":
        return True
    if harness.get("kit_id") == "harbor" or "harbor_task_key" in harness:
        return True
    return any("harbor_" in str(ref.get("path", "")) for ref in _artifact_refs(sample))


def _harness_metadata(sample: dict[str, Any]) -> dict[str, Any]:
    metadata = _mapping(_mapping(sample.get("sample")).get("metadata"))
    return _mapping(metadata.get("_harness"))


def _aggregate(samples: list[dict[str, Any]]) -> dict[str, Any]:
    for sample in samples:
        aggregate = sample.get("aggregate_result")
        if isinstance(aggregate, dict):
            return {
                key: value
                for key, value in aggregate.items()
                if key in {"trial_count", "completed_trial_count", "failed_trial_count", "metric_projection", "failure_rollup"}
            }
    return {}


def _job_status(samples: list[dict[str, Any]]) -> str | None:
    statuses = {_string(trial.get("status")) for sample in samples for trial in sample.get("trial_results") or [] if isinstance(trial, dict)}
    statuses.discard(None)
    if not statuses:
        return _string(samples[0].get("status")) if samples else None
    if statuses <= {"completed"}:
        return "completed"
    if "failed" in statuses:
        return "failed"
    if "aborted" in statuses or "cancelled" in statuses:
        return "aborted"
    return "running"


def _raw_ref_ids(evidence: Any, samples: list[dict[str, Any]]) -> list[str]:
    by_path = _evidence_ref_ids_by_path(evidence)
    ref_ids: list[str] = []
    for ref in [ref for sample in samples for ref in _artifact_refs(sample)]:
        path = _string(ref.get("path"))
        if not path:
            continue
        ref_ids.append(by_path.get(path, path))
    return sorted(set(ref_ids))


def _raw_ref_ids_by_trial(evidence: Any, samples: list[dict[str, Any]]) -> dict[str, list[str]]:
    by_path = _evidence_ref_ids_by_path(evidence)
    raw: dict[str, list[str]] = {}
    for ref in [ref for sample in samples for ref in _artifact_refs(sample)]:
        path = _string(ref.get("path"))
        trial_id = _trial_id_from_ref(ref, path)
        if not path or not trial_id:
            continue
        raw.setdefault(trial_id, []).append(by_path.get(path, path))
    return {key: sorted(set(value)) for key, value in raw.items()}


def _evidence_ref_ids_by_path(evidence: Any) -> dict[str, str]:
    refs = getattr(evidence, "evidence_refs", {})
    if not isinstance(refs, dict):
        return {}
    result: dict[str, str] = {}
    for ref_id, ref in refs.items():
        path = getattr(ref, "path", None)
        if isinstance(path, str):
            result[path] = getattr(ref, "ref_id", None) or str(ref_id)
    return result


def _artifact_refs(sample: dict[str, Any]) -> list[dict[str, Any]]:
    refs = sample.get("artifact_refs")
    return [ref for ref in refs if isinstance(ref, dict)] if isinstance(refs, list) else []


def _trial_metrics(trial: dict[str, Any]) -> dict[str, Any]:
    verifier = _mapping(trial.get("verifier_result"))
    metrics: dict[str, Any] = {}
    for key in ("score", "reward", "passed", "resolved"):
        if key in verifier:
            metrics[key] = verifier[key]
    return metrics


def _trial_id_from_ref(ref: dict[str, Any], path: str | None) -> str | None:
    trial_id = _string(ref.get("trial_id"))
    if trial_id:
        return trial_id
    if not path:
        return None
    parts = path.split("/")
    if "trials" not in parts:
        return None
    index = parts.index("trials")
    return parts[index + 1] if index + 1 < len(parts) else None


def _mapping(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _string(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value)
    return text if text else None
