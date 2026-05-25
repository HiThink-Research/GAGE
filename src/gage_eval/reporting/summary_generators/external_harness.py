"""External harness summary generator."""

from __future__ import annotations

from typing import Any, Mapping

from gage_eval.registry import registry
from gage_eval.reporting.contracts import AttentionCase, SummaryGeneratorResult
from gage_eval.reporting.summary_generators import SummaryGenerator
from gage_eval.reporting.summary_generators.base import records_from_context, section
from gage_eval.reporting.summary_generators.harbor import _build_harbor_summary


@registry.asset(
    "summary_generators",
    "external_harness_summary",
    desc="External harness summary generator",
    tags=("external_harness", "report"),
    default_enabled=True,
)
class ExternalHarnessSummaryGenerator(SummaryGenerator):
    """Builds generic external-harness rollups from report context samples."""

    name = "external_harness_summary"

    def generate(self, context: Mapping[str, Any]) -> SummaryGeneratorResult | None:
        records = records_from_context(context)
        harbor = _build_harbor_summary(records)
        if harbor:
            summary: dict[str, Any] = {
                "harbor": harbor,
                "sample_count": harbor.get("sample_count", 0),
            }
        else:
            sample_count = sum(1 for record in records if _is_external_harness(record))
            if sample_count == 0:
                return None
            summary = {
                "sample_count": sample_count,
                "failure_rollup": _failure_rollup(records),
                "raw_artifact_paths": _raw_artifact_paths(records),
            }
        return SummaryGeneratorResult(
            generator_id=self.name,
            summary_sections=[
                section("overview", "External Harness Summary", generator_id=self.name)
            ],
            attention_cases=_attention_cases(records),
            legacy_payload={"external_harness": summary},
        )


def _is_external_harness(record: Mapping[str, Any]) -> bool:
    sample = _as_mapping(record.get("sample"))
    task_type = str(sample.get("task_type") or record.get("task_type") or "")
    return task_type.startswith("external_harness.")


def _failure_rollup(records: list[dict[str, Any]]) -> dict[str, dict[str, int]]:
    rollup: dict[str, dict[str, int]] = {"status_counts": {}, "failure_codes": {}}
    for record in records:
        for trial in record.get("trial_results") or []:
            if not isinstance(trial, Mapping):
                continue
            status = str(trial.get("status") or "unknown")
            rollup["status_counts"][status] = rollup["status_counts"].get(status, 0) + 1
            failure = _as_mapping(trial.get("failure"))
            code = failure.get("failure_code")
            if code:
                key = str(code)
                rollup["failure_codes"][key] = rollup["failure_codes"].get(key, 0) + 1
    return rollup


def _raw_artifact_paths(records: list[dict[str, Any]]) -> list[str]:
    paths: set[str] = set()
    for record in records:
        for ref in record.get("artifact_refs") or []:
            if isinstance(ref, Mapping) and ref.get("path"):
                paths.add(str(ref["path"]))
    return sorted(paths)


def _attention_cases(records: list[dict[str, Any]]) -> list[AttentionCase]:
    cases: list[AttentionCase] = []
    for record in records:
        sample = _as_mapping(record.get("sample"))
        sample_id = str(sample.get("id") or record.get("sample_id") or "sample")
        for trial in record.get("trial_results") or []:
            if not isinstance(trial, Mapping):
                continue
            status = str(trial.get("status") or "")
            if status in {"completed", "passed", ""}:
                continue
            failure = _as_mapping(trial.get("failure"))
            code = str(failure.get("failure_code") or "runtime.error")
            cases.append(
                AttentionCase.from_dict(
                    {
                        "case_id": f"external_harness/{sample_id}/{trial.get('trial_id') or len(cases) + 1}",
                        "severity": "high"
                        if status in {"failed", "aborted"}
                        else "medium",
                        "reason_codes": [code],
                        "summary": f"External harness trial ended with status {status}.",
                        "evidence_ref_ids": [],
                        "sample_id": sample_id,
                        "trial_id": str(trial.get("trial_id") or ""),
                        "scoring": {
                            "frequency": 1.0,
                            "impact": "high",
                            "actionability": "medium",
                            "priority_score": 0.75,
                        },
                    }
                )
            )
    return cases


def _as_mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


__all__ = ["ExternalHarnessSummaryGenerator"]
