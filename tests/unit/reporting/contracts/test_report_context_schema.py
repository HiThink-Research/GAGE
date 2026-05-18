from __future__ import annotations

import copy

import pytest

from gage_eval.reporting.contracts import (
    AttentionCase,
    AttentionCaseScoring,
    EvidenceRef,
    MetricScope,
    ReportContext,
    ReportContextSchema,
    Severity,
)


pytestmark = pytest.mark.fast


def _minimal_context_dict() -> dict[str, object]:
    return {
        "schema": {
            "name": "gage.report_context",
            "major": 1,
            "minor": 1,
            "renderer_compat": ">=1.0,<2.0",
            "generated_by": {"component": "ReportPackBuilder", "version": "1.1.0"},
            "optional_future_field": "ignored by v1 renderer",
        },
        "run": {"run_id": "run-demo", "run_dir": "runs/run-demo", "duration_s": 3.2},
        "headline": {
            "verdict": "passed",
            "verdict_reason": "1/1 samples completed",
            "one_line_summary": "Run completed without attention cases.",
            "primary_metric": None,
            "key_metric_ids": [],
            "top_attention_case_ids": [],
            "top_failure_cluster_ids": [],
            "top_outlier_metric_ids": [],
        },
        "runtime_health": {
            "sample_count": 1,
            "completed_count": 1,
            "failed_count": 0,
            "aborted_count": 0,
        },
        "observability_health": {
            "events_emitted_total": 12,
            "observability_degraded": False,
        },
        "metrics": [{"metric_id": "reward_mean", "scope": "run", "value": 0.7}],
        "tasks": [],
        "summary_sections": [],
        "attention_cases": [],
        "outliers": [],
        "case_details": {},
        "reason_code_counts": {"runtime": {}, "system": {}},
        "failure_clusters": [],
        "evidence_refs": [],
        "scenario_profiles": {},
        "methodology": {},
        "locale": {
            "language": "zh-CN",
            "timezone": "Asia/Shanghai",
            "number_format": {"thousands": True, "max_decimal_places": 5},
        },
        "report_assets": {"diagrams": [], "charts": [], "static_assets": []},
        "scoring_config": {
            "formula": "0.30*frequency + 0.50*impact_weight + 0.20*actionability_weight",
            "weights": {"frequency": 0.3, "impact": 0.5, "actionability": 0.2},
            "impact_weights": {
                "critical": 1.0,
                "high": 0.85,
                "medium": 0.6,
                "low": 0.3,
                "unknown": 0.4,
            },
            "actionability_weights": {
                "high": 1.0,
                "medium": 0.65,
                "low": 0.3,
                "unknown": 0.4,
            },
        },
        "diagnostics": {
            "report_pack_status": "completed",
            "warnings": [],
            "errors": [],
            "source_files": {},
        },
    }


def test_report_context_accepts_v1_minor_optional_fields_and_round_trips() -> None:
    payload = _minimal_context_dict()
    context = ReportContext.from_dict(payload)

    assert context.schema.major == 1
    assert context.schema.minor == 1
    assert context.validate() == []
    assert context.to_dict()["schema"]["optional_future_field"] == "ignored by v1 renderer"


def test_report_context_validate_reports_missing_required_fields_without_throwing() -> None:
    payload = _minimal_context_dict()
    del payload["schema"]["generated_by"]
    del payload["headline"]

    context = ReportContext.from_dict(payload)

    diagnostics = context.validate()
    assert {"code": "report_context.required_missing", "path": "schema.generated_by"} in diagnostics
    assert {"code": "report_context.required_missing", "path": "headline"} in diagnostics


def test_schema_requires_core_version_fields() -> None:
    schema = ReportContextSchema.from_dict(
        {
            "name": "gage.report_context",
            "major": 1,
            "minor": 0,
            "renderer_compat": ">=1.0,<2.0",
        }
    )

    assert schema.validate() == [
        {"code": "report_context.required_missing", "path": "schema.generated_by"}
    ]


def test_severity_rejects_unknown_values() -> None:
    assert Severity.validate("critical") == []
    assert Severity.validate("urgent") == [
        {
            "code": "report_context.invalid_enum",
            "path": "severity",
            "message": "Unsupported severity: urgent",
        }
    ]


def test_evidence_ref_requires_integrity_fields_and_relative_path() -> None:
    evidence = EvidenceRef.from_dict(
        {
            "ref_id": "evidence://sample/task-1/sample-1",
            "kind": "sample_record",
            "path": "/tmp/run/sample.json",
            "mime_type": "application/json",
        }
    )

    diagnostics = evidence.validate()
    assert {"code": "report_context.required_missing", "path": "evidence_refs[].sha256"} in diagnostics
    assert {
        "code": "report_context.required_missing",
        "path": "evidence_refs[].size_bytes",
    } in diagnostics
    assert {
        "code": "report_context.required_missing",
        "path": "evidence_refs[].timestamp_iso",
    } in diagnostics
    assert {
        "code": "report_context.path_not_relative",
        "path": "evidence_refs[].path",
        "message": "EvidenceRef.path must be run_dir relative",
    } in diagnostics


def test_media_evidence_ref_accepts_external_digest_without_file_integrity_metadata() -> None:
    evidence = EvidenceRef.from_dict(
        {
            "ref_id": "evidence://media/abcdef123456",
            "kind": "media",
            "path": "external://sha256/" + "0" * 64,
            "mime_type": "image/png",
            "sha256": "0" * 64,
        }
    )

    assert evidence.validate() == []


def test_media_evidence_ref_rejects_unsafe_external_url_path() -> None:
    evidence = EvidenceRef.from_dict(
        {
            "ref_id": "evidence://media/abcdef123456",
            "kind": "media",
            "path": "https://host/image.png?token=secret",
            "mime_type": "image/png",
            "sha256": "0" * 64,
        }
    )

    assert {
        "code": "report_context.invalid_media_path",
        "path": "evidence_refs[].path",
        "message": "Media EvidenceRef.path must be external://sha256/<64 hex>",
    } in evidence.validate()


def test_media_evidence_ref_rejects_digest_mismatch() -> None:
    evidence = EvidenceRef.from_dict(
        {
            "ref_id": "evidence://media/abcdef123456",
            "kind": "media",
            "path": "external://sha256/" + "0" * 64,
            "mime_type": "image/png",
            "sha256": "1" * 64,
        }
    )

    assert {
        "code": "report_context.media_sha256_mismatch",
        "path": "evidence_refs[].sha256",
        "message": "Media EvidenceRef.sha256 must match path digest",
    } in evidence.validate()


def test_metric_scope_requires_matching_owner_fields() -> None:
    assert MetricScope.validate({"metric_id": "reward", "scope": "task"}, "metrics[]") == [
        {"code": "report_context.required_missing", "path": "metrics[].task_id"}
    ]
    assert MetricScope.validate({"metric_id": "latency", "scope": "section"}, "metrics[]") == [
        {"code": "report_context.required_missing", "path": "metrics[].section_id"}
    ]


def test_attention_case_serialization_keeps_scoring_contract() -> None:
    case = AttentionCase(
        case_id="task/sample-1",
        severity="high",
        scoring=AttentionCaseScoring(
            frequency=0.1,
            impact="high",
            actionability="high",
            priority_score=0.8,
        ),
        reason_codes=["scheduler.failed"],
        summary="Scheduler failed.",
        evidence_ref_ids=["evidence://sample/task/sample-1"],
    )

    reloaded = AttentionCase.from_dict(copy.deepcopy(case.to_dict()))

    assert reloaded.to_dict()["scoring"] == {
        "frequency": 0.1,
        "impact": "high",
        "actionability": "high",
        "priority_score": 0.8,
    }
    assert reloaded.validate() == []


def test_report_context_object_validate_includes_cross_section_diagnostics() -> None:
    payload = _minimal_context_dict()
    payload["scenario_profiles"] = {
        "agent": {"representative_ref_ids": ["evidence://artifact/missing"]}
    }
    context = ReportContext.from_dict(payload)

    diagnostics = context.validate()

    assert {
        "code": "report_context.evidence_ref_missing",
        "path": "scenario_profiles.agent.representative_ref_ids[0]",
        "message": "Scenario profile evidence ref is missing: evidence://artifact/missing",
    } in diagnostics
