from __future__ import annotations

from copy import deepcopy
from typing import Any

from gage_eval.reporting.assembly.attention_detector import SCORING_CONFIG
from gage_eval.reporting.assembly.case_details_builder import CaseDetailsBuilder
from gage_eval.reporting.assembly.failure_clusterer import FailureClusterer
from gage_eval.reporting.assembly.headline_builder import HeadlineBuilder
from gage_eval.reporting.assembly.methodology_builder import MethodologyBuilder
from gage_eval.reporting.contracts import (
    AttentionCase,
    CaseDetails,
    EvidenceRef,
    FailureCluster,
    OutlierGroup,
    ReportContext,
    ReportContextSchema,
)


class ReportContextBuilder:
    def build(
        self,
        *,
        index: Any,
        summary_payload: dict[str, Any],
        metrics: list[dict[str, Any]],
        tasks: list[dict[str, Any]],
        runtime_health: dict[str, Any],
        observability_health: dict[str, Any],
        generator_result: Any = None,
    ) -> ReportContext:
        diagnostics = {
            "warnings": [],
            "errors": [],
            "report_pack_status": "completed",
            "profile_ref_resolution_miss_count": 0,
        }
        index_diagnostics = getattr(index, "diagnostics", None)
        if index_diagnostics is not None:
            diagnostics["warnings"].extend(list(getattr(index_diagnostics, "warnings", []) or []))
            diagnostics["errors"].extend(list(getattr(index_diagnostics, "errors", []) or []))
            diagnostics["profile_ref_resolution_miss_count"] = int(
                getattr(index_diagnostics, "profile_ref_resolution_miss_count", 0) or 0
            )
        raw_evidence_refs = getattr(index, "evidence_refs", []) or []
        evidence_refs = _normalize_evidence_refs(raw_evidence_refs)
        summary_sections = list(getattr(generator_result, "summary_sections", []) or [])
        attention_cases = [
            AttentionCase.from_dict(item) if isinstance(item, dict) else item
            for item in (getattr(generator_result, "attention_cases", []) or [])
        ]
        _backfill_attention_case_evidence(attention_cases, evidence_refs)
        case_details = _assemble_case_details(
            attention_cases,
            evidence_refs,
            getattr(generator_result, "case_details", {}) or {},
        )
        outliers = [
            OutlierGroup.from_dict(item) if isinstance(item, dict) else item
            for item in (getattr(generator_result, "outliers", []) or [])
        ]
        cluster_result = FailureClusterer().cluster(attention_cases)
        failure_clusters = [
            FailureCluster.from_dict(item) if isinstance(item, dict) else item
            for item in (getattr(generator_result, "failure_clusters", []) or cluster_result.failure_clusters)
        ]
        normalized_metrics = _normalize_metrics(metrics, scope="run")
        normalized_tasks = _normalize_tasks(
            tasks,
            runtime_health=runtime_health,
            attention_cases=attention_cases,
            failure_clusters=failure_clusters,
        )
        context = ReportContext(
            schema=ReportContextSchema(
                name="gage.report_context",
                major=1,
                minor=1,
                renderer_compat=">=1.0,<2.0",
                generated_by={"component": "ReportContextBuilder", "version": "1.1.0"},
            ),
            run=_run_payload(summary_payload),
            headline=HeadlineBuilder().build(
                metrics=normalized_metrics,
                runtime_health=runtime_health,
                attention_cases=attention_cases,
                outliers=outliers,
                failure_clusters=failure_clusters,
                diagnostics=diagnostics,
            ),
            runtime_health=runtime_health,
            observability_health=observability_health,
            metrics=normalized_metrics,
            tasks=normalized_tasks,
            summary_sections=summary_sections,
            attention_cases=attention_cases,
            outliers=outliers,
            case_details=case_details,
            reason_code_counts=cluster_result.reason_code_counts,
            failure_clusters=failure_clusters,
            evidence_refs=evidence_refs,
            scenario_profiles={},
            methodology=MethodologyBuilder().build(
                run_metadata=summary_payload,
                metrics=normalized_metrics,
                runtime_health=runtime_health,
                diagnostics=diagnostics,
            ),
            locale={"language": "en"},
            report_assets={"assets": []},
            scoring_config=deepcopy(SCORING_CONFIG),
            diagnostics=diagnostics,
        )
        validation = context.validate()
        if validation:
            context.diagnostics = dict(context.diagnostics or {})
            context.diagnostics["errors"] = validation
            context.diagnostics["report_pack_status"] = "degraded"
            context.headline = HeadlineBuilder().build(
                metrics=normalized_metrics,
                runtime_health=runtime_health,
                attention_cases=attention_cases,
                outliers=outliers,
                failure_clusters=failure_clusters,
                diagnostics=context.diagnostics,
            )
        return context


def _normalize_evidence_refs(raw_evidence_refs: Any) -> list[EvidenceRef]:
    values = raw_evidence_refs.values() if isinstance(raw_evidence_refs, dict) else raw_evidence_refs
    refs: list[EvidenceRef] = []
    for item in values or []:
        refs.append(EvidenceRef.from_dict(item) if isinstance(item, dict) else item)
    return refs


def _backfill_attention_case_evidence(attention_cases: list[AttentionCase], evidence_refs: list[EvidenceRef]) -> None:
    for case in attention_cases:
        if case.evidence_ref_ids:
            continue
        case.evidence_ref_ids = _evidence_ref_ids_for_case(case, evidence_refs)


def _assemble_case_details(
    attention_cases: list[AttentionCase],
    evidence_refs: list[EvidenceRef],
    raw_case_details: Any,
) -> dict[str, CaseDetails]:
    case_details = _normalize_case_details(raw_case_details)
    refs_by_id = {
        str(getattr(ref, "ref_id", "") or ""): ref
        for ref in evidence_refs
        if str(getattr(ref, "ref_id", "") or "")
    }
    builder = CaseDetailsBuilder()
    for case in attention_cases:
        case_id = str(getattr(case, "case_id", "") or "").strip()
        if not case_id:
            continue
        generated = _minimal_case_detail(case, refs_by_id, builder)
        existing = case_details.get(case_id)
        case_details[case_id] = generated if existing is None else _fill_missing_case_detail(existing, generated)
    return case_details


def _normalize_case_details(raw_case_details: Any) -> dict[str, CaseDetails]:
    items = raw_case_details.items() if isinstance(raw_case_details, dict) else []
    return {
        str(case_id): CaseDetails.from_dict(detail.to_dict())
        if isinstance(detail, CaseDetails)
        else CaseDetails.from_dict(detail)
        for case_id, detail in items
    }


def _minimal_case_detail(
    case: AttentionCase,
    refs_by_id: dict[str, EvidenceRef],
    builder: CaseDetailsBuilder,
) -> CaseDetails:
    evidence_ref_ids = list(getattr(case, "evidence_ref_ids", []) or [])
    return builder.build(
        {
            "evidence_ref_ids": evidence_ref_ids,
            "artifact_preview_ref_ids": evidence_ref_ids[:3],
            "scoring_breakdown": _attention_case_scoring(case),
            "full_trace_ref_id": _select_full_trace_ref_id(evidence_ref_ids, refs_by_id),
        }
    )


def _attention_case_scoring(case: AttentionCase) -> dict[str, Any]:
    scoring = getattr(case, "scoring", None)
    if scoring is None:
        return {}
    if hasattr(scoring, "to_dict"):
        return scoring.to_dict()
    if isinstance(scoring, dict):
        return dict(scoring)
    return {}


def _select_full_trace_ref_id(evidence_ref_ids: list[str], refs_by_id: dict[str, EvidenceRef]) -> str | None:
    fallback: str | None = None
    for ref_id in evidence_ref_ids:
        ref = refs_by_id.get(ref_id)
        if ref is None:
            continue
        path = str(getattr(ref, "path", "") or "")
        if "trace.jsonl" in path:
            return ref_id
        if fallback is None and _is_trace_adjacent_ref(path):
            fallback = ref_id
    return fallback


def _is_trace_adjacent_ref(path: str) -> bool:
    normalized = path.replace("\\", "/").lower()
    # These markers are report-owned artifact file names in current AgentKit
    # outputs; they are only a fallback when a trace.jsonl ref is unavailable.
    return any(
        marker in normalized
        for marker in (
            "trial_result",
            "sample_record",
            "scheduler_result",
        )
    )


def _fill_missing_case_detail(existing: CaseDetails, generated: CaseDetails) -> CaseDetails:
    if not existing.evidence_ref_ids:
        existing.evidence_ref_ids = list(generated.evidence_ref_ids)
    if not existing.artifact_preview_ref_ids:
        existing.artifact_preview_ref_ids = list(generated.artifact_preview_ref_ids)
    if not existing.scoring_breakdown:
        existing.scoring_breakdown = dict(generated.scoring_breakdown)
    if existing.full_trace_ref_id is None:
        existing.full_trace_ref_id = generated.full_trace_ref_id
    return existing


def _evidence_ref_ids_for_case(case: AttentionCase, evidence_refs: list[EvidenceRef], *, limit: int = 5) -> list[str]:
    sample_id = str(case.sample_id or "").strip()
    if not sample_id:
        return []
    task_id = str(case.task_id or "").strip()
    trial_id = str(case.trial_id or "").strip()
    ref_ids: list[str] = []
    seen: set[str] = set()
    for ref in evidence_refs:
        ref_id = str(getattr(ref, "ref_id", "") or "").strip()
        if not ref_id or ref_id in seen:
            continue
        if task_id and getattr(ref, "task_id", None) and str(ref.task_id) != task_id:
            continue
        if trial_id and getattr(ref, "trial_id", None) and str(ref.trial_id) != trial_id:
            continue
        if not _evidence_ref_matches_sample(ref, sample_id):
            continue
        ref_ids.append(ref_id)
        seen.add(ref_id)
        if len(ref_ids) >= limit:
            break
    return ref_ids


def _evidence_ref_matches_sample(ref: EvidenceRef, sample_id: str) -> bool:
    ref_sample_id = str(getattr(ref, "sample_id", "") or "").strip()
    if ref_sample_id and (
        ref_sample_id == sample_id
        or ref_sample_id.endswith(f":{sample_id}")
        or sample_id.endswith(f":{ref_sample_id}")
    ):
        return True
    path = str(getattr(ref, "path", "") or "")
    if _path_contains_segment(path, sample_id):
        return True
    return len(sample_id) >= 4 and sample_id in path


def _path_contains_segment(path: str, sample_id: str) -> bool:
    if not path or not sample_id:
        return False
    normalized_parts = [part for part in path.replace("\\", "/").split("/") if part]
    return sample_id in normalized_parts


def _run_payload(summary_payload: dict[str, Any]) -> dict[str, Any]:
    run = dict(summary_payload.get("run") or {})
    run.setdefault("run_id", summary_payload.get("run_id", "run"))
    run["sample_count"] = summary_payload.get("sample_count", run.get("sample_count"))
    return run


def _normalize_metrics(
    metrics: list[dict[str, Any]] | None,
    *,
    scope: str,
    task_id: Any = None,
) -> list[dict[str, Any]]:
    normalized: list[dict[str, Any]] = []
    for metric in metrics or []:
        if not isinstance(metric, dict):
            continue
        item = dict(metric)
        item.setdefault("scope", scope)
        if scope == "task" and task_id is not None:
            item.setdefault("task_id", task_id)
        normalized.append(item)
    return normalized


def _normalize_tasks(
    tasks: list[dict[str, Any]] | None,
    *,
    runtime_health: dict[str, Any],
    attention_cases: list[AttentionCase],
    failure_clusters: list[FailureCluster],
) -> list[dict[str, Any]]:
    normalized: list[dict[str, Any]] = []
    for task in tasks or []:
        if not isinstance(task, dict):
            continue
        item = dict(task)
        task_id = item.get("task_id")
        execution = item.get("execution") if isinstance(item.get("execution"), dict) else {}
        status = item.get("status") or execution.get("status")
        if not status:
            failed = int(runtime_health.get("failed_count") or 0)
            completed = int(runtime_health.get("completed_count") or 0)
            status = "failed" if failed and not completed else "completed"
        item["status"] = str(status)
        item.setdefault("runtime_health", _task_runtime_health(item, runtime_health))
        item.setdefault("attention_case_count", _count_for_task(attention_cases, task_id))
        item.setdefault("failure_cluster_count", _count_for_task(failure_clusters, task_id))
        item["metrics"] = _normalize_metrics(item.get("metrics", []), scope="task", task_id=task_id)
        normalized.append(item)
    return normalized


def _task_runtime_health(task: dict[str, Any], runtime_health: dict[str, Any]) -> dict[str, Any]:
    sample_count = int(task.get("sample_count") or runtime_health.get("sample_count") or 0)
    execution = task.get("execution") if isinstance(task.get("execution"), dict) else {}
    completed = sample_count if execution.get("status") == "completed" else int(runtime_health.get("completed_count") or 0)
    failed = int(runtime_health.get("failed_count") or 0)
    return {
        "sample_count": sample_count,
        "completed_count": completed,
        "failed_count": failed,
        "aborted_count": int(runtime_health.get("aborted_count") or 0),
        "verifier_skipped_count": int(runtime_health.get("verifier_skipped_count") or 0),
        "scheduler_failed_count": int(runtime_health.get("scheduler_failed_count") or 0),
    }


def _count_for_task(items: list[Any], task_id: Any) -> int:
    if not task_id:
        return 0
    count = 0
    for item in items:
        value = getattr(item, "task_id", None)
        if value is None and isinstance(item, dict):
            value = item.get("task_id")
        if value == task_id:
            count += 1
    return count
