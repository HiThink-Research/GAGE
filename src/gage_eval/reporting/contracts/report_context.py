from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from gage_eval.reporting.contracts.attention_case import AttentionCase
from gage_eval.reporting.contracts.case_details import CaseDetails
from gage_eval.reporting.contracts.evidence_ref import EvidenceRef
from gage_eval.reporting.contracts.failure_cluster import FailureCluster
from gage_eval.reporting.contracts.outlier import MetricScope, OutlierGroup
from gage_eval.reporting.contracts.schema import ReportContextSchema
from gage_eval.reporting.contracts.severity import Severity


Diagnostic = dict[str, Any]


@dataclass
class ReportContext:
    """Defines the stable read model consumed by report renderers."""

    schema: ReportContextSchema = field(default_factory=ReportContextSchema)
    run: dict[str, Any] | None = None
    headline: dict[str, Any] | None = None
    runtime_health: dict[str, Any] | None = None
    observability_health: dict[str, Any] | None = None
    metrics: list[dict[str, Any]] = field(default_factory=list)
    tasks: list[dict[str, Any]] = field(default_factory=list)
    summary_sections: list[dict[str, Any]] = field(default_factory=list)
    attention_cases: list[AttentionCase] = field(default_factory=list)
    outliers: list[OutlierGroup] = field(default_factory=list)
    case_details: dict[str, CaseDetails] = field(default_factory=dict)
    reason_code_counts: dict[str, dict[str, int]] = field(
        default_factory=lambda: {"runtime": {}, "system": {}}
    )
    failure_clusters: list[FailureCluster] = field(default_factory=list)
    evidence_refs: list[EvidenceRef] = field(default_factory=list)
    scenario_profiles: dict[str, Any] = field(default_factory=dict)
    methodology: dict[str, Any] = field(default_factory=dict)
    locale: dict[str, Any] | None = None
    report_assets: dict[str, Any] | None = None
    scoring_config: dict[str, Any] | None = None
    diagnostics: dict[str, Any] | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def minimal(cls, run_id: str = "run") -> "ReportContext":
        """Builds the smallest valid v1 report context for tests and fallback runs."""
        return cls(
            schema=ReportContextSchema(
                name="gage.report_context",
                major=1,
                minor=0,
                renderer_compat=">=1.0,<2.0",
                generated_by={"component": "ReportPackBuilder", "version": "1.0.0"},
            ),
            run={"run_id": run_id, "run_dir": f"runs/{run_id}", "duration_s": 0.0},
            headline={
                "verdict": "passed",
                "verdict_reason": "0 samples processed",
                "one_line_summary": "Run completed without attention cases.",
                "primary_metric": None,
                "key_metric_ids": [],
                "top_attention_case_ids": [],
                "top_failure_cluster_ids": [],
                "top_outlier_metric_ids": [],
            },
            runtime_health={
                "sample_count": 0,
                "completed_count": 0,
                "failed_count": 0,
                "aborted_count": 0,
            },
            observability_health={
                "events_emitted_total": 0,
                "observability_degraded": False,
            },
            locale={
                "language": "zh-CN",
                "timezone": "Asia/Shanghai",
                "number_format": {"thousands": True, "max_decimal_places": 5},
            },
            report_assets={"diagrams": [], "charts": [], "static_assets": []},
            scoring_config={
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
            diagnostics={
                "report_pack_status": "completed",
                "warnings": [],
                "errors": [],
                "source_files": {},
            },
        )

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> "ReportContext":
        """Builds a report context from a JSON-compatible mapping."""
        payload = dict(data or {})
        context = cls(
            schema=ReportContextSchema.from_dict(payload.pop("schema", None)),
            run=payload.pop("run", None),
            headline=payload.pop("headline", None),
            runtime_health=payload.pop("runtime_health", None),
            observability_health=payload.pop("observability_health", None),
            metrics=list(payload.pop("metrics", [])),
            tasks=list(payload.pop("tasks", [])),
            summary_sections=list(payload.pop("summary_sections", [])),
            attention_cases=[
                AttentionCase.from_dict(item)
                for item in payload.pop("attention_cases", [])
            ],
            outliers=[OutlierGroup.from_dict(item) for item in payload.pop("outliers", [])],
            case_details={
                key: CaseDetails.from_dict(value)
                for key, value in payload.pop("case_details", {}).items()
            },
            reason_code_counts=dict(
                payload.pop("reason_code_counts", {"runtime": {}, "system": {}})
            ),
            failure_clusters=[
                FailureCluster.from_dict(item)
                for item in payload.pop("failure_clusters", [])
            ],
            evidence_refs=[
                EvidenceRef.from_dict(item) for item in payload.pop("evidence_refs", [])
            ],
            scenario_profiles=dict(payload.pop("scenario_profiles", {})),
            methodology=dict(payload.pop("methodology", {})),
            locale=payload.pop("locale", None),
            report_assets=payload.pop("report_assets", None),
            scoring_config=payload.pop("scoring_config", None),
            diagnostics=payload.pop("diagnostics", None),
            extra=payload,
        )
        return context

    def to_dict(self) -> dict[str, Any]:
        """Serializes the report context to a JSON-compatible mapping."""
        data: dict[str, Any] = {
            "schema": self.schema.to_dict(),
            "run": self.run,
            "headline": self.headline,
            "runtime_health": self.runtime_health,
            "observability_health": self.observability_health,
            "metrics": list(self.metrics),
            "tasks": list(self.tasks),
            "summary_sections": list(self.summary_sections),
            "attention_cases": [item.to_dict() for item in self.attention_cases],
            "outliers": [item.to_dict() for item in self.outliers],
            "case_details": {
                key: value.to_dict() for key, value in self.case_details.items()
            },
            "reason_code_counts": dict(self.reason_code_counts),
            "failure_clusters": [item.to_dict() for item in self.failure_clusters],
            "evidence_refs": [item.to_dict() for item in self.evidence_refs],
            "scenario_profiles": dict(self.scenario_profiles),
            "methodology": dict(self.methodology),
            "locale": self.locale,
            "report_assets": self.report_assets,
            "scoring_config": self.scoring_config,
            "diagnostics": self.diagnostics,
        }
        data.update(self.extra)
        return data

    def validate(self, renderer_major: int = 1) -> list[Diagnostic] | list[str]:
        """Returns diagnostics for contract violations without raising."""
        if not isinstance(self, ReportContext):
            return _validate_payload_for_renderer(self, renderer_major=renderer_major)

        diagnostics: list[Diagnostic] = []
        diagnostics.extend(self.schema.validate())
        for field_name in _REQUIRED_TOP_LEVEL_FIELDS:
            value = getattr(self, field_name)
            if value is None:
                diagnostics.append(
                    {
                        "code": "report_context.required_missing",
                        "path": field_name,
                    }
                )

        diagnostics.extend(_validate_metrics(self.metrics, "metrics[]", expected_scope="run"))
        for index, task in enumerate(self.tasks):
            task_path = f"tasks[{index}]"
            for field_name in _REQUIRED_TASK_FIELDS:
                if field_name not in task:
                    diagnostics.append(
                        {
                            "code": "report_context.required_missing",
                            "path": f"{task_path}.{field_name}",
                        }
                    )
            diagnostics.extend(
                _validate_metrics(task.get("metrics", []), f"{task_path}.metrics[]", expected_scope="task")
            )

        for index, section in enumerate(self.summary_sections):
            section_path = f"summary_sections[{index}]"
            if "severity" in section:
                diagnostics.extend(Severity.validate(section["severity"], f"{section_path}.severity"))
            metrics = section.get("metrics")
            if isinstance(metrics, dict):
                diagnostics.extend(MetricScope.validate(metrics, f"{section_path}.metrics"))

        for item in self.attention_cases:
            diagnostics.extend(item.validate())
        for item in self.outliers:
            diagnostics.extend(item.validate())
        for case_id, detail in self.case_details.items():
            diagnostics.extend(detail.validate(f"case_details.{case_id}"))
        for item in self.failure_clusters:
            diagnostics.extend(item.validate())
        for item in self.evidence_refs:
            diagnostics.extend(item.validate())
        return diagnostics


def _validate_payload_for_renderer(payload: Any, *, renderer_major: int = 1) -> list[str]:
    if not isinstance(payload, dict):
        return ["report_context must be a JSON object"]
    context = ReportContext.from_dict(payload)
    diagnostics = context.validate()
    errors = [
        str(item.get("path") or item.get("code") or item)
        for item in diagnostics
        if isinstance(item, dict)
    ]
    schema_major = context.schema.major
    if schema_major != renderer_major:
        errors.append(f"schema.major {schema_major} is not compatible with renderer major {renderer_major}")
    errors.extend(_cross_section_errors(context))
    return errors


def _cross_section_errors(context: ReportContext) -> list[str]:
    errors: list[str] = []
    ref_ids = {ref.ref_id for ref in context.evidence_refs if ref.ref_id}
    for case in context.attention_cases:
        for ref_id in case.evidence_ref_ids:
            if ref_id not in ref_ids:
                errors.append(f"evidence ref missing: {ref_id}")
        if _contains_secret_marker(case.to_dict()):
            errors.append(f"secret leak in attention case: {case.case_id}")
    return errors


def _contains_secret_marker(value: Any) -> bool:
    if isinstance(value, str):
        return "Bearer " in value or "sk-" in value or "Authorization:" in value
    if isinstance(value, dict):
        return any(_contains_secret_marker(child) for child in value.values())
    if isinstance(value, list):
        return any(_contains_secret_marker(child) for child in value)
    return False


def _validate_metrics(
    metrics: list[dict[str, Any]], path: str, expected_scope: str | None = None
) -> list[Diagnostic]:
    diagnostics: list[Diagnostic] = []
    for metric in metrics:
        diagnostics.extend(MetricScope.validate(metric, path))
        if expected_scope is not None and metric.get("scope") != expected_scope:
            diagnostics.append(
                {
                    "code": "report_context.metric_scope_mismatch",
                    "path": f"{path}.scope",
                    "message": f"Expected metric scope {expected_scope}, got {metric.get('scope')}",
                }
            )
    return diagnostics


_REQUIRED_TOP_LEVEL_FIELDS = (
    "run",
    "headline",
    "runtime_health",
    "observability_health",
    "locale",
    "report_assets",
    "scoring_config",
    "diagnostics",
)

_REQUIRED_TASK_FIELDS = (
    "task_id",
    "sample_count",
    "status",
    "metrics",
    "runtime_health",
    "attention_case_count",
    "failure_cluster_count",
)
