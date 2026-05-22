from __future__ import annotations

from gage_eval.reporting.contracts.attention_case import (
    AttentionCase,
    AttentionCaseScoring,
)
from gage_eval.reporting.contracts.case_details import CaseDetails
from gage_eval.reporting.contracts.evidence_ref import EvidenceRef
from gage_eval.reporting.contracts.failure_cluster import FailureCluster
from gage_eval.reporting.contracts.outlier import MetricScope, OutlierEntry, OutlierGroup
from gage_eval.reporting.contracts.reason_codes import (
    ReasonCodeEntry,
    ReasonCodeRegistry,
)
from gage_eval.reporting.contracts.redaction import (
    RedactionFinding,
    RedactionResult,
    SecretPattern,
)
from gage_eval.reporting.contracts.report_context import ReportContext
from gage_eval.reporting.contracts.schema import ReportContextSchema
from gage_eval.reporting.contracts.severity import Severity
from gage_eval.reporting.contracts.summary_result import SummaryGeneratorResult


def validate_metric_scope(metric: dict, path: str = "metrics[]") -> list[str]:
    """Validate metric scope and return human-readable diagnostic strings."""
    diagnostics = MetricScope.validate(metric, path)
    return [
        str(item.get("path") or item.get("message") or item)
        for item in diagnostics
        if isinstance(item, dict)
    ]

__all__ = [
    "AttentionCase",
    "AttentionCaseScoring",
    "CaseDetails",
    "EvidenceRef",
    "FailureCluster",
    "MetricScope",
    "OutlierEntry",
    "OutlierGroup",
    "ReasonCodeEntry",
    "ReasonCodeRegistry",
    "RedactionFinding",
    "RedactionResult",
    "ReportContext",
    "ReportContextSchema",
    "SecretPattern",
    "Severity",
    "SummaryGeneratorResult",
    "validate_metric_scope",
]
