from __future__ import annotations

import json
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from gage_eval.reporting.contracts import ReportContext
from gage_eval.reporting.privacy.secret_filter import SecretFilter


SCHEMA_VERSION = "1.1.0"


def normalize_context(context: ReportContext | dict[str, Any] | None) -> dict[str, Any]:
    if isinstance(context, ReportContext):
        payload = context.to_dict()
    elif isinstance(context, dict):
        payload = dict(context)
    else:
        payload = {}
    return SecretFilter().redact(_jsonable(payload)).value


def deterministic_json(value: Any) -> str:
    redacted = SecretFilter().redact(_jsonable(value)).value
    return json.dumps(redacted, ensure_ascii=False, indent=2, sort_keys=True)


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def install_report_context_minimal() -> None:
    if hasattr(ReportContext, "minimal"):
        return

    @classmethod
    def minimal(cls, run_id: str = "run") -> ReportContext:
        return cls.from_dict(
            {
                "schema": {
                    "name": "gage.report_context",
                    "major": 1,
                    "minor": 1,
                    "renderer_compat": ">=1.0,<2.0",
                    "generated_by": {
                        "component": "ReportPackBuilder",
                        "version": SCHEMA_VERSION,
                    },
                },
                "run": {"run_id": run_id, "run_dir": f"runs/{run_id}"},
                "headline": {
                    "verdict": "completed",
                    "verdict_reason": "Report context is available.",
                    "one_line_summary": "Run report generated.",
                    "primary_metric": None,
                    "key_metric_ids": [],
                    "top_attention_case_ids": [],
                    "top_failure_cluster_ids": [],
                    "top_outlier_metric_ids": [],
                },
                "runtime_health": {
                    "sample_count": 0,
                    "completed_count": 0,
                    "failed_count": 0,
                    "aborted_count": 0,
                },
                "observability_health": {
                    "events_emitted_total": 0,
                    "observability_degraded": False,
                },
                "metrics": [],
                "tasks": [],
                "summary_sections": [],
                "attention_cases": [],
                "outliers": [],
                "case_details": {},
                "reason_code_counts": {"runtime": {}, "system": {}},
                "failure_clusters": [],
                "evidence_refs": [],
                "scenario_profiles": {},
                "methodology": {
                    "generated_from": ["summary.json", "samples.jsonl"],
                    "notes": ["Evidence is referenced by redacted EvidenceRef entries."],
                },
                "locale": {
                    "language": "en-US",
                    "timezone": "UTC",
                    "number_format": {"thousands": True, "max_decimal_places": 5},
                },
                "report_assets": {"diagrams": [], "charts": [], "static_assets": []},
                "scoring_config": {},
                "diagnostics": {
                    "report_pack_status": "completed",
                    "warnings": [],
                    "errors": [],
                    "source_files": {},
                },
            }
        )

    setattr(ReportContext, "minimal", minimal)


def summarize_mapping(value: Any) -> str:
    if isinstance(value, dict):
        return ", ".join(f"{key}={_scalar_text(child)}" for key, child in sorted(value.items()))
    return _scalar_text(value)


def atomic_write_text(path: Path, text: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp")
    tmp.write_text(text, encoding="utf-8")
    tmp.replace(path)
    return path


def _jsonable(value: Any) -> Any:
    if is_dataclass(value):
        return _jsonable(asdict(value))
    if isinstance(value, dict):
        return {str(key): _jsonable(child) for key, child in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_jsonable(child) for child in value]
    if hasattr(value, "to_dict") and callable(value.to_dict):
        return _jsonable(value.to_dict())
    return value


def _scalar_text(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.5g}"
    if isinstance(value, (str, int, bool)) or value is None:
        return str(value)
    return json.dumps(_jsonable(value), ensure_ascii=False, sort_keys=True)


install_report_context_minimal()
