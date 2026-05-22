"""Core observability event payload schema registry."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class EventSchema:
    """Declares required and optional payload fields for a core event."""

    required_fields: frozenset[str]
    optional_fields: frozenset[str] = frozenset()


@dataclass(frozen=True, slots=True)
class EventSchemaWarning:
    """Describes a non-blocking event schema validation warning."""

    code: str
    event: str
    field: str
    message: str

    def to_dict(self) -> dict[str, str]:
        """Return a JSON-serializable warning payload."""

        return {
            "code": self.code,
            "event": self.event,
            "field": self.field,
            "message": self.message,
        }


CORE_EVENT_SCHEMAS: dict[str, EventSchema] = {
    "runtime_bootstrap": EventSchema(
        required_fields=frozenset({"config_digest", "runtime_kind"}),
        optional_fields=frozenset({"profile", "entrypoint"}),
    ),
    "runtime_ready": EventSchema(
        required_fields=frozenset({"runtime_kind", "ready_state"}),
        optional_fields=frozenset({"environment_id", "capabilities"}),
    ),
    "task_start": EventSchema(
        required_fields=frozenset({"task_id", "sample_count"}),
        optional_fields=frozenset({"dataset_id", "shuffle_strategy"}),
    ),
    "task_end": EventSchema(
        required_fields=frozenset({"task_id", "status", "completed_count", "failed_count"}),
        optional_fields=frozenset({"aborted_count", "duration_s"}),
    ),
    "step_execution_started": EventSchema(
        required_fields=frozenset({"step_type", "step_id"}),
        optional_fields=frozenset({"component_kind", "component_id"}),
    ),
    "step_execution_completed": EventSchema(
        required_fields=frozenset({"step_type", "step_id", "elapsed_s"}),
        optional_fields=frozenset({"output_ref_ids"}),
    ),
    "step_execution_failed": EventSchema(
        required_fields=frozenset({"step_type", "step_id", "failure_code"}),
        optional_fields=frozenset({"error_type", "error_ref_id"}),
    ),
    "sample_start": EventSchema(
        required_fields=frozenset({"sample_id", "task_id"}),
        optional_fields=frozenset({"trial_id", "sample_index"}),
    ),
    "sample_end": EventSchema(
        required_fields=frozenset({"sample_id", "status", "elapsed_s"}),
        optional_fields=frozenset({"score", "artifact_ref_ids"}),
    ),
    "sample_failed": EventSchema(
        required_fields=frozenset({"sample_id", "failure_code", "failure_domain"}),
        optional_fields=frozenset({"error_ref_id", "retry_count"}),
    ),
    "auto_eval_sample": EventSchema(
        required_fields=frozenset({"sample_id", "status"}),
        optional_fields=frozenset({"judge_status", "score", "artifact_ref_ids"}),
    ),
    "runtime_session_start": EventSchema(
        required_fields=frozenset({"session_id", "runtime_kind"}),
        optional_fields=frozenset({"task_id", "sample_id"}),
    ),
    "runtime_session_end": EventSchema(
        required_fields=frozenset({"session_id", "status", "elapsed_s"}),
        optional_fields=frozenset({"failure_code", "artifact_ref_ids"}),
    ),
    "runtime_failure": EventSchema(
        required_fields=frozenset({"failure_code", "failure_domain"}),
        optional_fields=frozenset({"error_type", "error_ref_id", "recoverable"}),
    ),
    "external_harness_progress": EventSchema(
        required_fields=frozenset({"harness_id", "job_id", "progress"}),
        optional_fields=frozenset({"completed_trials", "total_trials", "raw_ref_ids"}),
    ),
    "report_finalize": EventSchema(
        required_fields=frozenset({"sample_count", "metric_count"}),
        optional_fields=frozenset({"task_count", "summary_ref_id"}),
    ),
    "report_pack_generated": EventSchema(
        required_fields=frozenset({"report_pack_path", "report_pack_status"}),
        optional_fields=frozenset({"context_ref_id", "html_ref_id"}),
    ),
    "report_pack_failed": EventSchema(
        required_fields=frozenset({"failure_code", "report_pack_status"}),
        optional_fields=frozenset({"error_ref_id", "partial_outputs"}),
    ),
}


def validate_event_payload(event: str, payload: dict[str, Any]) -> list[EventSchemaWarning]:
    """Validate a core event payload and return non-blocking warnings."""

    schema = CORE_EVENT_SCHEMAS.get(event)
    if schema is None:
        return []
    if _is_legacy_payload(event, payload):
        return []
    warnings: list[EventSchemaWarning] = []
    for field in sorted(schema.required_fields):
        if field not in payload:
            warnings.append(
                EventSchemaWarning(
                    code="missing_required_field",
                    event=event,
                    field=field,
                    message=f"Core event '{event}' is missing required payload field '{field}'.",
                )
            )
    return warnings


def _is_legacy_payload(event: str, payload: dict[str, Any]) -> bool:
    if event != "external_harness_progress":
        return False
    legacy_fields = {"job_name", "completed", "total", "phase", "elapsed_s"}
    new_fields = {"harness_id", "job_id", "progress", "completed_trials", "total_trials", "raw_ref_ids"}
    return bool(legacy_fields.intersection(payload)) and not bool(new_fields.intersection(payload))
