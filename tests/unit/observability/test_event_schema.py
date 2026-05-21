from __future__ import annotations

import pytest

from gage_eval.observability.event_schema import CORE_EVENT_SCHEMAS, validate_event_payload
from gage_eval.observability.trace import ObservabilityTrace
from gage_eval.reporting.recorders import InMemoryRecorder


@pytest.mark.fast
def test_core_event_schema_registry_contains_expected_events() -> None:
    assert set(CORE_EVENT_SCHEMAS) == {
        "runtime_bootstrap",
        "runtime_ready",
        "task_start",
        "task_end",
        "step_execution_started",
        "step_execution_completed",
        "step_execution_failed",
        "sample_start",
        "sample_end",
        "sample_failed",
        "auto_eval_sample",
        "runtime_session_start",
        "runtime_session_end",
        "runtime_failure",
        "external_harness_progress",
        "report_finalize",
        "report_pack_generated",
        "report_pack_failed",
    }


@pytest.mark.fast
def test_core_event_validation_reports_missing_required_fields() -> None:
    warnings = validate_event_payload("runtime_bootstrap", {"runtime_kind": "local"})

    assert [warning.to_dict() for warning in warnings] == [
        {
            "code": "missing_required_field",
            "event": "runtime_bootstrap",
            "field": "config_digest",
            "message": "Core event 'runtime_bootstrap' is missing required payload field 'config_digest'.",
        }
    ]


@pytest.mark.fast
def test_unknown_event_schema_passes_through_without_warnings() -> None:
    assert validate_event_payload("legacy_freeform_event", {}) == []


@pytest.mark.fast
def test_legacy_external_harness_progress_payload_passes_through_without_warnings() -> None:
    assert (
        validate_event_payload(
            "external_harness_progress",
            {
                "job_name": "tb2_job",
                "completed": 1,
                "total": 3,
                "phase": "running",
                "elapsed_s": 12.5,
            },
        )
        == []
    )


@pytest.mark.fast
def test_trace_emit_attaches_schema_warnings_to_invalid_core_event() -> None:
    recorder = InMemoryRecorder(run_id="schema-warning")
    trace = ObservabilityTrace(recorder=recorder, run_id="schema-warning")

    trace.emit("runtime_bootstrap", {"runtime_kind": "local"})

    event = trace.events[0]
    assert event["event"] == "runtime_bootstrap"
    assert event["payload"]["runtime_kind"] == "local"
    assert event["payload"]["schema_warnings"][0]["field"] == "config_digest"


@pytest.mark.fast
def test_trace_emit_leaves_unknown_event_payload_unchanged() -> None:
    trace = ObservabilityTrace(
        recorder=InMemoryRecorder(run_id="schema-unknown"),
        run_id="schema-unknown",
    )

    trace.emit("legacy_freeform_event", {"anything": True})

    assert trace.events[0]["payload"] == {"anything": True}
