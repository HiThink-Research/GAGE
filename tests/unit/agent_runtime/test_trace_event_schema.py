from __future__ import annotations

from datetime import datetime, timezone

import pytest
from pydantic import ValidationError

from gage_eval.agent_runtime.trace_schema import ArtifactRef, SampleRecord, TraceEvent, TrialResult


def _artifact_ref(name: str = "trace.jsonl") -> ArtifactRef:
    return ArtifactRef(
        owner="infra",
        name=name,
        path=f"artifacts/task-1/sample-1/trials/trial_0001/infra/{name}",
        mime_type="application/jsonl",
        size_bytes=17,
        sha256="a" * 64,
    )


def _trace_payload(**overrides: object) -> dict[str, object]:
    payload: dict[str, object] = {
        "run_id": "run-1",
        "task_id": "task-1",
        "sample_id": "sample-1",
        "trial_id": "trial_0001",
        "sequence_no": 1,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "actor": "environment",
        "event_type": "environment.acquire",
        "payload": {"environment_descriptor": {"provider": "local_process"}},
    }
    payload.update(overrides)
    return payload


def test_trace_event_schema_rejects_missing_trial_id() -> None:
    payload = _trace_payload()
    del payload["trial_id"]

    with pytest.raises(ValidationError):
        TraceEvent.model_validate(payload)


def test_environment_acquire_trace_event_has_actor_and_descriptor() -> None:
    event = TraceEvent.model_validate(_trace_payload())

    assert event.actor == "environment"
    assert event.event_type == "environment.acquire"
    assert event.payload["environment_descriptor"]["provider"] == "local_process"

    with pytest.raises(ValidationError, match="environment_descriptor"):
        TraceEvent.model_validate(_trace_payload(payload={}))


def test_verifier_result_trace_event_has_metric_and_artifact_refs() -> None:
    event = TraceEvent.model_validate(
        _trace_payload(
            actor="verifier",
            event_type="verifier.result",
            payload={"metric": {"score": 1.0}, "verifier_result": {"status": "passed"}},
            artifact_refs=[_artifact_ref("verifier_result.json").model_dump()],
        )
    )

    assert event.actor == "verifier"
    assert event.payload["metric"]["score"] == 1.0
    assert event.artifact_refs[0].owner == "infra"

    with pytest.raises(ValidationError, match="metric"):
        TraceEvent.model_validate(
            _trace_payload(
                actor="verifier",
                event_type="verifier.result",
                payload={"verifier_result": {"status": "passed"}},
                artifact_refs=[_artifact_ref("verifier_result.json").model_dump()],
            )
        )

    with pytest.raises(ValidationError, match="artifact_refs"):
        TraceEvent.model_validate(
            _trace_payload(
                actor="verifier",
                event_type="verifier.result",
                payload={
                    "metric": {"score": 1.0},
                    "verifier_result": {"status": "passed"},
                    "artifact_refs": [{"not": "an artifact ref"}],
                },
            )
        )


def test_trace_payload_rejects_large_streams_patch_full_diff_and_known_secret() -> None:
    with pytest.raises(ValidationError, match="stdout"):
        TraceEvent.model_validate(_trace_payload(payload={"stdout": "x" * 4097}))

    with pytest.raises(ValidationError, match="patch"):
        TraceEvent.model_validate(
            _trace_payload(
                payload={
                    "submission_patch": "diff --git a/a.py b/a.py\n--- a/a.py\n+++ b/a.py\n@@ -1 +1 @@\n-a\n+b\n"
                }
            )
        )

    with pytest.raises(ValidationError, match="secret"):
        TraceEvent.model_validate(
            _trace_payload(payload={"environment_descriptor": {"api_key": "sk-secret"}}),
            context={"secret_values": ["sk-secret"]},
        )


def test_artifact_ref_carries_required_fields_and_rejects_unsafe_paths() -> None:
    ref = _artifact_ref()

    assert set(ref.model_dump()) == {"owner", "name", "path", "mime_type", "size_bytes", "sha256"}

    with pytest.raises(ValidationError):
        ArtifactRef(owner="infra", name="trace.jsonl", path=ref.path, mime_type="application/jsonl", size_bytes=1)

    with pytest.raises(ValidationError, match="unsafe"):
        ArtifactRef(
            owner="infra",
            name="../trace.jsonl",
            path="../trace.jsonl",
            mime_type="application/jsonl",
            size_bytes=1,
            sha256="b" * 64,
        )


def test_sample_record_carries_required_fields() -> None:
    ref = _artifact_ref("effective_config.json")
    trial = TrialResult(
        trial_id="trial_0001",
        status="completed",
        scheduler_result={"status": "completed"},
        verifier_result={"score": 1.0},
        environment_descriptor={"provider": "local_process"},
        artifact_refs=[_artifact_ref("agent.json")],
        trace_ref=_artifact_ref(),
        failure=None,
    )

    record = SampleRecord(
        run_id="run-1",
        task_id="task-1",
        sample_id="sample-1",
        dut_id="dut-1",
        input_ref={"dataset": "demo", "row": 1},
        trial_policy={"trials": 1},
        trial_results=[trial],
        aggregate_result={"trial_count": 1},
        scheduler_result={"status": "completed"},
        verifier_result={"score": 1.0},
        environment_descriptor={"provider": "local_process"},
        effective_config_ref=ref,
        artifacts=[ref],
        status="completed",
        failure=None,
    )

    assert set(record.model_dump()) == {
        "run_id",
        "task_id",
        "sample_id",
        "dut_id",
        "input_ref",
        "trial_policy",
        "trial_results",
        "aggregate_result",
        "scheduler_result",
        "verifier_result",
        "environment_descriptor",
        "effective_config_ref",
        "artifacts",
        "status",
        "failure",
    }

    with pytest.raises(ValidationError):
        SampleRecord(
            run_id="run-1",
            task_id="task-1",
            sample_id="sample-1",
            dut_id="dut-1",
            input_ref={},
            trial_policy={},
            trial_results=[],
            aggregate_result={},
            scheduler_result={},
            verifier_result={},
            environment_descriptor={},
            effective_config_ref=ref,
            artifacts=[],
        )


def test_trial_result_carries_required_fields() -> None:
    result = TrialResult(
        trial_id="trial_0001",
        status="completed",
        scheduler_result={"status": "completed"},
        verifier_result={"score": 1.0},
        environment_descriptor={"provider": "local_process"},
        artifact_refs=[_artifact_ref("agent.json")],
        trace_ref=_artifact_ref(),
        failure=None,
    )

    assert set(result.model_dump()) == {
        "trial_id",
        "status",
        "scheduler_result",
        "verifier_result",
        "environment_descriptor",
        "artifact_refs",
        "trace_ref",
        "failure",
    }

    with pytest.raises(ValidationError):
        TrialResult(
            trial_id="trial_0001",
            scheduler_result={},
            verifier_result={},
            environment_descriptor={},
            artifact_refs=[],
            trace_ref=_artifact_ref(),
            failure=None,
        )
