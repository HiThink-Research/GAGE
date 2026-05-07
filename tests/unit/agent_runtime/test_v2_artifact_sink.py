from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pytest
from pydantic import BaseModel, ConfigDict, ValidationError

from gage_eval.agent_runtime.contracts.failure import FailureEnvelope, FailureEnvelopeError
from gage_eval.agent_runtime.artifacts import ArtifactWriteError, RuntimeArtifactSink
from gage_eval.agent_runtime.session import AgentRuntimeSession
from gage_eval.environment.manager import EnvironmentManagerError


def _artifact_ref_payload(name: str = "trace.jsonl") -> dict[str, object]:
    return {
        "owner": "infra",
        "name": name,
        "path": f"artifacts/task-1/sample-1/trials/trial_0001/infra/{name}",
        "mime_type": "application/jsonl",
        "size_bytes": 17,
        "sha256": "a" * 64,
    }


def _trial_result_payload() -> dict[str, object]:
    return {
        "trial_id": "trial_0001",
        "status": "completed",
        "scheduler_result": {"status": "completed"},
        "verifier_result": {"score": 1.0},
        "environment_descriptor": {"provider": "local_process"},
        "artifact_refs": [_artifact_ref_payload("agent.json")],
        "trace_ref": _artifact_ref_payload(),
        "failure": None,
    }


def _sample_record_payload(**overrides: object) -> dict[str, object]:
    effective_config_ref = _artifact_ref_payload("effective_config.json")
    payload: dict[str, object] = {
        "run_id": "run-1",
        "task_id": "task-1",
        "sample_id": "sample-1",
        "dut_id": "dut-1",
        "input_ref": {"dataset": "demo", "row": 1},
        "trial_policy": {"trials": 1},
        "trial_results": [_trial_result_payload()],
        "aggregate_result": {"trial_count": 1},
        "scheduler_result": {"status": "completed"},
        "verifier_result": {"score": 1.0},
        "environment_descriptor": {"provider": "local_process"},
        "effective_config_ref": effective_config_ref,
        "artifacts": [effective_config_ref],
        "status": "completed",
        "failure": None,
    }
    payload.update(overrides)
    return payload


def test_agent_artifact_layout_uses_trial_owner_dirs(tmp_path: Path) -> None:
    sink = RuntimeArtifactSink(base_dir=str(tmp_path))

    layout = sink.build_layout(run_id="run-1", task_id="task-1", sample_id="sample-1")
    assert layout["artifacts_root"].endswith("run-1/artifacts/task-1/sample-1")
    assert layout["sample_infra_dir"].endswith("run-1/artifacts/task-1/sample-1/infra")
    assert layout["effective_config"].endswith("run-1/artifacts/task-1/sample-1/infra/effective_config.json")
    assert layout["sample_record"].endswith("run-1/artifacts/task-1/sample-1/infra/sample_record.json")
    assert layout["trial_aggregate"].endswith("run-1/artifacts/task-1/sample-1/infra/trial_aggregate.json")
    assert not (tmp_path / "run-1" / "samples" / "runtime").exists()

    sample_ref = sink.write_artifact(
        run_id="run-1",
        task_id="task-1",
        sample_id="sample-1",
        owner="infra",
        name="sample_record.json",
        content=_sample_record_payload(),
    )
    assert sample_ref.path == "artifacts/task-1/sample-1/infra/sample_record.json"
    assert (tmp_path / "run-1" / sample_ref.path).is_file()

    for owner in ("agent", "infra", "verifier"):
        ref = sink.write_artifact(
            run_id="run-1",
            task_id="task-1",
            sample_id="sample-1",
            trial_id="trial_0001",
            owner=owner,
            name=f"{owner}.json",
            content={"owner": owner},
        )
        assert ref.path == f"artifacts/task-1/sample-1/trials/trial_0001/{owner}/{owner}.json"
        assert (tmp_path / "run-1" / ref.path).is_file()


def test_sample_level_write_without_trial_only_allows_infra_records(tmp_path: Path) -> None:
    sink = RuntimeArtifactSink(base_dir=str(tmp_path))

    sample_contents = {
        "effective_config.json": {},
        "sample_record.json": _sample_record_payload(run_id="run-2"),
        "trial_aggregate.json": {},
    }
    for name, content in sample_contents.items():
        ref = sink.write_artifact(
            run_id="run-2",
            task_id="task-1",
            sample_id="sample-1",
            owner="infra",
            name=name,
            content=content,
        )
        assert ref.path == f"artifacts/task-1/sample-1/infra/{name}"

    with pytest.raises(ValueError, match="trial_id"):
        sink.write_artifact(
            run_id="run-2",
            task_id="task-1",
            sample_id="sample-1",
            owner="agent",
            name="message.json",
            content={},
        )

    with pytest.raises(ValueError, match="sample-level"):
        sink.write_artifact(
            run_id="run-2",
            task_id="task-1",
            sample_id="sample-1",
            owner="verifier",
            name="trial_aggregate.json",
            content={},
        )


def test_sample_record_json_is_schema_validated_before_write(tmp_path: Path) -> None:
    sink = RuntimeArtifactSink(base_dir=str(tmp_path))

    with pytest.raises(ValidationError):
        sink.write_artifact(
            run_id="run-invalid",
            task_id="task-1",
            sample_id="sample-1",
            owner="infra",
            name="sample_record.json",
            content={"status": "completed"},
        )

    assert not (tmp_path / "run-invalid" / "artifacts/task-1/sample-1/infra/sample_record.json").exists()


def test_environment_lease_call_shape_returns_stable_dict(tmp_path: Path) -> None:
    sink = RuntimeArtifactSink(base_dir=str(tmp_path))

    ref = sink.write_artifact(
        owner="lease-artifacts",
        name="stdout.txt",
        content="abcdef",
        metadata={"stream": "stdout"},
    )

    assert ref == {
        "owner": "lease-artifacts",
        "name": "stdout.txt",
        "path": "artifacts/lease-artifacts/stdout.txt",
        "mime_type": "text/plain",
        "size_bytes": 6,
        "sha256": ref["sha256"],
        "metadata": {"stream": "stdout"},
    }
    assert (tmp_path / "artifacts" / "lease-artifacts" / "stdout.txt").read_text(encoding="utf-8") == "abcdef"


def test_effective_config_records_five_source_layers(tmp_path: Path) -> None:
    sink = RuntimeArtifactSink(base_dir=str(tmp_path))
    secret = "sk-test-secret"
    sentinel = "<redacted:reference:ENV.MODEL_API_KEY>"
    source_layers = [
        {"name": "raw_config", "values": {"agent": {"model": "raw-model"}}},
        {"name": "loader_defaults", "values": {"agent": {"model": "loader-model"}}},
        {"name": "kit_profile_defaults", "values": {"agent": {"model": "kit-model"}}},
        {"name": "provider_defaults", "values": {"agent": {"model": "provider-model"}}},
        {"name": "cli_overrides", "values": {"agent": {"model": "cli-model"}, "api_key": secret}},
    ]

    ref = sink.write_effective_config(
        run_id="run-3",
        task_id="task-1",
        sample_id="sample-1",
        final_config={"agent": {"model": "cli-model"}, "api_key": secret},
        source_layers=source_layers,
        secret_values={secret: sentinel},
    )

    payload = json.loads((tmp_path / "run-3" / ref.path).read_text(encoding="utf-8"))
    assert [layer["name"] for layer in payload["source_layers"]] == [
        "raw_config",
        "loader_defaults",
        "kit_profile_defaults",
        "provider_defaults",
        "cli_overrides",
    ]
    assert payload["final_config"]["agent"]["model"] == "cli-model"
    assert payload["final_config"]["api_key"] == sentinel
    assert payload["override_chain"]["agent.model"] == [
        {"layer": "raw_config", "value": "raw-model"},
        {"layer": "loader_defaults", "value": "loader-model"},
        {"layer": "kit_profile_defaults", "value": "kit-model"},
        {"layer": "provider_defaults", "value": "provider-model"},
        {"layer": "cli_overrides", "value": "cli-model"},
    ]
    assert secret not in json.dumps(payload)


def test_secret_value_not_written_to_trace_or_sample_record(tmp_path: Path) -> None:
    sink = RuntimeArtifactSink(base_dir=str(tmp_path))
    secret = "secret-token-value"
    sentinel = "<redacted:reference:ENV.MODEL_API_KEY>"
    secret_values = {secret: sentinel}

    config_ref = sink.write_effective_config(
        run_id="run-4",
        task_id="task-1",
        sample_id="sample-1",
        final_config={"api_key": secret, "agent": {"model": "demo"}},
        source_layers=[{"name": "raw_config", "values": {"api_key": secret}}],
        secret_values=secret_values,
    )
    config_text = (tmp_path / "run-4" / config_ref.path).read_text(encoding="utf-8")
    assert secret not in config_text
    assert sentinel in config_text

    sample_ref = sink.write_artifact(
        run_id="run-4",
        task_id="task-1",
        sample_id="sample-1",
        owner="infra",
        name="sample_record.json",
        content=_sample_record_payload(run_id="run-4", environment_descriptor={"provider": "local", "token": secret}),
        secret_values=secret_values,
    )
    sample_text = (tmp_path / "run-4" / sample_ref.path).read_text(encoding="utf-8")
    assert secret not in sample_text
    assert sentinel in sample_text

    trace_ref = sink.append_trace_event(
        run_id="run-4",
        task_id="task-1",
        sample_id="sample-1",
        trial_id="trial_0001",
        actor="environment",
        event_type="environment.acquire",
        payload={"environment_descriptor": {"token": secret}},
        secret_values=secret_values,
    )
    trace_text = (tmp_path / "run-4" / trace_ref.path).read_text(encoding="utf-8")
    assert secret not in trace_text
    assert sentinel in trace_text


def test_secret_keyname_values_are_redacted_without_secret_reference(tmp_path: Path) -> None:
    sink = RuntimeArtifactSink(base_dir=str(tmp_path))

    config_ref = sink.write_effective_config(
        run_id="run-keyname",
        task_id="task-1",
        sample_id="sample-1",
        final_config={"api_key": "dummy", "nested": {"access_token": "hardcoded-token"}},
        source_layers=[{"name": "raw_config", "values": {"api_key": "dummy"}}],
    )
    trace_ref = sink.append_trace_event(
        run_id="run-keyname",
        task_id="task-1",
        sample_id="sample-1",
        trial_id="trial_0001",
        actor="environment",
        event_type="environment.acquire",
        payload={"environment_descriptor": {"provider_config": {"api_key": "dummy"}}},
    )

    config_text = (tmp_path / "run-keyname" / config_ref.path).read_text(encoding="utf-8")
    trace_text = (tmp_path / "run-keyname" / trace_ref.path).read_text(encoding="utf-8")
    assert '"dummy"' not in config_text
    assert "hardcoded-token" not in config_text
    assert "<redacted:keyname:api_key>" in config_text
    assert "<redacted:keyname:access_token>" in config_text
    assert '"dummy"' not in trace_text
    assert "<redacted:keyname:api_key>" in trace_text


def test_secret_redaction_does_not_redact_token_usage_fields(tmp_path: Path) -> None:
    sink = RuntimeArtifactSink(base_dir=str(tmp_path))

    ref = sink.write_artifact(
        run_id="run-usage",
        task_id="task-1",
        sample_id="sample-1",
        trial_id="trial_0001",
        owner="agent",
        name="model_response_turn_1.json",
        content={
            "usage": {
                "prompt_tokens": 123,
                "completion_tokens": 456,
                "total_tokens": 579,
                "completion_tokens_details": {"reasoning_tokens": 7},
                "prompt_tokens_details": None,
                "agent_total_tokens": 579,
                "user_total_tokens": 89,
            },
            "api_key": "sk-real",
            "access_token": "access-real",
            "session_token": "session-real",
            "token": "plain-token-real",
        },
    )

    payload = json.loads((tmp_path / "run-usage" / ref.path).read_text(encoding="utf-8"))
    usage = payload["usage"]
    assert usage["prompt_tokens"] == 123
    assert usage["completion_tokens"] == 456
    assert usage["total_tokens"] == 579
    assert usage["completion_tokens_details"]["reasoning_tokens"] == 7
    assert usage["prompt_tokens_details"] is None
    assert usage["agent_total_tokens"] == 579
    assert usage["user_total_tokens"] == 89
    assert payload["api_key"] == "<redacted:keyname:api_key>"
    assert payload["access_token"] == "<redacted:keyname:access_token>"
    assert payload["session_token"] == "<redacted:keyname:session_token>"
    assert payload["token"] == "<redacted:keyname:token>"


def test_overlapping_secret_values_redact_longest_match_without_suffix_leakage(tmp_path: Path) -> None:
    sink = RuntimeArtifactSink(base_dir=str(tmp_path))

    ref = sink.write_artifact(
        run_id="run-overlap",
        task_id="task-1",
        sample_id="sample-1",
        trial_id="trial_0001",
        owner="infra",
        name="stdout.txt",
        content="token=abcdef",
        secret_values={
            "abc": "<redacted:reference:ENV.SHORT>",
            "abcdef": "<redacted:reference:ENV.LONG>",
        },
    )

    text = (tmp_path / "run-overlap" / ref.path).read_text(encoding="utf-8")
    assert "abcdef" not in text
    assert "def" not in text
    assert "<redacted:reference:ENV.LONG>" in text


def test_append_trace_event_normalizes_non_json_payload_values(tmp_path: Path) -> None:
    sink = RuntimeArtifactSink(base_dir=str(tmp_path))
    observed_at = datetime(2026, 4, 29, 1, 2, 3, tzinfo=timezone.utc)

    ref = sink.append_trace_event(
        run_id="run-trace-normalize",
        task_id="task-1",
        sample_id="sample-1",
        trial_id="trial_0001",
        actor="environment",
        event_type="environment.acquire",
        payload={
            "environment_descriptor": {
                "provider": "local_process",
                "observed_at": observed_at,
                "workdir": Path("/tmp/work"),
            }
        },
    )

    line = (tmp_path / "run-trace-normalize" / ref.path).read_text(encoding="utf-8").strip()
    event = json.loads(line)
    descriptor = event["payload"]["environment_descriptor"]
    assert descriptor["observed_at"] == {"object_type": "datetime.datetime"}
    assert descriptor["workdir"] == "/tmp/work"


def test_raw_error_preserves_environment_validation_details(tmp_path: Path) -> None:
    class _StrictDockerConfig(BaseModel):
        model_config = ConfigDict(extra="forbid")

        image: str

    sink = RuntimeArtifactSink(base_dir=str(tmp_path))
    layout = sink.build_layout(run_id="run-raw", task_id="task-1", sample_id="sample-1")
    session = AgentRuntimeSession(
        session_id="session-raw",
        run_id="run-raw",
        task_id="task-1",
        sample_id="sample-1",
        benchmark_kit_id="swebench",
        scheduler_type="framework_loop",
        artifact_layout=layout,
    )
    try:
        _StrictDockerConfig.model_validate({"provider_config": {"image": "fake-image:1"}})
    except ValidationError as exc:
        manager_error = EnvironmentManagerError(
            "environment.preflight_failed",
            "provider=docker profile_id=swebench_runtime",
            cause=exc,
        )
    else:  # pragma: no cover - validation model intentionally rejects the payload
        raise AssertionError("strict docker config accepted invalid payload")

    failure = FailureEnvelope(
        failure_domain="environment",
        failure_stage="acquire_lease",
        failure_code="environment.acquire_lease.resource_manager.acquire_failed",
        component_kind="resource_manager",
        component_id="swebench.resource.acquire",
        owner="runtime_resource_core",
        retryable=False,
        summary="environment failed",
        first_bad_step="environment.acquire_lease",
        suspect_files=("src/gage_eval/agent_runtime/resources/manager.py",),
    )
    try:
        raise FailureEnvelopeError(failure) from manager_error
    except FailureEnvelopeError as exc:
        path = sink.persist_raw_error(session=session, error=exc)

    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    validation_errors = payload["validation_errors"]
    validation_locs = {tuple(error["loc"]) for error in validation_errors}
    assert payload["failure"]["failure_code"] == "environment.acquire_lease.resource_manager.acquire_failed"
    assert payload["cause"]["error_type"] == "EnvironmentManagerError"
    assert payload["cause"]["manager_failure"]["failure_code"] == "environment.preflight_failed"
    assert ("image",) in validation_locs
    assert ("provider_config",) in validation_locs


def test_append_trace_event_sequence_scan_failure_maps_persistence_code(tmp_path: Path) -> None:
    sink = RuntimeArtifactSink(base_dir=str(tmp_path))
    trace_path = tmp_path / "run-trace-dir" / "artifacts/task-1/sample-1/trials/trial_0001/infra/trace.jsonl"
    trace_path.mkdir(parents=True)

    with pytest.raises(ArtifactWriteError) as excinfo:
        sink.append_trace_event(
            run_id="run-trace-dir",
            task_id="task-1",
            sample_id="sample-1",
            trial_id="trial_0001",
            actor="environment",
            event_type="environment.acquire",
            payload={"environment_descriptor": {"provider": "local_process"}},
        )

    assert excinfo.value.code == "persistence.artifact.write_failed"
    assert excinfo.value.failure["failure_code"] == "persistence.artifact.write_failed"


def test_artifact_write_failure_maps_persistence_code(tmp_path: Path) -> None:
    file_base = tmp_path / "not-a-directory"
    file_base.write_text("base", encoding="utf-8")
    sink = RuntimeArtifactSink(base_dir=str(file_base))

    with pytest.raises(ArtifactWriteError) as excinfo:
        sink.write_artifact(
            run_id="run-5",
            task_id="task-1",
            sample_id="sample-1",
            trial_id="trial_0001",
            owner="infra",
            name="trace.jsonl",
            content="{}",
        )

    assert excinfo.value.code == "persistence.artifact.write_failed"
    assert excinfo.value.failure["failure_code"] == "persistence.artifact.write_failed"
    assert excinfo.value.failure["component_kind"] == "artifact_sink"


def test_path_components_reject_traversal_and_absolute_paths(tmp_path: Path) -> None:
    sink = RuntimeArtifactSink(base_dir=str(tmp_path))

    with pytest.raises(ValueError, match="unsafe"):
        sink.write_artifact(
            run_id="run-6",
            task_id="../task",
            sample_id="sample-1",
            trial_id="trial_0001",
            owner="infra",
            name="trace.jsonl",
            content="{}",
        )

    with pytest.raises(ValueError, match="unsafe"):
        sink.write_artifact(
            run_id="run-6",
            task_id="task-1",
            sample_id="sample-1",
            trial_id="trial_0001",
            owner="infra",
            name="/absolute.json",
            content="{}",
        )
