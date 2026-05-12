from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

import pytest

from gage_eval.agent_runtime.artifacts import RuntimeArtifactSink
from gage_eval.agent_runtime.compiled_plan import CompiledRuntimePlan, SchedulerWorkflowBundle
from gage_eval.agent_runtime.contracts.failure import FailureEnvelope, FailureEnvelopeError
from gage_eval.agent_runtime.contracts.scheduler import SchedulerResult
from gage_eval.agent_runtime.executor import (
    AgentRuntimeSessionFactory,
    CompiledRuntimeExecutor,
    DefaultVerifierRunner,
)
from gage_eval.agent_runtime.failure_mapper import FailureMapper
from gage_eval.agent_runtime.resources.contracts import ResourceLease
from gage_eval.agent_runtime.resources.manager import RuntimeLeaseBinding
from gage_eval.agent_runtime.session import AgentRuntimeSession
from gage_eval.agent_runtime.spec import AgentRuntimeSpec
from gage_eval.agent_runtime.trace_schema import ArtifactRef, TrialResult
from gage_eval.agent_runtime.trials import (
    TrialManager,
    TrialPolicy,
    TrialPolicyError,
    aggregate_trial_results,
)
from gage_eval.agent_runtime.verifier.binding import JudgeBinding
from gage_eval.agent_runtime.verifier.contracts import RuntimeJudgeOutcome, VerifierInput, VerifierResult
from gage_eval.environment.lease import EnvironmentLease
from gage_eval.observability.trace import ObservabilityTrace
from gage_eval.reporting.recorders import InMemoryRecorder


def _artifact_ref(name: str, trial_id: str = "trial_0001") -> ArtifactRef:
    mime_type = "application/jsonl" if name.endswith(".jsonl") else "application/json"
    return ArtifactRef(
        owner="infra",
        name=name,
        path=f"artifacts/task-1/sample-1/trials/{trial_id}/infra/{name}",
        mime_type=mime_type,
        size_bytes=2,
        sha256="a" * 64,
    )


def _trial(
    trial_id: str,
    *,
    status: str = "completed",
    score: float | None = None,
    resolved: bool | None = None,
    passed: bool | None = None,
    failure_code: str | None = None,
) -> TrialResult:
    verifier_result: dict[str, Any] = {"status": status}
    if score is not None:
        verifier_result["score"] = score
    if resolved is not None:
        verifier_result["resolved"] = resolved
    if passed is not None:
        verifier_result["passed"] = passed
    failure = None
    if failure_code:
        failure = {"failure_code": failure_code, "failure_domain": "verifier"}
    return TrialResult(
        trial_id=trial_id,
        status=status,  # type: ignore[arg-type]
        scheduler_result={"status": status},
        verifier_result=verifier_result,
        environment_descriptor={"lease_id": f"lease-{trial_id}"},
        artifact_refs=[_artifact_ref("artifact.json", trial_id)],
        trace_ref=_artifact_ref("trace.jsonl", trial_id),
        failure=failure,
    )


def test_trial_policy_defaults_to_single_trial() -> None:
    policy = TrialPolicy.from_mapping({})

    assert policy.trials == 1
    assert policy.environment_scope == "per_trial"
    assert policy.parallelism == 1
    assert policy.aggregation == "single"
    assert policy.trial_ids() == ["trial_0001"]


@pytest.mark.parametrize(
    "payload",
    [
        {"trials": 2, "aggregation": "single"},
        {"trials": 2, "parallelism": 2},
        {"trials": 1, "environment_scope": "per_sample"},
    ],
)
def test_invalid_trial_policy_maps_failure_code(payload: dict[str, Any]) -> None:
    with pytest.raises(TrialPolicyError) as excinfo:
        TrialPolicy.from_mapping(payload)

    assert excinfo.value.code == "config.trial_policy.invalid"
    assert "config.trial_policy.invalid" in str(excinfo.value)


def test_trial_aggregate_projects_samples_jsonl_to_primary_trial() -> None:
    aggregate = aggregate_trial_results(
        [
            _trial("trial_0001", score=0.0, resolved=False),
            _trial("trial_0002", score=1.0, resolved=True),
        ],
        aggregation="all",
    )

    assert aggregate.primary_trial_id == "trial_0001"
    assert aggregate.score_mean == 0.5
    assert aggregate.pass_rate == 0.5
    assert aggregate.samples_jsonl_projection["score"]["value"] == 0.0
    assert aggregate.samples_jsonl_projection["score"]["source_trial_id"] == "trial_0001"
    assert aggregate.samples_jsonl_projection["primary_trial_id"] == "trial_0001"


def test_trial_aggregate_status_projection_uses_trial_status_not_verifier_status() -> None:
    trial = _trial("trial_0001", status="failed", failure_code="client_execution.tool_retry_budget_exhausted")
    trial.verifier_result["status"] = "skipped"

    aggregate = aggregate_trial_results([trial], aggregation="single")

    assert aggregate.samples_jsonl_projection["status"] == {
        "value": "failed",
        "source_trial_id": "trial_0001",
    }


def test_session_factory_attaches_artifact_sink_for_kit_owned_artifacts(tmp_path: Path) -> None:
    sink = RuntimeArtifactSink(base_dir=str(tmp_path))
    plan = _minimal_plan(
        trial_policy={"trials": 1},
        verifier_environment_policy="reuse",
        judge_binding=JudgeBinding(judge_mode="runtime_verifier"),
    )

    session = AgentRuntimeSessionFactory(sink).create(
        plan=plan,
        sample={"id": "sample-1"},
        payload={"execution_context": {"run_id": "run-1", "task_id": "task-1"}},
    )

    assert session.artifact_sink is sink


def test_trial_aggregate_contains_full_metric_and_failure_fields() -> None:
    aggregate = aggregate_trial_results(
        [
            _trial("trial_0001", status="completed", score=0.25, resolved=False),
            _trial("trial_0002", status="completed", score=0.75, passed=True),
            _trial("trial_0003", status="failed", failure_code="trial.execution_failed"),
        ],
        aggregation="all",
    )
    payload = aggregate.to_dict()

    assert set(payload) == {
        "aggregation",
        "trial_count",
        "completed_trial_count",
        "failed_trial_count",
        "primary_trial_id",
        "score_mean",
        "score_min",
        "score_max",
        "pass_count",
        "pass_rate",
        "samples_jsonl_projection",
        "metric_projection",
        "failure_rollup",
        "trial_result_refs",
    }
    assert payload["trial_count"] == 3
    assert payload["completed_trial_count"] == 2
    assert payload["failed_trial_count"] == 1
    assert payload["score_mean"] == 0.5
    assert payload["score_min"] == 0.25
    assert payload["score_max"] == 0.75
    assert payload["pass_count"] == 1
    assert payload["pass_rate"] == 0.5
    assert payload["metric_projection"]["primary_trial_id"] == "trial_0001"
    assert payload["failure_rollup"]["failure_codes"] == {"trial.execution_failed": 1}
    assert [ref["name"] for ref in payload["trial_result_refs"]] == [
        "trial_result.json",
        "trial_result.json",
        "trial_result.json",
    ]


def test_default_trials_one_generates_only_trial_0001(tmp_path: Path) -> None:
    manager = _manager(tmp_path)

    aggregate = manager.run(
        run_id="run-1",
        task_id="task-1",
        sample_id="sample-1",
        trial_policy={},
    )

    assert [result.trial_id for result in aggregate.trial_results] == ["trial_0001"]
    assert aggregate.trial_count == 1
    assert (tmp_path / "run-1" / "artifacts/task-1/sample-1/trials/trial_0001/infra/trace.jsonl").is_file()
    assert not (tmp_path / "run-1" / "artifacts/task-1/sample-1/trials/trial_0002").exists()


def test_trials_two_run_sequentially_with_independent_lease_trace_and_artifact(tmp_path: Path) -> None:
    manager = _manager(tmp_path)

    aggregate = manager.run(
        run_id="run-2",
        task_id="task-1",
        sample_id="sample-1",
        trial_policy={"trials": 2},
    )

    assert [result.trial_id for result in aggregate.trial_results] == ["trial_0001", "trial_0002"]
    assert [result.environment_descriptor["lease_id"] for result in aggregate.trial_results] == [
        "scheduler-lease-1",
        "scheduler-lease-2",
    ]
    for trial_id in ("trial_0001", "trial_0002"):
        assert (tmp_path / "run-2" / f"artifacts/task-1/sample-1/trials/{trial_id}/infra/trace.jsonl").is_file()
        assert (tmp_path / "run-2" / f"artifacts/task-1/sample-1/trials/{trial_id}/agent/scheduler_result.json").is_file()
        assert (tmp_path / "run-2" / f"artifacts/task-1/sample-1/trials/{trial_id}/verifier/verifier_result.json").is_file()

    aggregate_payload = json.loads(
        (tmp_path / "run-2" / "artifacts/task-1/sample-1/infra/trial_aggregate.json").read_text(encoding="utf-8")
    )
    assert aggregate_payload["trial_count"] == 2
    assert aggregate_payload["samples_jsonl_projection"]["primary_trial_id"] == "trial_0001"


@pytest.mark.parametrize(
    ("verifier_policy", "expected_scheduler_acquires", "expected_verifier_acquires"),
    [
        ("reuse", 3, 0),
        ("fresh_from_profile", 3, 3),
    ],
)
def test_environment_acquire_count_matches_resource_table(
    tmp_path: Path,
    verifier_policy: str,
    expected_scheduler_acquires: int,
    expected_verifier_acquires: int,
) -> None:
    calls = {"scheduler_acquire": 0, "verifier_acquire": 0}
    manager = _manager(tmp_path, calls=calls, verifier_environment_policy=verifier_policy)

    manager.run(
        run_id=f"run-{verifier_policy}",
        task_id="task-1",
        sample_id="sample-1",
        trial_policy={"trials": 3},
    )

    assert calls["scheduler_acquire"] == expected_scheduler_acquires
    assert calls["verifier_acquire"] == expected_verifier_acquires


def test_executor_trials_two_run_sequentially_with_independent_leases_and_v2_artifacts(tmp_path: Path) -> None:
    resource_manager = _ExecutorResourceManager()
    scheduler = _ExecutorScheduler()
    verifier = _ExecutorVerifierRunner()
    executor = _executor(tmp_path, resource_manager=resource_manager, scheduler=scheduler, verifier=verifier)

    result = asyncio.run(
        executor.aexecute(
            sample={"id": "sample-1", "prompt": "demo"},
            payload={
                "execution_context": {
                    "run_id": "run-executor",
                    "task_id": "task-1",
                    "sample_id": "sample-1",
                }
            },
        )
    )

    assert resource_manager.acquired_trial_ids == ["trial_0001", "trial_0002"]
    assert resource_manager.released_lease_ids == ["lease-1", "lease-2"]
    assert scheduler.trial_ids == ["trial_0001", "trial_0002"]
    assert verifier.trial_ids == ["trial_0001", "trial_0002"]
    assert len(set(scheduler.session_ids)) == 2
    assert len(set(verifier.session_ids)) == 2
    assert result["answer"] == "trial_0001"
    assert result["agent_eval"]["trial_aggregate"]["trial_count"] == 2
    assert result["agent_eval"]["trial_aggregate"]["primary_trial_id"] == "trial_0001"
    assert result["agent_eval"]["trial_aggregate"]["score_mean"] == 0.5

    for trial_id in ("trial_0001", "trial_0002"):
        trial_root = tmp_path / "run-executor" / f"artifacts/task-1/sample-1/trials/{trial_id}"
        scheduler_layout = scheduler.artifact_layouts[trial_id]
        verifier_layout = verifier.artifact_layouts[trial_id]
        assert scheduler_layout["sample_root"].endswith(f"artifacts/task-1/sample-1/trials/{trial_id}")
        assert scheduler_layout["runtime_metadata"].endswith(f"trials/{trial_id}/infra/trial_result.json")
        assert scheduler_layout["verifier_result"].endswith(f"trials/{trial_id}/verifier/verifier_result.json")
        assert verifier_layout["sample_root"].endswith(f"artifacts/task-1/sample-1/trials/{trial_id}")
        assert (trial_root / "infra/trace.jsonl").is_file()
        assert (trial_root / "infra/trial_result.json").is_file()
        assert (trial_root / "agent/scheduler_result.json").is_file()
        assert (trial_root / "verifier/verifier_result.json").is_file()
        trace_events = [
            json.loads(line)
            for line in (trial_root / "infra/trace.jsonl").read_text(encoding="utf-8").splitlines()
        ]
        event_types = [event["event_type"] for event in trace_events]
        assert "verifier.result" in event_types
        assert "trial.end" in event_types
        verifier_event = next(event for event in trace_events if event["event_type"] == "verifier.result")
        trial_end_event = next(event for event in trace_events if event["event_type"] == "trial.end")
        assert verifier_event["actor"] == "verifier"
        assert verifier_event["payload"]["metric"]["score"] == (1.0 if trial_id == "trial_0002" else 0.0)
        assert verifier_event["artifact_refs"]
        assert trial_end_event["payload"]["status"] == "completed"
        assert trial_end_event["payload"]["failure"] is None

    aggregate_payload = json.loads(
        (tmp_path / "run-executor" / "artifacts/task-1/sample-1/infra/trial_aggregate.json").read_text(
            encoding="utf-8"
        )
    )
    assert aggregate_payload["samples_jsonl_projection"]["primary_trial_id"] == "trial_0001"
    sample_infra = tmp_path / "run-executor" / "artifacts/task-1/sample-1/infra"
    assert (sample_infra / "effective_config.json").is_file()
    assert (sample_infra / "sample_record.json").is_file()
    assert not (tmp_path / "run-executor" / "samples" / "runtime").exists()


def test_executor_single_trial_returns_trial_aggregate_projection(tmp_path: Path) -> None:
    resource_manager = _ExecutorResourceManager()
    scheduler = _ExecutorScheduler()
    verifier = _ExecutorVerifierRunner()
    executor = _executor(
        tmp_path,
        resource_manager=resource_manager,
        scheduler=scheduler,
        verifier=verifier,
        trial_policy={"trials": 1},
    )

    result = asyncio.run(
        executor.aexecute(
            sample={"id": "sample-1", "prompt": "demo"},
            payload={
                "execution_context": {
                    "run_id": "run-single-projection",
                    "task_id": "task-1",
                    "sample_id": "sample-1",
                }
            },
        )
    )

    aggregate = result["agent_eval"]["trial_aggregate"]
    assert aggregate["trial_count"] == 1
    assert aggregate["samples_jsonl_projection"]["status"] == {
        "value": "completed",
        "source_trial_id": "trial_0001",
    }
    assert result["agent_eval"]["trial_results"][0]["trial_id"] == "trial_0001"


def test_executor_multi_trial_preserves_framework_loop_tooling_failure_code(tmp_path: Path) -> None:
    resource_manager = _ExecutorResourceManager()
    scheduler = _ExecutorToolingFailureScheduler("client_execution.tool_protocol_parse_error")
    verifier = _ExecutorVerifierRunner()
    executor = _executor(tmp_path, resource_manager=resource_manager, scheduler=scheduler, verifier=verifier)

    result = asyncio.run(
        executor.aexecute(
            sample={"id": "sample-1", "prompt": "demo"},
            payload={
                "execution_context": {
                    "run_id": "run-tooling-failure",
                    "task_id": "task-1",
                    "sample_id": "sample-1",
                }
            },
        )
    )

    assert [
        trial["failure"]["failure_code"]
        for trial in result["agent_eval"]["trial_results"]
    ] == [
        "client_execution.tool_protocol_parse_error",
        "client_execution.tool_protocol_parse_error",
    ]
    for trial_id in ("trial_0001", "trial_0002"):
        trace_path = (
            tmp_path
            / "run-tooling-failure"
            / f"artifacts/task-1/sample-1/trials/{trial_id}/infra/trace.jsonl"
        )
        trace_events = [json.loads(line) for line in trace_path.read_text(encoding="utf-8").splitlines()]
        trial_end_event = next(event for event in trace_events if event["event_type"] == "trial.end")
        assert trial_end_event["payload"]["status"] == "failed"
        assert trial_end_event["payload"]["failure"]["failure_code"] == "client_execution.tool_protocol_parse_error"


def test_executor_release_failure_records_actual_exception_details(tmp_path: Path) -> None:
    resource_manager = _ExecutorResourceManager(
        release_failures={"lease-1": RuntimeError("cleanup exploded")}
    )
    scheduler = _ExecutorScheduler()
    verifier = _ExecutorVerifierRunner()
    executor = _executor(tmp_path, resource_manager=resource_manager, scheduler=scheduler, verifier=verifier)

    result = asyncio.run(
        executor.aexecute(
            sample={"id": "sample-1", "prompt": "demo"},
            payload={"execution_context": {"run_id": "run-cleanup", "task_id": "task-1", "sample_id": "sample-1"}},
        )
    )

    trial_failure = result["agent_eval"]["trial_results"][0]["failure"]
    assert trial_failure["summary"] == "cleanup exploded"
    assert trial_failure["normalized_signals"]["exception_type"] == "RuntimeError"


def test_executor_release_failure_is_suppressed_when_trial_already_failed(tmp_path: Path) -> None:
    resource_manager = _ExecutorResourceManager(
        release_failures={"lease-1": RuntimeError("cleanup after verifier failed")}
    )
    scheduler = _ExecutorScheduler()
    verifier = _ExecutorVerifierRunner(fail_trial_ids={"trial_0001"})
    executor = _executor(tmp_path, resource_manager=resource_manager, scheduler=scheduler, verifier=verifier)

    result = asyncio.run(
        executor.aexecute(
            sample={"id": "sample-1", "prompt": "demo"},
            payload={
                "execution_context": {
                    "run_id": "run-suppressed-cleanup",
                    "task_id": "task-1",
                    "sample_id": "sample-1",
                }
            },
        )
    )

    trial_failure = result["agent_eval"]["trial_results"][0]["failure"]
    cleanup_failures = trial_failure["details"]["cleanup_failures"]
    assert trial_failure["summary"] == "verifier failed for trial_0001"
    assert cleanup_failures[0]["summary"] == "cleanup after verifier failed"
    assert cleanup_failures[0]["normalized_signals"]["exception_type"] == "RuntimeError"


def test_executor_fresh_verifier_acquire_overrides_scheduler_profile(tmp_path: Path) -> None:
    resource_manager = _ExecutorResourceManager()
    scheduler = _ExecutorScheduler()
    verifier = _ExecutorVerifierRunner()
    executor = _executor(
        tmp_path,
        resource_manager=resource_manager,
        scheduler=scheduler,
        verifier=verifier,
        verifier_environment_policy="fresh_from_profile",
        verifier_environment_profile_id="verifier-profile",
        resource_plan=_resource_plan(profile_id="scheduler-profile"),
    )

    asyncio.run(
        executor.aexecute(
            sample={
                "id": "sample-1",
                "prompt": "demo",
                "metadata": {"environment_overrides": {"template_name": "scheduler-template"}},
            },
            payload={
                "execution_context": {
                    "run_id": "run-fresh-verifier",
                    "task_id": "task-1",
                    "sample_id": "sample-1",
                }
            },
        )
    )

    verifier_acquire_plan = resource_manager.acquire_resource_plans[1]
    verifier_acquire_sample = resource_manager.acquire_samples[1]
    assert verifier_acquire_plan["environment_profile"]["profile_id"] == "verifier-profile"
    assert verifier_acquire_plan["resource_kind"] == "docker"
    assert "sandbox" not in verifier_acquire_sample


def test_executor_fresh_verifier_acquire_preserves_sample_environment_overrides_when_provider_resolver_exists(
    tmp_path: Path,
) -> None:
    resource_manager = _ExecutorResourceManager()
    scheduler = _ExecutorScheduler()
    verifier = _ExecutorVerifierRunner()
    executor = _executor(
        tmp_path,
        resource_manager=resource_manager,
        scheduler=scheduler,
        verifier=verifier,
        verifier_environment_policy="fresh_from_profile",
        verifier_environment_profile_id="verifier-profile",
        resource_plan={
            **_resource_plan(profile_id="scheduler-profile"),
            "provider_config_resolver": lambda **kwargs: dict(kwargs["base_provider_config"]),
        },
    )

    asyncio.run(
        executor.aexecute(
            sample={
                "id": "sample-1",
                "prompt": "demo",
                "metadata": {
                    "environment_overrides": {"image_uri": "jefzda/sweap-images:sample-1"},
                },
            },
            payload={
                "execution_context": {
                    "run_id": "run-fresh-verifier-provider-resolver",
                    "task_id": "task-1",
                    "sample_id": "sample-1",
                }
            },
        )
    )

    verifier_acquire_plan = resource_manager.acquire_resource_plans[1]
    verifier_acquire_sample = resource_manager.acquire_samples[1]
    assert verifier_acquire_plan["environment_profile"]["profile_id"] == "verifier-profile"
    assert verifier_acquire_plan["resource_kind"] == "docker"
    assert (
        verifier_acquire_sample["metadata"]["environment_overrides"]["image_uri"]
        == "jefzda/sweap-images:sample-1"
    )
    assert "sandbox" not in verifier_acquire_sample


def test_executor_fresh_verifier_acquire_uses_verifier_profile_config(tmp_path: Path) -> None:
    resource_manager = _ExecutorResourceManager()
    scheduler = _ExecutorScheduler()
    verifier = _ExecutorVerifierRunner()
    executor = _executor(
        tmp_path,
        resource_manager=resource_manager,
        scheduler=scheduler,
        verifier=verifier,
        verifier_environment_policy="fresh_from_profile",
        verifier_environment_profile_id="swebench-e2b-wrapper",
        environment_provider="e2b",
        kit_entry=type(
            "KitEntry",
            (),
            {
                "environment_profiles": {
                    "swebench-e2b-wrapper": {
                        "asset_dir": "src/gage_eval/agent_eval_kits/swebench/environment/e2b",
                        "config": {
                            "template_id": "gage-swebench-pro-wrapper",
                            "request_timeout_s": 30.0,
                        },
                        "capabilities": {"supports_upload_download": True},
                    }
                }
            },
        )(),
        resource_plan=_resource_plan(
            resource_kind="e2b",
            profile_id="scheduler-e2b-profile",
            provider_config={
                "template_id": "scheduler-custom-template",
                "request_timeout_s": 5.0,
            },
        ),
    )

    asyncio.run(
        executor.aexecute(
            sample={
                "id": "sample-1",
                "prompt": "demo",
                "metadata": {"environment_overrides": {"template_id": "sample-scheduler-template"}},
            },
            payload={
                "execution_context": {
                    "run_id": "run-fresh-verifier-e2b-config",
                    "task_id": "task-1",
                    "sample_id": "sample-1",
                }
            },
        )
    )

    verifier_acquire_plan = resource_manager.acquire_resource_plans[1]
    verifier_profile = verifier_acquire_plan["environment_profile"]
    verifier_config = verifier_acquire_plan["provider_config"]
    assert verifier_profile["profile_id"] == "swebench-e2b-wrapper"
    assert verifier_profile["provider"] == "e2b"
    assert verifier_config["template_id"] == "gage-swebench-pro-wrapper"
    assert verifier_config["request_timeout_s"] == 30.0
    assert verifier_profile["config"]["template_id"] == "gage-swebench-pro-wrapper"

    verifier_acquire_sample = resource_manager.acquire_samples[1]
    assert "sandbox" not in verifier_acquire_sample


def test_executor_single_trial_fresh_verifier_acquires_verifier_profile(tmp_path: Path) -> None:
    resource_manager = _ExecutorResourceManager()
    scheduler = _ExecutorScheduler()
    verifier = _ExecutorVerifierRunner()
    executor = _executor(
        tmp_path,
        resource_manager=resource_manager,
        scheduler=scheduler,
        verifier=verifier,
        verifier_environment_policy="fresh_from_profile",
        verifier_environment_profile_id="verifier-profile",
        trial_policy={"trials": 1},
        resource_plan=_resource_plan(profile_id="scheduler-profile"),
    )

    asyncio.run(
        executor.aexecute(
            sample={"id": "sample-1", "prompt": "demo"},
            payload={
                "execution_context": {
                    "run_id": "run-single-fresh-verifier",
                    "task_id": "task-1",
                    "sample_id": "sample-1",
                }
            },
        )
    )

    assert len(resource_manager.acquire_resource_plans) == 2
    verifier_acquire_plan = resource_manager.acquire_resource_plans[1]
    assert verifier_acquire_plan["environment_profile"]["profile_id"] == "verifier-profile"
    assert verifier_acquire_plan["resource_kind"] == "docker"


def test_executor_carries_scheduler_submission_patch_into_fresh_verifier(tmp_path: Path) -> None:
    scheduler_sandbox = _SubmissionPatchSandbox("diff --git a/from-scheduler b/from-scheduler\n")
    verifier_sandbox = _SubmissionPatchSandbox("")
    resource_manager = _ExecutorResourceManager(
        sandbox_handles=[
            _SandboxHandleStub(scheduler_sandbox),
            _SandboxHandleStub(verifier_sandbox),
        ]
    )
    verifier = _PatchCapturingVerifierRunner()
    executor = _executor(
        tmp_path,
        resource_manager=resource_manager,
        scheduler=_ExecutorScheduler(),
        verifier=verifier,
        verifier_environment_policy="fresh_from_profile",
        verifier_environment_profile_id="verifier-profile",
        trial_policy={"trials": 1},
        benchmark_kit_id="swebench",
    )

    asyncio.run(
        executor.aexecute(
            sample={"id": "sample-1", "prompt": "demo"},
            payload={
                "execution_context": {
                    "run_id": "run-carry-submission-patch",
                    "task_id": "task-1",
                    "sample_id": "sample-1",
                }
            },
        )
    )

    assert verifier.patch_content == "diff --git a/from-scheduler b/from-scheduler\n"


def test_executor_verifier_preflight_skips_fresh_verifier_acquire(tmp_path: Path) -> None:
    resource_manager = _ExecutorResourceManager()
    verifier = _PreflightOnlyVerifierRunner()
    executor = _executor(
        tmp_path,
        resource_manager=resource_manager,
        scheduler=_ExecutorScheduler(),
        verifier=verifier,
        verifier_environment_policy="fresh_from_profile",
        verifier_environment_profile_id="verifier-profile",
        trial_policy={"trials": 1},
        benchmark_kit_id="swebench",
    )

    asyncio.run(
        executor.aexecute(
            sample={"id": "sample-1", "prompt": "demo"},
            payload={
                "execution_context": {
                    "run_id": "run-preflight-fresh-verifier",
                    "task_id": "task-1",
                    "sample_id": "sample-1",
                }
            },
        )
    )

    assert verifier.preflight_calls == ["trial_0001"]
    assert len(resource_manager.acquire_resource_plans) == 1


def test_executor_skips_verifier_when_scheduler_result_failed(tmp_path: Path) -> None:
    resource_manager = _ExecutorResourceManager()
    scheduler = _ExecutorFailedResultScheduler("client_execution.tool_retry_budget_exhausted")
    verifier = _ExecutorVerifierRunner()
    recorder = InMemoryRecorder(run_id="run-scheduler-failed", min_flush_events=1)
    trace = ObservabilityTrace(recorder=recorder, run_id="run-scheduler-failed")
    executor = _executor(
        tmp_path,
        resource_manager=resource_manager,
        scheduler=scheduler,
        verifier=verifier,
        verifier_environment_policy="fresh_from_profile",
        verifier_environment_profile_id="verifier-profile",
        trial_policy={"trials": 1},
    )

    result = asyncio.run(
        executor.aexecute(
            sample={"id": "sample-1", "prompt": "demo"},
            payload={
                "execution_context": {
                    "run_id": "run-scheduler-failed",
                    "task_id": "task-1",
                    "sample_id": "sample-1",
                }
            },
            trace=trace,
        )
    )
    trace.flush()

    judge_payload = result["runtime_judge_outcome"]["verifier_result"]["payload"]
    assert judge_payload["failure_code"] == "verifier.skipped_due_to_scheduler_failure"
    assert judge_payload["failure_reason"] == "client_execution.tool_retry_budget_exhausted"
    assert verifier.trial_ids == []
    assert len(resource_manager.acquire_resource_plans) == 1
    verifier_result = json.loads(
        (
            tmp_path
            / "run-scheduler-failed/artifacts/task-1/sample-1/trials/trial_0001/verifier/verifier_result.json"
        ).read_text(encoding="utf-8")
    )
    assert verifier_result["verifier_result"]["status"] == "skipped"
    sample_record = json.loads(
        (tmp_path / "run-scheduler-failed/artifacts/task-1/sample-1/infra/sample_record.json").read_text(
            encoding="utf-8"
        )
    )
    assert sample_record["status"] == "failed"
    assert sample_record["failure"]["failure_code"] == "client_execution.tool_retry_budget_exhausted"
    sample_failed = [event for event in recorder.buffered_events() if event["event"] == "sample.failed"]
    assert sample_failed
    assert sample_failed[0]["payload"]["failure_code"] == "client_execution.tool_retry_budget_exhausted"
    assert sample_failed[0]["payload"]["verifier_failure_code"] == "verifier.skipped_due_to_scheduler_failure"


def test_default_verifier_runner_uses_adapter_preflight() -> None:
    class _Adapter:
        judge_source = "test.adapter"

        def preflight(self, verifier_input: VerifierInput) -> VerifierResult:
            assert "environment_lease" not in verifier_input.runtime_context
            return VerifierResult(
                status="failed",
                payload={
                    "status": "failed",
                    "resolved": False,
                    "score": 0.0,
                    "failure_code": "config.kit_schema.validation_failed",
                },
            )

    plan = _minimal_plan(
        trial_policy={"trials": 1},
        verifier_environment_policy="fresh_from_profile",
        judge_binding=JudgeBinding(
            judge_mode="runtime_verifier",
            verifier_resource_refs={"adapter": _Adapter()},
        ),
    )
    session = AgentRuntimeSession(
        session_id="session-1",
        run_id="run-1",
        task_id="task-1",
        sample_id="sample-1",
        benchmark_kit_id="demo-kit",
        scheduler_type="framework_loop",
        artifact_layout={"verifier_result": "verifier/result.json"},
    )
    scheduler_result = SchedulerResult(
        scheduler_type="framework_loop",
        benchmark_kit_id="demo-kit",
        status="completed",
        agent_output={},
    )

    outcome = DefaultVerifierRunner().preflight(
        plan=plan,
        session=session,
        sample={"id": "sample-1"},
        scheduler_result=scheduler_result,
    )

    assert outcome is not None
    assert outcome.verifier_result.status == "failed"
    assert outcome.judge_output["failure_code"] == "config.kit_schema.validation_failed"


def test_default_verifier_runner_preserves_prepared_environment_lease() -> None:
    prepared_environment_lease = object()

    class _Adapter:
        judge_source = "test.adapter"

        def run(self, verifier_input: VerifierInput) -> VerifierResult:
            assert verifier_input.runtime_context["environment_lease"] is prepared_environment_lease
            return VerifierResult(
                status="completed",
                payload={"status": "completed", "resolved": True, "score": 1.0},
            )

    plan = _minimal_plan(
        trial_policy={"trials": 1},
        verifier_environment_policy="reuse",
        judge_binding=JudgeBinding(
            judge_mode="runtime_verifier",
            verifier_resource_refs={"adapter": _Adapter()},
        ),
    )
    session = AgentRuntimeSession(
        session_id="session-1",
        run_id="run-1",
        task_id="task-1",
        sample_id="sample-1",
        benchmark_kit_id="demo-kit",
        scheduler_type="framework_loop",
        artifact_layout={"verifier_result": "verifier/result.json"},
    )
    session.resource_lease = ResourceLease(
        lease_id="resource-only",
        resource_kind="docker",
        profile_id="scheduler-profile",
        lifecycle="per_sample",
    )
    session.runtime_context["environment_lease"] = prepared_environment_lease
    scheduler_result = SchedulerResult(
        scheduler_type="framework_loop",
        benchmark_kit_id="demo-kit",
        status="completed",
        agent_output={"answer": "ok"},
    )

    outcome = DefaultVerifierRunner().run(
        plan=plan,
        session=session,
        sample={"id": "sample-1"},
        scheduler_result=scheduler_result,
        sandbox_provider=None,
    )

    assert outcome.judge_output["resolved"] is True


def _resource_plan(
    *,
    resource_kind: str = "docker",
    profile_id: str = "scheduler-profile",
    provider_config: dict[str, Any] | None = None,
    resources: dict[str, Any] | None = None,
    startup_env: dict[str, Any] | None = None,
    lifecycle: str = "per_sample",
) -> dict[str, Any]:
    config = dict(provider_config or {})
    return {
        "resource_kind": resource_kind,
        "environment_profile": {
            "profile_id": profile_id,
            "provider": resource_kind,
            "config": config,
            "resources": dict(resources or {}),
            "startup_env": dict(startup_env or {}),
            "lifecycle": lifecycle,
        },
        "provider_config": config,
        "resources": dict(resources or {}),
        "startup_env": dict(startup_env or {}),
        "lifecycle": lifecycle,
    }


def _minimal_plan(
    *,
    trial_policy: dict[str, Any],
    verifier_environment_policy: str,
    judge_binding: JudgeBinding,
) -> CompiledRuntimePlan:
    runtime_spec = AgentRuntimeSpec(
        agent_runtime_id="demo-runtime",
        benchmark_kit_id="demo-kit",
        scheduler_type="framework_loop",
    )
    return CompiledRuntimePlan(
        run_id="run-1",
        dut_id="dut-1",
        agent_id="agent-1",
        env_id="env-1",
        benchmark_id="benchmark-1",
        trial_policy=trial_policy,
        kit_id="demo-kit",
        kit_entry=None,
        kit_config={},
        agent_config={},
        scheduler_type="framework_loop",
        scheduler_config={},
        environment_provider="docker",
        environment_profile_id="scheduler-profile",
        environment_profile={},
        lifecycle="per_sample",
        provider_config={},
        startup_env={},
        resources={},
        verifier_environment_policy=verifier_environment_policy,
        verifier_environment_profile_id=None,
        workflow_bundle=SchedulerWorkflowBundle(
            bundle_id="demo-workflow",
            benchmark_kit_id="demo-kit",
            scheduler_type="framework_loop",
        ),
        tool_registry=None,
        tool_provider_adapter=None,
        verifier_adapter=None,
        artifact_sink=None,
        plan_id="demo-plan",
        runtime_spec=runtime_spec,
        scheduler_handle=None,
        kit_runtime_ref=None,
        judge_binding=judge_binding,
        resource_plan=_resource_plan(),
    )


def _manager(
    tmp_path: Path,
    *,
    calls: dict[str, int] | None = None,
    verifier_environment_policy: str = "reuse",
) -> TrialManager:
    counters = calls if calls is not None else {"scheduler_acquire": 0, "verifier_acquire": 0}

    def acquire_scheduler(*, trial_id: str, **_: Any) -> dict[str, Any]:
        counters["scheduler_acquire"] += 1
        return {
            "lease_id": f"scheduler-lease-{counters['scheduler_acquire']}",
            "profile_id": "scheduler-profile",
            "trial_id": trial_id,
        }

    def acquire_verifier(*, trial_id: str, **_: Any) -> dict[str, Any]:
        counters["verifier_acquire"] += 1
        return {
            "lease_id": f"verifier-lease-{counters['verifier_acquire']}",
            "profile_id": "verifier-profile",
            "trial_id": trial_id,
        }

    def run_scheduler(*, trial_id: str, scheduler_lease: dict[str, Any], **_: Any) -> dict[str, Any]:
        return {
            "status": "completed",
            "trial_id": trial_id,
            "lease_id": scheduler_lease["lease_id"],
            "agent_output": {"answer": trial_id},
        }

    def run_verifier(*, trial_id: str, scheduler_result: dict[str, Any], **_: Any) -> dict[str, Any]:
        score = 1.0 if trial_id.endswith("2") else 0.0
        return {
            "status": "completed",
            "trial_id": trial_id,
            "score": score,
            "resolved": score == 1.0,
            "scheduler_lease_id": scheduler_result["lease_id"],
        }

    return TrialManager(
        artifact_sink=RuntimeArtifactSink(base_dir=str(tmp_path)),
        scheduler_acquire=acquire_scheduler,
        scheduler_run=run_scheduler,
        verifier_run=run_verifier,
        verifier_acquire=acquire_verifier,
        verifier_environment_policy=verifier_environment_policy,
    )


class _ExecutorResourceManager:
    def __init__(
        self,
        *,
        release_failures: dict[str, BaseException] | None = None,
        sandbox_handles: list[Any] | None = None,
    ) -> None:
        self.acquired_trial_ids: list[str] = []
        self.released_lease_ids: list[str] = []
        self.acquire_resource_plans: list[dict[str, Any]] = []
        self.acquire_samples: list[dict[str, Any]] = []
        self._release_failures = dict(release_failures or {})
        self._sandbox_handles = list(sandbox_handles or [])

    def acquire(self, session, *, resource_plan, trace=None, sample=None) -> RuntimeLeaseBinding:
        trial_id = str(session.runtime_context["trial_id"])
        self.acquired_trial_ids.append(trial_id)
        self.acquire_resource_plans.append(dict(resource_plan or {}))
        captured_sample = dict(sample or {})
        self.acquire_samples.append(captured_sample)
        lease = ResourceLease(
            lease_id=f"lease-{len(self.acquired_trial_ids)}",
            resource_kind="docker",
            profile_id="profile",
            lifecycle="per_sample",
            metadata={"trial_id": trial_id},
        )
        sandbox_handle = (
            self._sandbox_handles[len(self.acquired_trial_ids) - 1]
            if len(self.acquired_trial_ids) <= len(self._sandbox_handles)
            else None
        )
        environment_lease = None
        if sandbox_handle is not None:
            environment_lease = EnvironmentLease(
                lease_id=f"env-{lease.lease_id}",
                environment=sandbox_handle.sandbox,
                provider="docker",
                profile_id="profile",
                lifecycle="per_sample",
                exclusive=True,
            )
        return RuntimeLeaseBinding(resource_lease=lease, environment_lease=environment_lease)

    def release(self, binding: RuntimeLeaseBinding) -> None:
        assert binding.resource_lease is not None
        self.released_lease_ids.append(binding.resource_lease.lease_id)
        failure = self._release_failures.get(binding.resource_lease.lease_id)
        if failure is not None:
            raise failure


class _ExecutorRuntime:
    def bootstrap(self, *, session, sample, payload, sandbox_provider) -> dict[str, Any]:
        return {"runtime_context": {"bootstrapped_trial_id": session.runtime_context["trial_id"]}}


class _ExecutorScheduler:
    def __init__(self) -> None:
        self.trial_ids: list[str] = []
        self.session_ids: list[str] = []
        self.artifact_layouts: dict[str, dict[str, str]] = {}

    async def arun(self, *, session, sample, payload, workflow_bundle, sandbox_provider) -> SchedulerResult:
        trial_id = str(session.runtime_context["trial_id"])
        self.trial_ids.append(trial_id)
        self.session_ids.append(session.session_id)
        self.artifact_layouts[trial_id] = dict(session.artifact_layout)
        return SchedulerResult(
            scheduler_type="framework_loop",
            benchmark_kit_id="demo-kit",
            status="completed",
            agent_output={"answer": trial_id, "trial_id": trial_id},
            runtime_state={"trial_id": trial_id},
        )


class _ExecutorToolingFailureScheduler:
    def __init__(self, failure_code: str) -> None:
        self.failure_code = failure_code

    async def arun(self, *, session, sample, payload, workflow_bundle, sandbox_provider) -> SchedulerResult:
        del sample, payload, workflow_bundle, sandbox_provider
        raise FailureEnvelopeError(
            FailureEnvelope(
                failure_domain="client_execution",
                failure_stage="run_scheduler",
                failure_code=self.failure_code,
                component_kind="provider",
                component_id="demo-workflow.tooling",
                owner="runtime_scheduler_core",
                retryable=False,
                summary="tooling failed",
                first_bad_step="framework_loop.runtime_tooling",
                suspect_files=("src/gage_eval/agent_runtime/tooling/provider_adapters.py",),
                details={"trial_id": session.runtime_context["trial_id"]},
            )
        )


class _ExecutorFailedResultScheduler:
    def __init__(self, failure_code: str) -> None:
        self.failure_code = failure_code

    async def arun(self, *, session, sample, payload, workflow_bundle, sandbox_provider) -> SchedulerResult:
        del sample, payload, workflow_bundle, sandbox_provider
        return SchedulerResult(
            scheduler_type="framework_loop",
            benchmark_kit_id="demo-kit",
            status="failed",
            agent_output={
                "answer": "",
                "failure_code": self.failure_code,
                "failure_reason": "tool retry budget exhausted",
                "trial_id": session.runtime_context["trial_id"],
            },
        )


class _ExecutorVerifierRunner:
    def __init__(self, *, fail_trial_ids: set[str] | None = None) -> None:
        self.trial_ids: list[str] = []
        self.session_ids: list[str] = []
        self.artifact_layouts: dict[str, dict[str, str]] = {}
        self._fail_trial_ids = set(fail_trial_ids or set())

    def run(self, *, plan, session, sample, scheduler_result, sandbox_provider=None, environment_lease=None) -> RuntimeJudgeOutcome:
        del environment_lease
        trial_id = str(session.runtime_context["trial_id"])
        self.trial_ids.append(trial_id)
        self.session_ids.append(session.session_id)
        self.artifact_layouts[trial_id] = dict(session.artifact_layout)
        if trial_id in self._fail_trial_ids:
            raise ValueError(f"verifier failed for {trial_id}")
        score = 1.0 if trial_id == "trial_0002" else 0.0
        verifier_input = VerifierInput(
            benchmark_kit_id="demo-kit",
            scheduler_type="framework_loop",
            sample_id=session.sample_id,
            sample=sample,
            scheduler_result=scheduler_result.to_dict(),
        )
        return RuntimeJudgeOutcome(
            verifier_input=verifier_input,
            verifier_result=VerifierResult(status="completed", payload={"score": score, "resolved": score == 1.0}),
            judge_output={"status": "completed", "score": score, "resolved": score == 1.0},
            persisted_path=session.artifact_layout["verifier_result"],
        )

    def build_failed_outcome(self, *, plan, session, sample, failure) -> RuntimeJudgeOutcome:
        verifier_input = VerifierInput(
            benchmark_kit_id="demo-kit",
            scheduler_type="framework_loop",
            sample_id=session.sample_id,
            sample=sample,
            scheduler_result={},
        )
        return RuntimeJudgeOutcome(
            verifier_input=verifier_input,
            verifier_result=VerifierResult(status="failed", payload={"failure": failure.to_dict()}),
            judge_output={"status": "failed", "resolved": False, "score": 0.0},
            persisted_path=session.artifact_layout["verifier_result"],
            failure=failure,
        )


class _PreflightOnlyVerifierRunner(_ExecutorVerifierRunner):
    def __init__(self) -> None:
        super().__init__()
        self.preflight_calls: list[str] = []

    def preflight(self, *, plan, session, sample, scheduler_result) -> RuntimeJudgeOutcome:
        del plan, sample
        trial_id = str(session.runtime_context["trial_id"])
        self.preflight_calls.append(trial_id)
        verifier_input = VerifierInput(
            benchmark_kit_id=session.benchmark_kit_id,
            scheduler_type=session.scheduler_type,
            sample_id=session.sample_id,
            sample={},
            scheduler_result=scheduler_result.to_dict(),
        )
        return RuntimeJudgeOutcome(
            verifier_input=verifier_input,
            verifier_result=VerifierResult(
                status="failed",
                payload={
                    "status": "failed",
                    "resolved": False,
                    "score": 0.0,
                    "failure_code": "config.kit_schema.validation_failed",
                    "failure_reason": "missing_base_commit",
                },
            ),
            judge_output={
                "status": "failed",
                "resolved": False,
                "score": 0.0,
                "failure_code": "config.kit_schema.validation_failed",
                "failure_reason": "missing_base_commit",
            },
            persisted_path=session.artifact_layout["verifier_result"],
        )

    def run(self, **_kwargs) -> RuntimeJudgeOutcome:
        raise AssertionError("verifier run should not execute after preflight failure")


class _PatchCapturingVerifierRunner(_ExecutorVerifierRunner):
    def __init__(self) -> None:
        super().__init__()
        self.patch_content = ""

    def run(self, *, plan, session, sample, scheduler_result, sandbox_provider=None, environment_lease=None) -> RuntimeJudgeOutcome:
        self.patch_content = str(scheduler_result.agent_output.get("patch_content") or "")
        return super().run(
            plan=plan,
            session=session,
            sample=sample,
            scheduler_result=scheduler_result,
            sandbox_provider=sandbox_provider,
        )


class _SubmissionPatchSandbox:
    def __init__(self, submission_patch: str) -> None:
        self.submission_patch = submission_patch

    async def read_file(self, path: str, *, max_bytes: int = 16 * 1024 * 1024) -> bytes:
        del max_bytes
        if path != "/workspace/submission.patch" or not self.submission_patch:
            raise FileNotFoundError(path)
        return self.submission_patch.encode("utf-8")

    async def write_file(self, path: str, content: bytes | str) -> None:
        del path, content
        return None

    async def exec(self, command: str, **kwargs: Any):
        del kwargs
        return type("ExecResult", (), {"exit_code": 0, "stdout": "", "stderr": "", "command": command})()


class _SandboxHandleStub:
    def __init__(self, sandbox: Any) -> None:
        self.sandbox = sandbox


def _executor(
    tmp_path: Path,
    *,
    resource_manager: _ExecutorResourceManager,
    scheduler: _ExecutorScheduler,
    verifier: _ExecutorVerifierRunner,
    verifier_environment_policy: str = "reuse",
    verifier_environment_profile_id: str | None = None,
    trial_policy: dict[str, Any] | None = None,
    resource_plan: dict[str, Any] | None = None,
    benchmark_kit_id: str = "demo-kit",
    environment_provider: str = "docker",
    kit_entry: Any | None = None,
) -> CompiledRuntimeExecutor:
    sink = RuntimeArtifactSink(base_dir=str(tmp_path))
    runtime_spec = AgentRuntimeSpec(
        agent_runtime_id="demo-runtime",
        benchmark_kit_id=benchmark_kit_id,
        scheduler_type="framework_loop",
        client_id="demo-client",
    )
    plan = CompiledRuntimePlan(
        run_id="run-executor",
        dut_id="dut-1",
        agent_id="agent-1",
        env_id="env-1",
        benchmark_id="benchmark-1",
        trial_policy=trial_policy or {"trials": 2},
        kit_id=benchmark_kit_id,
        kit_entry=kit_entry,
        kit_config={},
        agent_config={},
        scheduler_type="framework_loop",
        scheduler_config={},
        environment_provider=environment_provider,
        environment_profile_id="profile",
        environment_profile={},
        lifecycle="per_sample",
        provider_config={},
        startup_env={},
        resources={},
        verifier_environment_policy=verifier_environment_policy,
        verifier_environment_profile_id=verifier_environment_profile_id,
        workflow_bundle=SchedulerWorkflowBundle(
            bundle_id="demo-workflow",
            benchmark_kit_id=benchmark_kit_id,
            scheduler_type="framework_loop",
        ),
        tool_registry=None,
        tool_provider_adapter=None,
        verifier_adapter=None,
        artifact_sink=sink,
        plan_id="demo-plan",
        runtime_spec=runtime_spec,
        scheduler_handle=scheduler,
        kit_runtime_ref=_ExecutorRuntime(),
        judge_binding=JudgeBinding(judge_mode="runtime_verifier"),
        resource_plan=resource_plan or _resource_plan(),
    )
    return CompiledRuntimeExecutor(
        compiled_plan=plan,
        resource_manager=resource_manager,
        session_factory=AgentRuntimeSessionFactory(sink),
        verifier_runner=verifier,
        artifact_sink=sink,
        trace_emitter=type("NoopTraceEmitter", (), {
            "emit_session_start": lambda *args, **kwargs: None,
            "emit_session_end": lambda *args, **kwargs: None,
            "emit_failure": lambda *args, **kwargs: None,
        })(),
        failure_mapper=FailureMapper(),
    )
