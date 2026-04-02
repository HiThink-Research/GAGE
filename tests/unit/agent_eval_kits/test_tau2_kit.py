from __future__ import annotations

from types import SimpleNamespace

import pytest

from gage_eval.agent_eval_kits.tau2.judge_bridge import build_verifier_input
from gage_eval.agent_eval_kits.tau2.kit import build_kit
from gage_eval.agent_eval_kits.tau2.resources import build_resource_requirements
from gage_eval.agent_eval_kits.tau2.sub_workflow import finalize_result, prepare_inputs
from gage_eval.agent_runtime.artifacts.layout import ArtifactLayout
from gage_eval.agent_runtime.schedulers import SchedulerResult
from gage_eval.agent_runtime.spec import AgentRuntimeSpec, ClientSurfacePolicy, ResourcePolicy, SandboxPolicy
from gage_eval.agent_runtime.verifier.base import VerifierInput


def _build_plan() -> SimpleNamespace:
    runtime_spec = AgentRuntimeSpec(
        agent_runtime_id="codex_tau2",
        scheduler="installed_client",
        benchmark_kit_id="tau2",
        client_id="codex",
        resource_policy=ResourcePolicy(environment_kind="fake", timeout_sec=900),
        client_surface_policy=ClientSurfacePolicy(optional=("terminal", "fs")),
        sandbox_policy=SandboxPolicy(sandbox_profile_id="tau2_runtime"),
    )
    return SimpleNamespace(
        runtime_spec=runtime_spec,
        scheduler_type=runtime_spec.scheduler,
        benchmark_kit_id=runtime_spec.benchmark_kit_id,
        client_id=runtime_spec.client_id,
        sandbox_profile_id="tau2_runtime",
        role_adapter_id="dut_codex_tau2",
        required_surfaces=runtime_spec.client_surface_policy.required,
        optional_surfaces=runtime_spec.client_surface_policy.optional,
    )


def _build_artifacts() -> ArtifactLayout:
    return ArtifactLayout.for_sample("runs", "run-1", "airline_task-1__trial_0")


@pytest.mark.fast
def test_tau2_build_kit_declares_optional_shell_surfaces() -> None:
    kit = build_kit()

    assert kit.kit_id == "tau2"
    assert kit.verifier_kind == "judge_adapter"
    assert kit.required_surfaces == ()
    assert kit.optional_surfaces == ("terminal", "fs")


@pytest.mark.fast
def test_tau2_build_resource_requirements() -> None:
    requirements = build_resource_requirements({"id": "airline_task-1__trial_0"}, _build_plan())

    assert requirements["benchmark_kit_id"] == "tau2"
    assert requirements["environment_kind"] == "fake"
    assert requirements["timeout_sec"] == 900
    assert requirements["optional_surfaces"] == ("terminal", "fs")


@pytest.mark.fast
def test_tau2_prepare_inputs_uses_tau2_metadata_prompt() -> None:
    sample = {
        "id": "airline_task-1__trial_0",
        "metadata": {
            "tau2": {
                "task_id": "task-1",
                "agent_instruction": "Help the customer finish the booking.",
                "gage_instruction": "When you want to respond, use the benchmark protocol.",
                "policy": "Always verify the final itinerary.",
            }
        },
    }
    session = SimpleNamespace(
        trace=SimpleNamespace(run_id="run-1"),
        plan=_build_plan(),
        artifacts=_build_artifacts(),
        metadata={},
        resources=SimpleNamespace(metadata={}),
    )

    payload = prepare_inputs(sample, session)

    assert payload["sample_id"] == "airline_task-1__trial_0"
    assert "Help the customer finish the booking." in payload["instruction"]
    assert "Always verify the final itinerary." in payload["instruction"]
    assert payload["metadata"]["benchmark_kit_id"] == "tau2"
    assert payload["artifact_paths"]["patch_path"].endswith("submission.patch")


@pytest.mark.fast
def test_tau2_build_verifier_input_uses_runtime_state() -> None:
    sample = {
        "id": "airline_task-1__trial_0",
        "metadata": {"tau2": {"task_id": "task-1", "domain": "airline"}},
    }
    artifacts = _build_artifacts()
    scheduler_result = SchedulerResult(
        status="success",
        raw_output={"runtime_state": {"task_id": "task-1", "domain": "airline", "reward": 1.0}},
    )

    verifier_input = build_verifier_input(sample, scheduler_result, artifacts)

    assert isinstance(verifier_input, VerifierInput)
    assert verifier_input.benchmark_kit_id == "tau2"
    assert verifier_input.sample_id == "airline_task-1__trial_0"
    assert verifier_input.payload["runtime_state"]["domain"] == "airline"


@pytest.mark.fast
def test_tau2_finalize_result() -> None:
    sample = {"id": "airline_task-1__trial_0", "metadata": {"tau2": {"task_id": "task-1"}}}
    artifacts = _build_artifacts()
    scheduler_result = SchedulerResult(
        status="success",
        stdout_path=artifacts.stdout_file,
        trajectory_path=artifacts.trajectory_file,
    )

    result = finalize_result(sample, scheduler_result, artifacts)

    assert result["sample_id"] == "airline_task-1__trial_0"
    assert result["status"] == "success"
    assert result["artifact_paths"]["stdout_path"].endswith("stdout.log")
