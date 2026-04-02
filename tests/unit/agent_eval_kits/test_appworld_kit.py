from __future__ import annotations

from types import SimpleNamespace

import pytest

from gage_eval.agent_eval_kits.appworld.judge_bridge import build_verifier_input
from gage_eval.agent_eval_kits.appworld.kit import build_kit
from gage_eval.agent_eval_kits.appworld.resources import build_resource_requirements
from gage_eval.agent_eval_kits.appworld.sub_workflow import finalize_result, prepare_inputs
from gage_eval.agent_runtime.artifacts.layout import ArtifactLayout
from gage_eval.agent_runtime.schedulers import SchedulerResult
from gage_eval.agent_runtime.spec import AgentRuntimeSpec, ClientSurfacePolicy, ResourcePolicy, SandboxPolicy
from gage_eval.agent_runtime.verifier.base import VerifierInput


def _build_plan() -> SimpleNamespace:
    runtime_spec = AgentRuntimeSpec(
        agent_runtime_id="codex_appworld",
        scheduler="installed_client",
        benchmark_kit_id="appworld",
        client_id="codex",
        resource_policy=ResourcePolicy(environment_kind="docker", timeout_sec=600),
        client_surface_policy=ClientSurfacePolicy(required=("terminal", "fs", "env", "api"), optional=("mcp",)),
        sandbox_policy=SandboxPolicy(sandbox_profile_id="appworld_local"),
    )
    return SimpleNamespace(
        runtime_spec=runtime_spec,
        scheduler_type=runtime_spec.scheduler,
        benchmark_kit_id=runtime_spec.benchmark_kit_id,
        client_id=runtime_spec.client_id,
        sandbox_profile_id="appworld_local",
        role_adapter_id="dut_codex_appworld",
        required_surfaces=runtime_spec.client_surface_policy.required,
        optional_surfaces=runtime_spec.client_surface_policy.optional,
    )


def _build_artifacts() -> ArtifactLayout:
    return ArtifactLayout.for_sample("runs", "run-1", "calendar_001")


@pytest.mark.fast
def test_appworld_build_kit_declares_runtime_surfaces() -> None:
    kit = build_kit()

    assert kit.kit_id == "appworld"
    assert kit.verifier_kind == "judge_adapter"
    assert kit.required_surfaces == ("terminal", "fs", "env", "api")
    assert kit.optional_surfaces == ("mcp",)


@pytest.mark.fast
def test_appworld_build_resource_requirements() -> None:
    sample = {
        "id": "calendar_001",
        "metadata": {"appworld": {"task_id": "calendar_001"}},
    }

    requirements = build_resource_requirements(sample, _build_plan())

    assert requirements["benchmark_kit_id"] == "appworld"
    assert requirements["environment_kind"] == "docker"
    assert requirements["timeout_sec"] == 600
    assert requirements["required_surfaces"] == ("terminal", "fs", "env", "api")
    assert requirements["sample_id"] == "calendar_001"


@pytest.mark.fast
def test_appworld_prepare_inputs() -> None:
    sample = {
        "id": "calendar_001",
        "messages": [{"role": "user", "content": [{"type": "text", "text": "Cancel the calendar event."}]}],
        "workspace_root": "/workspace/appworld",
        "env": {"APPWORLD_MODE": "smoke"},
        "metadata": {"appworld": {"task_id": "calendar_001", "allowed_apps": ["calendar", "api_docs"]}},
    }
    session = SimpleNamespace(
        trace=SimpleNamespace(run_id="run-1"),
        plan=_build_plan(),
        artifacts=_build_artifacts(),
        metadata={},
        resources=SimpleNamespace(metadata={"workspace_root": "/workspace/appworld"}),
    )

    payload = prepare_inputs(sample, session)

    assert payload["sample_id"] == "calendar_001"
    assert payload["instruction"] == "Cancel the calendar event."
    assert payload["cwd"] == "/workspace/appworld"
    assert payload["env"] == {"APPWORLD_MODE": "smoke"}
    assert payload["metadata"]["benchmark_kit_id"] == "appworld"
    assert payload["artifact_paths"]["stdout_path"].endswith("stdout.log")


@pytest.mark.fast
def test_appworld_build_verifier_input_uses_runtime_handle() -> None:
    sample = {
        "id": "calendar_001",
        "metadata": {"appworld": {"task_id": "calendar_001"}},
    }
    artifacts = _build_artifacts()
    scheduler_result = SchedulerResult(
        status="success",
        stdout_path=artifacts.stdout_file,
        raw_output={"runtime_handle": {"container_name": "appworld-smoke"}},
    )
    resources = SimpleNamespace(metadata={"runtime_handle": {"container_name": "appworld-container"}})

    verifier_input = build_verifier_input(sample, scheduler_result, artifacts, resources)

    assert isinstance(verifier_input, VerifierInput)
    assert verifier_input.benchmark_kit_id == "appworld"
    assert verifier_input.sample_id == "calendar_001"
    assert verifier_input.runtime_handle["container_name"] == "appworld-container"
    assert verifier_input.artifact_paths["stdout_path"].endswith("stdout.log")


@pytest.mark.fast
def test_appworld_finalize_result() -> None:
    sample = {"id": "calendar_001", "metadata": {"appworld": {"task_id": "calendar_001"}}}
    artifacts = _build_artifacts()
    scheduler_result = SchedulerResult(
        status="success",
        stdout_path=artifacts.stdout_file,
        trajectory_path=artifacts.trajectory_file,
    )

    result = finalize_result(sample, scheduler_result, artifacts)

    assert result["sample_id"] == "calendar_001"
    assert result["status"] == "success"
    assert result["stdout_path"].endswith("stdout.log")
    assert result["artifact_paths"]["trajectory_path"].endswith("trajectory.json")
