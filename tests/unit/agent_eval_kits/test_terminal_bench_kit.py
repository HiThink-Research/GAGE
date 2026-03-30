from __future__ import annotations

from types import SimpleNamespace

import pytest

from gage_eval.agent_eval_kits.terminal_bench.contracts import (
    TERMINAL_BENCH_KIT_ID,
    TERMINAL_BENCH_REQUIRED_SURFACES,
)
from gage_eval.agent_eval_kits.terminal_bench.judge_bridge import build_verifier_input
from gage_eval.agent_eval_kits.terminal_bench.kit import build_kit
from gage_eval.agent_eval_kits.terminal_bench.resources import build_resource_requirements
from gage_eval.agent_eval_kits.terminal_bench.sub_workflow import finalize_result, prepare_inputs
from gage_eval.agent_runtime.artifacts.layout import ArtifactLayout
from gage_eval.agent_runtime.schedulers import SchedulerResult
from gage_eval.agent_runtime.spec import AgentRuntimeSpec, ClientSurfacePolicy, ResourcePolicy, SandboxPolicy
from gage_eval.agent_runtime.verifier.base import VerifierInput


def _build_plan() -> SimpleNamespace:
    runtime_spec = AgentRuntimeSpec(
        agent_runtime_id="codex_terminal_bench",
        scheduler="installed_client",
        benchmark_kit_id=TERMINAL_BENCH_KIT_ID,
        client_id="codex",
        resource_policy=ResourcePolicy(environment_kind="remote", timeout_sec=300),
        client_surface_policy=ClientSurfacePolicy(required=TERMINAL_BENCH_REQUIRED_SURFACES),
        sandbox_policy=SandboxPolicy(prefer_remote=True, remote_mode="attached"),
    )
    return SimpleNamespace(
        runtime_spec=runtime_spec,
        scheduler_type=runtime_spec.scheduler,
        benchmark_kit_id=runtime_spec.benchmark_kit_id,
        client_id=runtime_spec.client_id,
        role_adapter_id="dut_codex_tb",
    )


def _build_artifacts() -> ArtifactLayout:
    return ArtifactLayout.for_sample("runs", "run-1", "tb2__smoke_1")


@pytest.mark.fast
def test_terminal_bench_build_kit_requires_terminal_fs() -> None:
    kit = build_kit()

    assert kit.kit_id == TERMINAL_BENCH_KIT_ID
    assert kit.verifier_kind == "native"
    assert kit.required_surfaces == TERMINAL_BENCH_REQUIRED_SURFACES


@pytest.mark.fast
def test_terminal_bench_build_resource_requirements() -> None:
    sample = {"instance_id": "tb2__smoke_1", "instruction": "Create hello.txt"}

    requirements = build_resource_requirements(sample, _build_plan())

    assert requirements["required_surfaces"] == TERMINAL_BENCH_REQUIRED_SURFACES
    assert requirements["environment_kind"] == "remote"
    assert requirements["remote_mode"] == "attached"
    assert requirements["timeout_sec"] == 300


@pytest.mark.fast
def test_terminal_bench_prepare_inputs() -> None:
    sample = {
        "instance_id": "tb2__smoke_1",
        "instruction": "Create hello.txt",
        "workspace_root": "/tmp/tb2-smoke",
        "env": {"TB_MODE": "smoke"},
    }
    session = SimpleNamespace(
        plan=_build_plan(),
        artifacts=_build_artifacts(),
        metadata={"source": "unit-test"},
    )

    prepared = prepare_inputs(sample, session)

    assert prepared["kit_id"] == TERMINAL_BENCH_KIT_ID
    assert prepared["task_context"]["sample_id"] == "tb2__smoke_1"
    assert prepared["task_context"]["instruction"] == "Create hello.txt"
    assert prepared["instruction"] == "Create hello.txt"
    assert prepared["cwd"] == "/tmp/tb2-smoke"
    assert prepared["env"] == {"TB_MODE": "smoke"}
    assert prepared["surface_requirements"] == TERMINAL_BENCH_REQUIRED_SURFACES
    assert prepared["resource_requirements"]["required_surfaces"] == TERMINAL_BENCH_REQUIRED_SURFACES
    assert prepared["metadata"]["benchmark_kit_id"] == TERMINAL_BENCH_KIT_ID
    assert prepared["artifact_paths"]["patch_file"].endswith("submission.patch")


@pytest.mark.fast
def test_terminal_bench_build_verifier_input() -> None:
    sample = {"instance_id": "tb2__smoke_1", "instruction": "Create hello.txt"}
    artifacts = _build_artifacts()
    scheduler_result = SchedulerResult(status="success", patch_path=artifacts.patch_file, stdout_path=artifacts.stdout_file)

    verifier_input = build_verifier_input(sample, scheduler_result, artifacts)

    assert isinstance(verifier_input, VerifierInput)
    assert verifier_input.benchmark_kit_id == TERMINAL_BENCH_KIT_ID
    assert verifier_input.payload["required_surfaces"] == TERMINAL_BENCH_REQUIRED_SURFACES
    assert verifier_input.artifact_paths["patch_file"].endswith("submission.patch")
    assert verifier_input.payload["scheduler_result"]["status"] == "success"


@pytest.mark.fast
def test_terminal_bench_finalize_result() -> None:
    sample = {"instance_id": "tb2__smoke_1"}
    artifacts = _build_artifacts()
    scheduler_result = SchedulerResult(
        status="success",
        patch_path=artifacts.patch_file,
        stdout_path=artifacts.stdout_file,
        trajectory_path=artifacts.trajectory_file,
        artifacts={"patch_file": artifacts.patch_file},
    )

    result = finalize_result(sample, scheduler_result, artifacts)

    assert result["kit_id"] == TERMINAL_BENCH_KIT_ID
    assert result["sample_id"] == "tb2__smoke_1"
    assert result["surface_requirements"] == TERMINAL_BENCH_REQUIRED_SURFACES
    assert result["status"] == "success"
    assert result["artifact_layout"]["stdout_file"].endswith("stdout.log")
    assert isinstance(result["verifier_input"], VerifierInput)
