from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from gage_eval.agent_eval_kits.swebench.judge_bridge import build_verifier_input
from gage_eval.agent_eval_kits.swebench.kit import build_kit
from gage_eval.agent_eval_kits.swebench.resources import build_resource_requirements
from gage_eval.agent_eval_kits.swebench.sub_workflow import finalize_result, prepare_inputs
from gage_eval.agent_runtime.artifacts.layout import ArtifactLayout
from gage_eval.config.pipeline_config import PipelineConfig


def test_swebench_build_kit_returns_definition() -> None:
    kit = build_kit()

    assert kit.kit_id == "swebench"
    assert kit.verifier_kind == "judge_adapter"
    assert "terminal" in kit.required_surfaces
    assert "fs" in kit.required_surfaces


def test_swebench_prepare_inputs() -> None:
    sample = {
        "id": "swebench_1",
        "messages": [{"role": "user", "content": "Fix the bug"}],
        "metadata": {"instance_id": "swebench_1", "repo": "example/repo"},
    }
    session = SimpleNamespace(
        trace=SimpleNamespace(run_id="run-1"),
        plan=SimpleNamespace(benchmark_kit_id="swebench"),
        artifacts=ArtifactLayout.for_sample("/tmp/runs", "run-1", "swebench_1"),
    )

    payload = prepare_inputs(sample, session)

    assert payload["sample_id"] == "swebench_1"
    assert payload["instruction"] == "Fix the bug"
    assert payload["cwd"] == "/app"
    assert payload["env"] == {}
    assert payload["artifacts"]["patch_file"].endswith("submission.patch")
    assert payload["metadata"]["artifact_paths"]["stdout_path"].endswith("stdout.log")


def test_swebench_build_verifier_input() -> None:
    sample = {
        "id": "swebench_1",
        "messages": [{"role": "user", "content": "Fix the bug"}],
        "metadata": {"instance_id": "swebench_1", "repo": "example/repo"},
    }
    scheduler_result = SimpleNamespace(
        status="success",
        answer="diff --git a/foo b/foo",
        patch_path="/tmp/runs/run-1/samples/swebench_1/agent/submission.patch",
        stdout_path="/tmp/runs/run-1/samples/swebench_1/agent/stdout.log",
        trajectory_path="/tmp/runs/run-1/samples/swebench_1/agent/trajectory.json",
        raw_output={"answer": "diff --git a/foo b/foo"},
    )
    artifacts = ArtifactLayout.for_sample("/tmp/runs", "run-1", "swebench_1")

    verifier_input = build_verifier_input(sample, scheduler_result, artifacts)

    assert verifier_input.benchmark_kit_id == "swebench"
    assert verifier_input.sample_id == "swebench_1"
    assert verifier_input.payload["model_output"] == "diff --git a/foo b/foo"
    assert verifier_input.artifact_paths["patch_file"].endswith("submission.patch")


def test_swebench_finalize_result() -> None:
    sample = {"id": "swebench_1", "metadata": {"instance_id": "swebench_1"}}
    scheduler_result = SimpleNamespace(
        status="success",
        answer="diff --git a/foo b/foo",
        raw_output={"answer": "diff --git a/foo b/foo"},
    )
    artifacts = ArtifactLayout.for_sample("/tmp/runs", "run-1", "swebench_1")

    result = finalize_result(sample, scheduler_result, artifacts)

    assert result["sample_id"] == "swebench_1"
    assert result["status"] == "success"
    assert "sample" not in result
    assert result["raw_output"] is not scheduler_result.raw_output
    assert result["artifacts"]["metadata_file"].endswith("runtime_metadata.json")
    assert result["stdout_path"].endswith("stdout.log")


def test_swebench_build_resource_requirements() -> None:
    sample = {"id": "swebench_1"}
    plan = SimpleNamespace(sandbox_profile_id="swebench_runtime")

    requirements = build_resource_requirements(sample, plan)

    assert requirements["benchmark_kit_id"] == "swebench"
    assert requirements["sandbox_profile_id"] == "swebench_runtime"


@pytest.mark.io
def test_smoke_installed_client_swebench_config_is_parseable() -> None:
    config_path = (
        Path(__file__).resolve().parents[3]
        / "config"
        / "custom"
        / "smoke_installed_client_swebench.yaml"
    )
    payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    config = PipelineConfig.from_dict(payload)
    metric_ids = [metric.metric_id for metric in config.metrics]

    assert payload["metadata"]["name"] == "smoke_installed_client_swebench"
    assert payload["agent_runtimes"][0]["agent_runtime_id"] == "codex_swebench"
    assert payload["benchmark_kits"][0]["kit_id"] == "swebench"
    assert config.metadata["name"] == "smoke_installed_client_swebench"
    assert metric_ids == ["swebench_resolve_rate", "swebench_failure_reason"]
