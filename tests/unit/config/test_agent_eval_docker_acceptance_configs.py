from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from gage_eval.config.pipeline_config import PipelineConfig


def _load_config(name: str) -> PipelineConfig:
    config_path = Path(__file__).resolve().parents[3] / "config" / "custom" / "acceptance" / name
    payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    return PipelineConfig.from_dict(payload)


@pytest.mark.io
def test_docker_installed_client_swebench_config_is_parseable() -> None:
    config = _load_config("docker_installed_client_swebench.yaml")

    runtime = config.agent_runtimes[0]

    assert config.metadata["name"] == "docker_installed_client_swebench"
    assert runtime.agent_runtime_id == "codex_swebench_docker"
    assert runtime.benchmark_kit_id == "swebench"
    assert runtime.resource_policy.environment_kind == "docker"
    assert runtime.params["image"] == "gage-codex-sandbox:latest"
    assert runtime.params["runtime_configs"]["exec_workdir"] == "/workspace"


@pytest.mark.io
def test_docker_installed_client_terminal_bench_config_is_parseable() -> None:
    config = _load_config("docker_installed_client_terminal_bench.yaml")

    runtime = config.agent_runtimes[0]
    metric_ids = [metric.metric_id for metric in config.metrics]

    assert config.metadata["name"] == "docker_installed_client_terminal_bench"
    assert runtime.agent_runtime_id == "codex_terminal_bench_docker"
    assert runtime.benchmark_kit_id == "terminal_bench"
    assert runtime.resource_policy.environment_kind == "docker"
    assert runtime.params["image"] == "gage-codex-sandbox:latest"
    assert [".", "/workspace"] in runtime.params["runtime_configs"]["volumes"]
    assert metric_ids == ["terminal_bench_resolve_rate", "terminal_bench_failure_reason"]


@pytest.mark.io
def test_real_attached_terminal_bench_config_is_parseable() -> None:
    config = _load_config("real_attached_terminal_bench.yaml")

    runtime = config.agent_runtimes[0]
    metric_ids = [metric.metric_id for metric in config.metrics]

    assert config.metadata["name"] == "real_attached_terminal_bench"
    assert runtime.agent_runtime_id == "codex_terminal_bench_real"
    assert runtime.benchmark_kit_id == "terminal_bench"
    assert runtime.resource_policy.environment_kind == "remote"
    assert runtime.sandbox_policy.remote_mode == "attached"
    assert metric_ids == ["terminal_bench_resolve_rate", "terminal_bench_failure_reason"]
