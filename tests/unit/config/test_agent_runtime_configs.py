from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from gage_eval.config.loader import load_pipeline_config_payload
from gage_eval.config.pipeline_builder import PipelineConfigBuildError
from gage_eval.config.pipeline_config import PipelineConfig
from gage_eval.config.registry import ConfigRegistry


def _load_config(path: Path) -> PipelineConfig:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    return PipelineConfig.from_dict(payload)


def test_appworld_runtime_config_parses() -> None:
    config = _load_config(
        Path(__file__).resolve().parents[3]
        / "config"
        / "custom"
        / "appworld"
        / "appworld_agent_demo_runtime.yaml"
    )

    dut_agent = next(spec for spec in config.role_adapters if spec.adapter_id == "dut_agent_main")
    assert dut_agent.agent_runtime_id == "appworld_framework_loop"
    assert dut_agent.params.get("pre_hooks") is None
    assert dut_agent.params.get("post_hooks") is None


def test_tau2_runtime_config_drops_bootstrap_support() -> None:
    config = _load_config(
        Path(__file__).resolve().parents[3]
        / "config"
        / "custom"
        / "tau2"
        / "tau2_telecom_runtime.yaml"
    )

    adapter_ids = {spec.adapter_id for spec in config.role_adapters}
    assert "tau2_bootstrap" not in adapter_ids
    dut_agent = next(spec for spec in config.role_adapters if spec.adapter_id == "tau2_agent")
    assert dut_agent.agent_runtime_id == "tau2_framework_loop"


def test_tau2_user_simulator_config_materializes_outside_sandbox_profile() -> None:
    config_path = (
        Path(__file__).resolve().parents[3]
        / "config"
        / "custom"
        / "tau2"
        / "tau2_telecom_runtime.yaml"
    )
    config = PipelineConfig.from_dict(load_pipeline_config_payload(config_path))

    raw_profile = next(spec for spec in config.sandbox_profiles if spec.sandbox_id == "tau2_local")
    assert "user_model" not in raw_profile.runtime_configs
    assert "user_simulator" not in raw_profile.runtime_configs

    profiles = ConfigRegistry().materialize_sandbox_profiles(config)
    assert "user_simulator" not in profiles["tau2_local"]["runtime_configs"]


def test_terminal_and_skillsbench_runtime_smokes_parse() -> None:
    base = Path(__file__).resolve().parents[3] / "config" / "custom"
    terminal = _load_config(base / "terminal_bench" / "terminal_bench_smoke_runtime.yaml")
    skills = _load_config(base / "skillsbench" / "skillsbench_smoke_runtime.yaml")

    terminal_agent = next(spec for spec in terminal.role_adapters if spec.adapter_id == "terminal_agent_main")
    skills_agent = next(spec for spec in skills.role_adapters if spec.adapter_id == "skillsbench_agent_main")
    assert terminal_agent.agent_runtime_id == "terminal_bench_framework_loop"
    assert skills_agent.agent_runtime_id == "skillsbench_framework_loop"


def test_builtin_codex_installed_client_configs_parse_without_agent_backend() -> None:
    base = Path(__file__).resolve().parents[3] / "config" / "custom"
    terminal = _load_config(
        base / "terminal_bench" / "terminal_bench_installed_client_codex.yaml"
    )
    swebench = _load_config(
        base / "swebench_pro" / "swebench_pro_smoke_installed_client_codex.yaml"
    )
    appworld = _load_config(
        base / "appworld" / "appworld_agent_demo_installed_client_codex.yaml"
    )

    terminal_agent = next(spec for spec in terminal.role_adapters if spec.adapter_id == "terminal_agent_main")
    swebench_agent = next(spec for spec in swebench.role_adapters if spec.adapter_id == "swebench_dut_agent")
    appworld_agent = next(spec for spec in appworld.role_adapters if spec.adapter_id == "dut_agent_main")

    assert terminal_agent.agent_runtime_id == "terminal_bench_installed_client"
    assert swebench_agent.agent_runtime_id == "swebench_installed_client"
    assert appworld_agent.agent_runtime_id == "appworld_installed_client"
    assert terminal_agent.backend_id is None
    assert swebench_agent.backend_id is None
    assert appworld_agent.backend_id is None


def test_installed_client_runtime_rejects_agent_backend_binding() -> None:
    payload = {
        "metadata": {"name": "invalid_installed_client_binding"},
        "datasets": [
            {
                "dataset_id": "demo_dataset",
                "loader": "jsonl",
                "params": {"path": "tests/data/samples/terminal_bench_demo.jsonl"},
            }
        ],
        "backends": [
            {
                "backend_id": "demo_backend",
                "type": "openai_http",
                "config": {"base_url": "http://127.0.0.1:11434/v1", "model": "dummy"},
            }
        ],
        "agent_runtimes": [
            {
                "agent_runtime_id": "terminal_bench_installed_client",
                "benchmark_kit_id": "terminal_bench",
                "scheduler_type": "installed_client",
                "client_id": "codex",
                "sandbox_profile_id": "terminal_bench_runtime",
                "resource_policy": {"resource_kind": "docker", "lifecycle": "per_sample"},
                "verifier_binding_id": "terminal_bench_native",
            }
        ],
        "role_adapters": [
            {
                "adapter_id": "dut_agent_main",
                "role_type": "dut_agent",
                "agent_runtime_id": "terminal_bench_installed_client",
                "backend_id": "demo_backend",
            }
        ],
        "tasks": [
            {
                "task_id": "demo_task",
                "dataset_id": "demo_dataset",
                "steps": [{"step": "inference", "adapter_id": "dut_agent_main"}],
            }
        ],
    }

    with pytest.raises(PipelineConfigBuildError, match="must not declare 'backend_id'"):
        PipelineConfig.from_dict(payload)
