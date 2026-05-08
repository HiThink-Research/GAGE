from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from gage_eval.agent_runtime.schedulers.framework_loop import StaticModelBackendAdapter
from gage_eval.config.registry import ConfigRegistry
from gage_eval.config.pipeline_builder import PipelineConfigBuildError
from gage_eval.config.pipeline_config import PipelineConfig


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
    config_path = (
        Path(__file__).resolve().parents[3]
        / "config"
        / "custom"
        / "tau2"
        / "tau2_telecom_runtime.yaml"
    )
    payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))

    assert "role_adapters" not in payload
    assert "sandbox_profiles" not in payload
    assert payload["agents"][0]["scheduler"]["type"] == "framework_loop"
    assert payload["benchmarks"][0]["kit_id"] == "tau2"
    assert payload["environments"][0]["provider"] == "local_process"


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
    appworld = _load_config(
        base / "appworld" / "appworld_agent_demo_installed_client_codex.yaml"
    )

    terminal_agent = next(spec for spec in terminal.role_adapters if spec.adapter_id == "terminal_agent_main")
    appworld_agent = next(spec for spec in appworld.role_adapters if spec.adapter_id == "dut_agent_main")

    assert terminal_agent.agent_runtime_id == "terminal_bench_installed_client"
    assert appworld_agent.agent_runtime_id == "appworld_installed_client"
    assert terminal_agent.agent_backend_id is None
    assert appworld_agent.agent_backend_id is None


def test_swebench_installed_client_alias_is_agentkit_v2_wrapper() -> None:
    base = Path(__file__).resolve().parents[3] / "config" / "custom"
    payload = yaml.safe_load(
        (base / "swebench_pro" / "swebench_pro_smoke_installed_client_codex.yaml").read_text(
            encoding="utf-8"
        )
    )

    assert "role_adapters" not in payload
    assert "agent_backends" not in payload
    assert payload["agents"][0]["scheduler"]["type"] == "framework_loop"
    assert payload["benchmarks"][0]["kit_id"] == "swebench"


@pytest.mark.parametrize(
    "config_name",
    [
        "agentkit_v2_swebench_pro_docker_lmstudio_smoke11.yaml",
        "agentkit_v2_swebench_pro_e2b_lmstudio_smoke11.yaml",
    ],
)
def test_manual_swebench_e2e_configs_use_non_streaming_hf_dataset(config_name: str) -> None:
    config_path = Path(__file__).resolve().parents[3] / "config" / "custom" / "manual_e2e" / config_name
    payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))

    dataset = next(item for item in payload["datasets"] if item["dataset_id"] == "swebench_pro_smoke11")

    assert dataset["hub"] == "huggingface"
    assert dataset["hub_params"]["hub_id"] == "ScaleAI/SWE-bench_Pro"
    assert dataset["params"]["streaming"] is False


@pytest.mark.parametrize(
    "config_name",
    [
        "agentkit_v2_swebench_pro_docker_lmstudio_smoke11.yaml",
        "agentkit_v2_swebench_pro_e2b_lmstudio_smoke11.yaml",
    ],
)
def test_manual_swebench_e2e_configs_do_not_use_legacy_sandbox_preprocess_kwargs(
    config_name: str,
) -> None:
    config_path = Path(__file__).resolve().parents[3] / "config" / "custom" / "manual_e2e" / config_name
    payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))

    dataset = next(item for item in payload["datasets"] if item["dataset_id"] == "swebench_pro_smoke11")
    preprocess_kwargs = dataset["params"].get("preprocess_kwargs") or {}

    assert "sandbox_id" not in preprocess_kwargs
    assert "sandbox_runtime" not in preprocess_kwargs
    assert "sandbox_lifecycle" not in preprocess_kwargs


@pytest.mark.parametrize(
    "config_name",
    [
        "agentkit_v2_swebench_pro_docker_lmstudio_smoke11.yaml",
        "agentkit_v2_swebench_pro_e2b_lmstudio_smoke11.yaml",
    ],
)
def test_manual_swebench_e2e_configs_use_cost_limit_as_primary_guard(config_name: str) -> None:
    config_path = Path(__file__).resolve().parents[3] / "config" / "custom" / "manual_e2e" / config_name
    payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))

    scheduler_config = payload["agents"][0]["scheduler"]["config"]

    assert scheduler_config["max_turns"] == "${SWEBENCH_MAX_TURNS:-200}"
    assert scheduler_config["cost_limit_usd"] == "${SWEBENCH_COST_LIMIT:-3.0}"


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
        "agent_backends": [
            {
                "agent_backend_id": "demo_agent_backend",
                "type": "model_backend",
                "backend_id": "demo_backend",
            }
        ],
        "role_adapters": [
            {
                "adapter_id": "dut_agent_main",
                "role_type": "dut_agent",
                "agent_runtime_id": "terminal_bench_installed_client",
                "agent_backend_id": "demo_agent_backend",
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

    with pytest.raises(PipelineConfigBuildError, match="must not declare 'agent_backend_id'"):
        PipelineConfig.from_dict(payload)


def test_framework_loop_runtime_reuses_top_level_static_backend() -> None:
    payload = {
        "metadata": {"name": "tau2_static_backend_runtime"},
        "datasets": [
            {
                "dataset_id": "demo_dataset",
                "loader": "jsonl",
                "params": {"path": "tests/data/samples/tau2_demo.jsonl"},
            }
        ],
        "backends": [
            {
                "backend_id": "static_model",
                "type": "dummy",
                "config": {"answer": "ok"},
            }
        ],
        "role_adapters": [
            {
                "adapter_id": "tau2_agent",
                "role_type": "dut_agent",
                "agent_runtime_id": "tau2_framework_loop",
                "backend_id": "static_model",
            }
        ],
        "tasks": [
            {
                "task_id": "demo_task",
                "dataset_id": "demo_dataset",
                "steps": [{"step": "inference", "adapter_id": "tau2_agent"}],
            }
        ],
    }
    config = PipelineConfig.from_dict(payload)
    backend = object()

    adapters = ConfigRegistry().materialize_role_adapters(
        config,
        backends={"static_model": backend},
        agent_backends={},
        sandbox_profiles={},
        mcp_clients={},
    )

    spec = config.role_adapters[0]
    scheduler = adapters["tau2_agent"].executor_ref.compiled_plan.scheduler_handle
    assert spec.backend_id == "static_model"
    assert spec.agent_backend_id is None
    assert isinstance(scheduler._backend, StaticModelBackendAdapter)
    assert scheduler._backend.static_backend is backend
