from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from gage_eval.config.pipeline_config import PipelineConfig


REPO_ROOT = Path(__file__).resolve().parents[3]


def _load_config(config_path: Path) -> PipelineConfig:
    payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    return PipelineConfig.from_dict(payload)


@pytest.mark.parametrize(
    ("config_relpath", "adapter_id", "agent_runtime_id"),
    [
        (
            "config/custom/terminal_bench/terminal_bench_framework_loop_ollama.yaml",
            "terminal_agent_main",
            "terminal_bench_framework_loop",
        ),
        (
            "config/custom/terminal_bench/terminal_bench_installed_client_ollama.yaml",
            "terminal_agent_main",
            "terminal_bench_installed_client",
        ),
        (
            "config/custom/swebench_pro/swebench_pro_smoke_runtime_ollama_local.yaml",
            "swebench_dut_agent",
            "swebench_framework_loop",
        ),
        (
            "config/custom/swebench_pro/swebench_pro_smoke_installed_client_ollama_local.yaml",
            "swebench_dut_agent",
            "swebench_installed_client",
        ),
        (
            "config/custom/appworld/appworld_agent_demo_runtime_ollama.yaml",
            "dut_agent_main",
            "appworld_framework_loop",
        ),
        (
            "config/custom/appworld/appworld_agent_demo_installed_client_ollama.yaml",
            "dut_agent_main",
            "appworld_installed_client",
        ),
        (
            "config/custom/tau2/tau2_telecom_runtime_ollama.yaml",
            "tau2_agent",
            "tau2_framework_loop",
        ),
        (
            "config/custom/tau2/tau2_telecom_installed_client_ollama.yaml",
            "tau2_agent",
            "tau2_installed_client",
        ),
    ],
)
def test_phase1_8flow_local_configs_parse_with_expected_runtime_ids(
    config_relpath: str,
    adapter_id: str,
    agent_runtime_id: str,
) -> None:
    config = _load_config(REPO_ROOT / config_relpath)

    adapter = next(spec for spec in config.role_adapters if spec.adapter_id == adapter_id)
    assert adapter.agent_runtime_id == agent_runtime_id


@pytest.mark.parametrize(
    "config_relpath",
    [
        "config/custom/terminal_bench/terminal_bench_framework_loop_ollama.yaml",
        "config/custom/terminal_bench/terminal_bench_installed_client_ollama.yaml",
        "config/custom/swebench_pro/swebench_pro_smoke_runtime_ollama_local.yaml",
        "config/custom/swebench_pro/swebench_pro_smoke_installed_client_ollama_local.yaml",
        "config/custom/appworld/appworld_agent_demo_runtime_ollama.yaml",
        "config/custom/appworld/appworld_agent_demo_installed_client_ollama.yaml",
        "config/custom/tau2/tau2_telecom_runtime_ollama.yaml",
        "config/custom/tau2/tau2_telecom_installed_client_ollama.yaml",
    ],
)
def test_phase1_8flow_local_configs_disable_required_api_keys(config_relpath: str) -> None:
    config = _load_config(REPO_ROOT / config_relpath)

    for backend in config.backends:
        if backend.type != "openai_http":
            continue
        assert backend.config.get("require_api_key") is False


@pytest.mark.parametrize(
    "config_relpath",
    [
        "config/custom/terminal_bench/terminal_bench_framework_loop_ollama.yaml",
        "config/custom/terminal_bench/terminal_bench_installed_client_ollama.yaml",
        "config/custom/swebench_pro/swebench_pro_smoke_runtime_ollama_local.yaml",
        "config/custom/swebench_pro/swebench_pro_smoke_installed_client_ollama_local.yaml",
        "config/custom/appworld/appworld_agent_demo_runtime_ollama.yaml",
        "config/custom/appworld/appworld_agent_demo_installed_client_ollama.yaml",
        "config/custom/tau2/tau2_telecom_runtime_ollama.yaml",
        "config/custom/tau2/tau2_telecom_installed_client_ollama.yaml",
    ],
)
def test_phase1_8flow_local_configs_pin_ollama_api_key(config_relpath: str) -> None:
    config = _load_config(REPO_ROOT / config_relpath)

    for backend in config.backends:
        if backend.type != "openai_http":
            continue
        assert backend.config.get("api_key") == "${OLLAMA_API_KEY:-dummy}"


@pytest.mark.parametrize(
    ("config_relpath", "adapter_id"),
    [
        (
            "config/custom/terminal_bench/terminal_bench_installed_client_ollama.yaml",
            "terminal_agent_main",
        ),
        (
            "config/custom/swebench_pro/swebench_pro_smoke_installed_client_ollama_local.yaml",
            "swebench_dut_agent",
        ),
        (
            "config/custom/appworld/appworld_agent_demo_installed_client_ollama.yaml",
            "dut_agent_main",
        ),
        (
            "config/custom/tau2/tau2_telecom_installed_client_ollama.yaml",
            "tau2_agent",
        ),
    ],
)
def test_phase1_installed_client_local_configs_do_not_declare_agent_backends(
    config_relpath: str,
    adapter_id: str,
) -> None:
    config = _load_config(REPO_ROOT / config_relpath)

    adapter = next(spec for spec in config.role_adapters if spec.adapter_id == adapter_id)
    assert adapter.agent_backend_id is None
    assert not config.agent_backends
