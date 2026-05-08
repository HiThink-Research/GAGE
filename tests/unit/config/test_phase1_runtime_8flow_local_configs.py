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
            "config/custom/appworld/appworld_agent_demo_runtime_ollama.yaml",
            "dut_agent_main",
            "appworld_framework_loop",
        ),
        (
            "config/custom/appworld/appworld_agent_demo_installed_client_ollama.yaml",
            "dut_agent_main",
            "appworld_installed_client",
        ),
    ],
)
def test_phase1_legacy_local_configs_parse_with_expected_runtime_ids(
    config_relpath: str,
    adapter_id: str,
    agent_runtime_id: str,
) -> None:
    config = _load_config(REPO_ROOT / config_relpath)

    adapter = next(spec for spec in config.role_adapters if spec.adapter_id == adapter_id)
    assert adapter.agent_runtime_id == agent_runtime_id


@pytest.mark.parametrize(
    ("config_relpath", "kit_id"),
    [
        ("config/custom/swebench_pro/swebench_pro_smoke_runtime_ollama_local.yaml", "swebench"),
        ("config/custom/swebench_pro/swebench_pro_smoke_installed_client_ollama_local.yaml", "swebench"),
        ("config/custom/tau2/tau2_telecom_runtime_ollama.yaml", "tau2"),
        ("config/custom/tau2/tau2_telecom_installed_client_ollama.yaml", "tau2"),
    ],
)
def test_phase1_swebench_and_tau2_local_configs_use_agentkit_v2_wrapper(
    config_relpath: str,
    kit_id: str,
) -> None:
    payload = yaml.safe_load((REPO_ROOT / config_relpath).read_text(encoding="utf-8"))

    assert "role_adapters" not in payload
    assert "sandbox_profiles" not in payload
    assert payload["agents"][0]["scheduler"]["type"] == "framework_loop"
    assert payload["benchmarks"][0]["kit_id"] == kit_id


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
    payload = yaml.safe_load((REPO_ROOT / config_relpath).read_text(encoding="utf-8"))
    backends = payload.get("backends") or []

    for backend in backends:
        if backend.get("type") != "openai_http":
            continue
        assert backend.get("config", {}).get("require_api_key") is False


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
    payload = yaml.safe_load((REPO_ROOT / config_relpath).read_text(encoding="utf-8"))
    backends = payload.get("backends") or []

    for backend in backends:
        if backend.get("type") != "openai_http":
            continue
        assert backend.get("config", {}).get("api_key") == "${OLLAMA_API_KEY:-dummy}"


@pytest.mark.parametrize(
    ("config_relpath", "adapter_id"),
    [
        (
            "config/custom/terminal_bench/terminal_bench_installed_client_ollama.yaml",
            "terminal_agent_main",
        ),
        (
            "config/custom/appworld/appworld_agent_demo_installed_client_ollama.yaml",
            "dut_agent_main",
        ),
    ],
)
def test_phase1_remaining_installed_client_local_configs_do_not_declare_agent_backends(
    config_relpath: str,
    adapter_id: str,
) -> None:
    config = _load_config(REPO_ROOT / config_relpath)

    adapter = next(spec for spec in config.role_adapters if spec.adapter_id == adapter_id)
    assert adapter.agent_backend_id is None
    assert not config.agent_backends


def test_runtime_custom_configs_do_not_use_legacy_agent_backends() -> None:
    offenders: list[str] = []
    for config_path in sorted((REPO_ROOT / "config/custom").rglob("*.yaml")):
        payload = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
        role_adapters = payload.get("role_adapters") if isinstance(payload.get("role_adapters"), list) else []
        runtime_adapters = [
            adapter
            for adapter in role_adapters
            if isinstance(adapter, dict) and adapter.get("agent_runtime_id")
        ]
        if not runtime_adapters:
            continue
        if payload.get("agent_backends"):
            offenders.append(f"{config_path.relative_to(REPO_ROOT)} declares agent_backends")
        for adapter in runtime_adapters:
            adapter_id = adapter.get("adapter_id")
            agent_runtime_id = str(adapter.get("agent_runtime_id") or "")
            if adapter.get("agent_backend_id"):
                offenders.append(
                    f"{config_path.relative_to(REPO_ROOT)}:{adapter_id} declares agent_backend_id"
                )
            if agent_runtime_id.endswith("_framework_loop") and not adapter.get("backend_id"):
                offenders.append(
                    f"{config_path.relative_to(REPO_ROOT)}:{adapter_id} missing backend_id"
                )

    assert offenders == []
