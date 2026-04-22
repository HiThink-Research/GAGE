from __future__ import annotations

from pathlib import Path

import pytest
import yaml


REPO_ROOT = Path(__file__).resolve().parents[3]
CONFIG_ROOT = REPO_ROOT / "config" / "custom" / "tau2"


@pytest.mark.fast
@pytest.mark.parametrize("config_path", sorted(CONFIG_ROOT.glob("*.yaml")))
def test_tau2_configs_use_default_tool_call_retry_budget(config_path: Path) -> None:
    payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    adapters = [
        adapter
        for adapter in payload.get("role_adapters", [])
        if adapter.get("role_type") == "dut_agent"
        and str(adapter.get("agent_runtime_id", "")).startswith("tau2_")
    ]

    assert adapters, f"{config_path.name} has no tau2 DUT adapter"
    for adapter in adapters:
        params = adapter.get("params") or {}
        assert "tool_call_retry_budget" not in params


@pytest.mark.fast
@pytest.mark.parametrize("config_path", sorted(CONFIG_ROOT.glob("*.yaml")))
def test_tau2_framework_loop_configs_enable_plain_text_response_fallback(
    config_path: Path,
) -> None:
    payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    adapters = [
        adapter
        for adapter in payload.get("role_adapters", [])
        if adapter.get("role_type") == "dut_agent"
        and str(adapter.get("agent_runtime_id", "")).startswith("tau2_")
        and "installed_client" not in str(adapter.get("agent_runtime_id", ""))
    ]

    if not adapters:
        pytest.skip(f"{config_path.name} has no tau2 framework-loop DUT adapter")
    for adapter in adapters:
        params = adapter.get("params") or {}
        formats = set(params.get("plain_text_response_formats") or [])
        assert params.get("plain_text_response_tool") == "${TAU2_PLAIN_TEXT_RESPONSE_TOOL:-respond}"
        assert {"qwen", "qwen3", "qwen3.5", "qwen3.6", "gemma4"} <= formats


@pytest.mark.fast
def test_tau2_tmp_vllm_full_config_runs_three_subsets() -> None:
    config_path = CONFIG_ROOT / "tau2_telecom_vllm_full_tmp.yaml"
    payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))

    datasets = {
        dataset["dataset_id"]: dataset.get("params", {}).get("domain")
        for dataset in payload.get("datasets", [])
    }
    tasks = {
        task["task_id"]: task
        for task in payload.get("tasks", [])
    }

    assert datasets == {
        "tau2_airline_base": "airline",
        "tau2_retail_base": "retail",
        "tau2_telecom_base": "telecom",
    }
    assert set(tasks) == {
        "tau2_airline_vllm_full",
        "tau2_retail_vllm_full",
        "tau2_telecom_vllm_full",
    }
    assert {
        task["dataset_id"]
        for task in tasks.values()
    } == set(datasets)
    for task in tasks.values():
        assert task.get("max_samples") == 10
        assert [step["step"] for step in task.get("steps", [])] == [
            "inference",
            "judge",
            "auto_eval",
        ]
