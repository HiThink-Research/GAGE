from __future__ import annotations

from pathlib import Path

import pytest
import yaml


@pytest.mark.io
def test_swebench_smoke_agent_config() -> None:
    config_path = (
        Path(__file__).resolve().parents[3]
        / "config"
        / "custom"
        / "swebench_pro"
        / "swebench_pro_smoke_agent.yaml"
    )
    payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))

    assert payload["metadata"]["name"] == "swebench_pro_smoke_agent"
    assert "role_adapters" not in payload
    assert "sandbox_profiles" not in payload
    assert "agent_backends" not in payload

    assert payload["agents"][0]["scheduler"] == {
        "type": "framework_loop",
        "backend_id": "lmstudio_litellm",
        "config": {
            "max_turns": "${SWEBENCH_MAX_TURNS:-200}",
            "cost_limit_usd": "${SWEBENCH_COST_LIMIT:-3.0}",
        },
    }

    benchmark = payload["benchmarks"][0]
    assert benchmark["kit_id"] == "swebench"
    assert benchmark["config"] == {"split": "test"}

    dut_agent = payload["dut_agents"][0]
    assert dut_agent["agent_id"] == "swebench_agent"
    assert dut_agent["env_id"] == "swebench_docker"
    assert dut_agent["benchmark_id"] == "swebench_pro_smoke"
