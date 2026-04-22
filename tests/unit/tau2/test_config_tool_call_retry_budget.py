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
