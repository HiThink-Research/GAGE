from __future__ import annotations

from pathlib import Path

import pytest
import yaml


@pytest.mark.io
@pytest.mark.parametrize(
    "config_name",
    ["swebench_pro_smoke_agent.yaml"],
)
def test_swebench_smoke_sandbox_profile(config_name: str) -> None:
    config_path = (
        Path(__file__).resolve().parents[3]
        / "config"
        / "custom"
        / "swebench_pro"
        / config_name
    )
    payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))

    assert "sandbox_profiles" not in payload
    assert "role_adapters" not in payload

    env = payload["environments"][0]
    assert env["provider"] == "docker"
    assert env["lifecycle"] == "per_sample"
    assert env["profile"]["asset_dir"] == "src/gage_eval/agent_eval_kits/swebench/environment/docker"

    provider_config = env["provider_config"]
    assert provider_config["network_policy"] == "block"
    assert provider_config["docker_platform"] == "linux/amd64"
    assert provider_config["entrypoint"] == []
    assert provider_config["keepalive_command"] == ["sleep", "infinity"]
    assert provider_config["exec_workdir"] == "/app"
