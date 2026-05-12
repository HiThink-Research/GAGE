from pathlib import Path

import pytest
import yaml

from tests._support.helpers.network_probe import probe_host_bridge


@pytest.mark.io
def test_network_bridge_from_config() -> None:
    config_path = (
        Path(__file__).resolve().parents[3]
        / "config"
        / "custom"
        / "appworld"
        / "appworld_agent_demo.yaml"
    )
    payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    provider_config = payload["environments"][0]["provider_config"]
    result = probe_host_bridge({"network_mode": provider_config.get("network_mode", "bridge_host")})

    assert result.network_mode == "bridge"
    assert result.host_alias == "host.docker.internal"
    assert "host.docker.internal:host-gateway" in result.extra_hosts
