from pathlib import Path

import pytest
import yaml

from gage_eval.config.pipeline_config import PipelineConfig
from tests.integration.runtime.network_probe import probe_host_bridge


@pytest.mark.io
def test_network_bridge_from_config() -> None:
    config_path = Path(__file__).resolve().parents[3] / "config" / "custom" / "appworld_agent_demo.yaml"
    payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    config = PipelineConfig.from_dict(payload)

    profile = next(spec for spec in config.sandbox_profiles if spec.sandbox_id == "appworld_local")
    result = probe_host_bridge(profile.runtime_configs)

    assert result.network_mode == "bridge"
    assert result.host_alias == "host.docker.internal"
    assert "host.docker.internal:host-gateway" in result.extra_hosts
