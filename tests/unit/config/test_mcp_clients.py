from pathlib import Path

import pytest
import yaml

from gage_eval.config.pipeline_config import PipelineConfig


@pytest.mark.io
def test_appworld_mcp_client_configured() -> None:
    config_path = Path(__file__).resolve().parents[3] / "config" / "custom" / "appworld_agent_demo.yaml"
    payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    config = PipelineConfig.from_dict(payload)

    mcp_clients = {spec.mcp_client_id: spec for spec in config.mcp_clients}
    assert "appworld_env" in mcp_clients
    allowlist = list(mcp_clients["appworld_env"].allowlist)
    assert allowlist == []
    assert mcp_clients["appworld_env"].transport == "streamable_http"
    assert mcp_clients["appworld_env"].endpoint == "http://127.0.0.1:5001"
    assert mcp_clients["appworld_env"].params.get("output_type") == "structured_data_only"

    toolchain = next(spec for spec in config.role_adapters if spec.adapter_id == "toolchain_main")
    assert toolchain.mcp_client_id == "appworld_env"
