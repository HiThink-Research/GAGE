from pathlib import Path

import pytest
import yaml

from gage_eval.config.pipeline_config import PipelineConfig


@pytest.mark.io
def test_appworld_official_jsonl_remote_config_parses() -> None:
    config_path = (
        Path(__file__).resolve().parents[3]
        / "config"
        / "custom"
        / "appworld"
        / "appworld_official_jsonl_remote.yaml"
    )
    payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    config = PipelineConfig.from_dict(payload)

    assert config.metadata.get("name") == "appworld_official_jsonl_remote"
    assert config.datasets[0].dataset_id == "appworld_dev"

    profiles = {spec.sandbox_id: spec for spec in config.sandbox_profiles}
    remote_profile = profiles["appworld_remote"]
    assert remote_profile.runtime == "remote"
    runtime_configs = remote_profile.runtime_configs
    assert runtime_configs.get("control_endpoint") == "${SANDBOX_PLATFORM_URL}"
    assert runtime_configs.get("env_endpoint") == "${APPWORLD_ENV_ENDPOINT:-http://127.0.0.1:8000}"
    assert runtime_configs.get("apis_endpoint") == "${APPWORLD_APIS_ENDPOINT:-http://127.0.0.1:9000}"
    assert runtime_configs.get("mcp_endpoint") == "${APPWORLD_MCP_ENDPOINT:-http://127.0.0.1:5001}"

    appworld_judge = next(
        spec for spec in config.role_adapters if spec.adapter_id == "appworld_judge"
    )
    implementation_params = appworld_judge.params.get("implementation_params") or {}
    assert implementation_params.get("container_name") == (
        "${APPWORLD_CONTAINER_NAME:-appworld-mcp-remote}"
    )

    task = config.tasks[0]
    assert task.concurrency == 1
