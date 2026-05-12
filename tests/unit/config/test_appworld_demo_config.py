from pathlib import Path

import pytest
import yaml

from gage_eval.config.loader import load_pipeline_config_payload
from gage_eval.config.pipeline_config import PipelineConfig


@pytest.mark.io
def test_appworld_demo_config_parses() -> None:
    config_path = (
        Path(__file__).resolve().parents[3]
        / "config"
        / "custom"
        / "appworld"
        / "appworld_agent_demo.yaml"
    )
    payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    assert "role_adapters" not in payload
    assert "sandbox_profiles" not in payload
    assert "agent_backends" not in payload
    assert payload["agents"][0]["scheduler"]["type"] == "framework_loop"

    config = PipelineConfig.from_dict(load_pipeline_config_payload(config_path))

    assert config.metadata.get("name") == "appworld_agent_demo"
    assert config.datasets[0].dataset_id == "appworld_demo"
    assert config.agent_backends == ()

    adapter_ids = {spec.adapter_id for spec in config.role_adapters}
    assert adapter_ids == {"dut_agent_main"}
    dut_agent = next(spec for spec in config.role_adapters if spec.adapter_id == "dut_agent_main")
    assert dut_agent.agent_runtime_id == "appworld_framework_loop"
    assert dut_agent.params["mcp_client_id"] == "appworld_env"
    assert dut_agent.params["force_tool_choice"] == "first_turn"

    task = config.tasks[0]
    step_types = [step.step_type for step in task.steps]
    assert step_types == ["inference", "auto_eval"]

    metric_ids = {metric.metric_id for metric in config.metrics}
    assert "latency" in metric_ids

    assert config.sandbox_profiles == ()
    environment = payload["environments"][0]
    assert environment["profile"]["metadata"]["env_endpoint"] == "http://127.0.0.1:8000"
    assert environment["provider_config"]["ports"] == ["8000:8000", "9000:9000", "5001:5001"]
