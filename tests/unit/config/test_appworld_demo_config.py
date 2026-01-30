from pathlib import Path

import pytest
import yaml

from gage_eval.config.pipeline_config import PipelineConfig


@pytest.mark.io
def test_appworld_demo_config_parses() -> None:
    config_path = Path(__file__).resolve().parents[3] / "config" / "custom" / "appworld_agent_demo.yaml"
    payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    config = PipelineConfig.from_dict(payload)

    assert config.metadata.get("name") == "appworld_agent_demo"
    assert config.datasets[0].dataset_id == "appworld_demo"
    assert config.agent_backends[0].agent_backend_id == "agent_model_main"

    adapter_ids = {spec.adapter_id for spec in config.role_adapters}
    assert {"toolchain_main", "dut_agent_main"}.issubset(adapter_ids)

    toolchain = next(spec for spec in config.role_adapters if spec.adapter_id == "toolchain_main")
    assert toolchain.params.get("meta_tool_mode") is False
    assert toolchain.params.get("tool_doc_format") == "schema_yaml"

    task = config.tasks[0]
    step_types = [step.step_type for step in task.steps]
    assert step_types == ["support", "inference", "auto_eval"]

    metric_ids = {metric.metric_id for metric in config.metrics}
    assert "latency" in metric_ids

    profiles = {spec.sandbox_id: spec for spec in config.sandbox_profiles}
    assert profiles["appworld_local"].runtime == "docker"
    runtime_configs = profiles["appworld_local"].runtime_configs
    assert runtime_configs.get("env_endpoint") == "http://127.0.0.1:8000"
