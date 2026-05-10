from pathlib import Path

import pytest
import yaml

from gage_eval.config.loader import load_pipeline_config_payload
from gage_eval.config.pipeline_config import PipelineConfig


@pytest.mark.io
def test_appworld_official_jsonl_config_parses() -> None:
    config_path = (
        Path(__file__).resolve().parents[3]
        / "config"
        / "custom"
        / "appworld"
        / "appworld_official_jsonl.yaml"
    )
    payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    assert "role_adapters" not in payload
    assert "sandbox_profiles" not in payload
    assert "agent_backends" not in payload

    config = PipelineConfig.from_dict(load_pipeline_config_payload(config_path))

    assert config.metadata.get("name") == "appworld_official_jsonl"
    assert config.datasets[0].dataset_id == "appworld_dev"

    adapter_ids = {spec.adapter_id for spec in config.role_adapters}
    assert adapter_ids == {"dut_agent_main"}

    task = config.tasks[0]
    step_types = [step.step_type for step in task.steps]
    assert step_types == ["inference", "auto_eval"]

    dut_agent = next(spec for spec in config.role_adapters if spec.adapter_id == "dut_agent_main")
    assert dut_agent.agent_runtime_id == "appworld_framework_loop"
    assert dut_agent.params["benchmark_config"] == {}
    assert dut_agent.params["provider_config"]["image"] == "appworld-mcp:latest"

    metric_ids = {metric.metric_id for metric in config.metrics}
    assert {
        "appworld_tgc",
        "appworld_sgc",
        "appworld_pass_count",
        "appworld_fail_count",
        "appworld_difficulty",
    }.issubset(metric_ids)
