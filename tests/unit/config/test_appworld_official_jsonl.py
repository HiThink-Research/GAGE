from pathlib import Path

import pytest
import yaml

from gage_eval.config.pipeline_config import PipelineConfig


@pytest.mark.io
def test_appworld_official_jsonl_config_parses() -> None:
    config_path = Path(__file__).resolve().parents[3] / "config" / "custom" / "appworld_official_jsonl.yaml"
    payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    config = PipelineConfig.from_dict(payload)

    assert config.metadata.get("name") == "appworld_official_jsonl"
    assert config.datasets[0].dataset_id == "appworld_dev"

    adapter_ids = {spec.adapter_id for spec in config.role_adapters}
    assert {
        "api_descriptions_context",
        "api_predictor",
        "toolchain_main",
        "dut_agent_main",
        "appworld_judge",
    }.issubset(adapter_ids)

    task = config.tasks[0]
    step_types = [step.step_type for step in task.steps]
    assert step_types == ["support", "support", "support", "inference", "judge", "auto_eval"]
    assert task.steps[0].adapter_id == "api_descriptions_context"
    assert task.steps[1].adapter_id == "api_predictor"
    assert task.steps[2].adapter_id == "toolchain_main"

    api_predictor = next(spec for spec in config.role_adapters if spec.adapter_id == "api_predictor")
    assert api_predictor.role_type == "helper_model"
    assert api_predictor.params.get("implementation") == "appworld_api_predictor"

    dut_agent = next(spec for spec in config.role_adapters if spec.adapter_id == "dut_agent_main")
    assert dut_agent.prompt_id == "dut/appworld@v1"

    metric_ids = {metric.metric_id for metric in config.metrics}
    assert {
        "appworld_tgc",
        "appworld_sgc",
        "appworld_pass_count",
        "appworld_fail_count",
        "appworld_difficulty",
    }.issubset(metric_ids)
