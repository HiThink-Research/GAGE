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
    assert {"toolchain_main", "dut_agent_main", "appworld_judge"}.issubset(adapter_ids)

    task = config.tasks[0]
    step_types = [step.step_type for step in task.steps]
    assert step_types == ["support", "inference", "judge", "auto_eval"]

    metric_ids = {metric.metric_id for metric in config.metrics}
    assert {
        "appworld_tgc",
        "appworld_sgc",
        "appworld_pass_count",
        "appworld_fail_count",
        "appworld_difficulty",
    }.issubset(metric_ids)
