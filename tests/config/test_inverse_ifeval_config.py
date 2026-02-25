from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from gage_eval.config.pipeline_config import PipelineConfig


@pytest.mark.fast
def test_inverse_ifeval_qwen_omni_suite_config_parses() -> None:
    config_path = (
        Path(__file__).resolve().parents[2]
        / "config"
        / "custom"
        / "inverse_ifeval"
        / "inverse_ifeval_qwen_omni_suite.yaml"
    )
    payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    config = PipelineConfig.from_dict(payload)

    assert config.metadata.get("name") == "inverse_ifeval_qwen_omni_suite"
    assert len(config.datasets) == 1

    dataset = config.datasets[0]
    assert dataset.dataset_id == "inverse_ifeval_test"
    assert dataset.hub == "huggingface"
    assert dataset.hub_params.get("hub_id") == "m-a-p/Inverse_IFEval"
    assert dataset.hub_params.get("split") == "train"
    assert dataset.hub_params.get("revision") == "main"
    assert dataset.params.get("preprocess") == "inverse_ifeval_preprocessor"

    backend_ids = {backend.backend_id for backend in config.backends}
    assert "qwen3_omni_backend" in backend_ids
    assert "inverse_ifeval_judge_backend" in backend_ids

    adapter_to_backend = {adapter.adapter_id: adapter.backend_id for adapter in config.role_adapters}
    assert adapter_to_backend["dut_qwen3_omni"] == "qwen3_omni_backend"
    assert adapter_to_backend["judge_inverse_ifeval"] == "inverse_ifeval_judge_backend"

    metric_ids = {metric.metric_id for metric in config.metrics}
    assert "inverse_ifeval_judge_pass_rate" in metric_ids
    assert "latency" in metric_ids

    task_map = {task.task_id: task for task in config.tasks}
    assert "inverse_ifeval_qwen3_omni" in task_map
    assert all(task.dataset_id == "inverse_ifeval_test" for task in task_map.values())
    task_steps = [step.step_type for step in task_map["inverse_ifeval_qwen3_omni"].steps]
    assert task_steps == ["inference", "judge", "auto_eval"]
