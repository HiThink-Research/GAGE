from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from gage_eval.config.pipeline_config import PipelineConfig


@pytest.mark.io
def test_swebench_smoke_model_config() -> None:
    config_path = Path(__file__).resolve().parents[3] / "config" / "custom" / "swebench_pro_smoke_model.yaml"
    payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    config = PipelineConfig.from_dict(payload)

    assert config.metadata.get("name") == "swebench_pro_smoke_model"
    assert not config.agent_backends
    assert config.datasets[0].dataset_id == "swebench_pro_smoke"

    steps = [step.step_type for step in config.custom.steps]
    assert steps == ["support", "inference", "judge", "auto_eval"]

    adapter_ids = {spec.adapter_id for spec in config.role_adapters}
    assert {"swebench_context_provider", "swebench_dut_model", "swebench_docker_judge"}.issubset(adapter_ids)

    dut_model = next(spec for spec in config.role_adapters if spec.adapter_id == "swebench_dut_model")
    assert dut_model.role_type == "dut_model"
    assert dut_model.backend_id == "swebench_openai_http"
