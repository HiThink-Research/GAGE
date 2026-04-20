from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from gage_eval.config.pipeline_config import PipelineConfig


@pytest.mark.io
def test_swebench_smoke_agent_config() -> None:
    config_path = (
        Path(__file__).resolve().parents[3]
        / "config"
        / "custom"
        / "swebench_pro"
        / "swebench_pro_smoke_agent.yaml"
    )
    payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    config = PipelineConfig.from_dict(payload)

    assert config.metadata.get("name") == "swebench_pro_smoke_agent"
    assert config.agent_runtimes
    assert config.agent_runtimes[0].agent_runtime_id == "swebench_framework_loop"

    steps = [step.step_type for step in config.custom.steps]
    assert steps == ["support", "support", "inference", "judge", "auto_eval"]

    dut_agent = next(spec for spec in config.role_adapters if spec.adapter_id == "swebench_dut_agent")
    assert dut_agent.role_type == "dut_agent"
    assert dut_agent.backend_id == "gpt52_openai_http"
    assert dut_agent.agent_runtime_id == "swebench_framework_loop"
    assert dut_agent.sandbox.get("sandbox_id") == "swebench_runtime"

    toolchain = next(spec for spec in config.role_adapters if spec.adapter_id == "swebench_toolchain")
    assert toolchain.role_type == "toolchain"
