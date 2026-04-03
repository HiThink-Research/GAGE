from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from gage_eval.config.pipeline_config import PipelineConfig


@pytest.mark.io
def test_tau2_airline_coding_plan_config_is_parseable() -> None:
    config_path = (
        Path(__file__).resolve().parents[3]
        / "config"
        / "custom"
        / "tau2"
        / "tau2_airline_coding_plan.yaml"
    )
    payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    config = PipelineConfig.from_dict(payload)

    runtime = payload["sandbox_profiles"][0]["runtime_configs"]
    backend = payload["backends"][0]["config"]

    assert config.metadata["name"] == "tau2_airline_coding_plan"
    assert backend["base_url"] == "${TAU2_OPENAI_BASE_URL:-https://coding.dashscope.aliyuncs.com/v1}"
    assert backend["model"] == "${TAU2_AGENT_MODEL:-qwen3.5-plus}"
    assert backend["api_key"] == "${ali_api_key:-}"
    assert runtime["user_model"] == "${TAU2_USER_MODEL:-openai/qwen3.5-plus}"
    assert runtime["user_model_args"]["api_base"] == "${TAU2_USER_API_BASE:-https://coding.dashscope.aliyuncs.com/v1}"
    assert runtime["user_model_args"]["api_key"] == "${ali_api_key:-}"
    assert runtime["user_model_args"]["num_retries"] == "${TAU2_USER_NUM_RETRIES:-0}"
