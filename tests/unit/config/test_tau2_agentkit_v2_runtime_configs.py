from __future__ import annotations

from pathlib import Path

import yaml

from gage_eval.config.loader import load_pipeline_config_payload


CONFIG_DIR = Path("config/custom/tau2")


def test_tau2_agentkit_v2_operational_configs_lower_to_runtime_adapters() -> None:
    paths = [
        CONFIG_DIR / "agentkit_v2_tau2_telecom_vllm_full.yaml",
        CONFIG_DIR / "agentkit_v2_tau2_multi_domain_vllm_smoke.yaml",
        CONFIG_DIR / "agentkit_v2_tau2_telecom_ollama_full.yaml",
        CONFIG_DIR / "agentkit_v2_tau2_telecom_vllm_runtime.yaml",
    ]

    for path in paths:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
        assert raw["kind"] == "PipelineConfig"
        assert "role_adapters" not in raw
        assert "sandbox_profiles" not in raw
        assert "agent_runtimes" not in raw
        prompt_text = "\n".join(prompt["template"] for prompt in raw["prompts"])
        assert "gage_instruction" not in prompt_text
        assert "gemma4_tool_instruction" not in prompt_text

        materialized = load_pipeline_config_payload(path)

        assert materialized["role_adapters"][0]["agent_runtime_id"] == "tau2_framework_loop"
        assert materialized["role_adapters"][0]["params"]["benchmark_config"]["respond_tool_name"] == "respond"
        assert materialized["role_adapters"][0]["params"]["provider_config"] == {}
        assert materialized["role_adapters"][0]["params"]["force_tool_choice"] == "never"
