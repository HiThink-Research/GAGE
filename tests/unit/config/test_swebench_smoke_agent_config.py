from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from gage_eval.config.pipeline_config import PipelineConfig


SWE_BENCH_MAX_TURNS_CONFIGS = [
    "swebench_pro_smoke_agent.yaml",
    "swebench_pro_smoke_installed_client_codex.yaml",
    "swebench_pro_smoke_installed_client_ollama_local.yaml",
    "swebench_pro_smoke_runtime.yaml",
    "swebench_pro_smoke_runtime_ollama_local.yaml",
]

SWE_BENCH_SMOKE_CONFIGS = [
    "swebench_pro_smoke_agent.yaml",
    "swebench_pro_smoke_installed_client_codex.yaml",
    "swebench_pro_smoke_installed_client_ollama_local.yaml",
    "swebench_pro_smoke_runtime.yaml",
    "swebench_pro_smoke_runtime_ollama_local.yaml",
]

SWE_BENCH_PREFIXED_THIRD_PARTY_PATH = "gage-eval-main/third_party/swebench_pro"


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
    assert dut_agent.params.get("max_turns") == 100
    assert dut_agent.params.get("max_total_invalid_tool_calls") == 20

    toolchain = next(spec for spec in config.role_adapters if spec.adapter_id == "swebench_toolchain")
    assert toolchain.role_type == "toolchain"

    prompt_template = config.prompts[0].template
    assert "ls -R" not in prompt_template
    assert "Use `ls -R`, `grep`, or `find` to locate relevant files." not in prompt_template
    assert '`rg --files`' in prompt_template
    assert '`rg "symbol|function|class" path`' in prompt_template
    assert "`find <target_dir> -maxdepth 2`" in prompt_template

    backend_config = config.backends[0].config
    assert backend_config.get("max_retries") == 20
    assert backend_config.get("generation_parameters", {}).get("temperature") == 0.0
    assert backend_config.get("generation_parameters", {}).get("max_new_tokens") == 16384
    assert (
        payload["datasets"][0]["params"]["preprocess_kwargs"]["smoke_ids_path"]
        == f"{SWE_BENCH_PREFIXED_THIRD_PARTY_PATH}/run_scripts/smoke_instance_ids.txt"
    )
    assert (
        payload["role_adapters"][-1]["params"]["implementation_params"]["scripts_dir"]
        == f"{SWE_BENCH_PREFIXED_THIRD_PARTY_PATH}/run_scripts"
    )
    assert (
        payload["role_adapters"][-1]["params"]["implementation_params"]["dockerfiles_dir"]
        == f"{SWE_BENCH_PREFIXED_THIRD_PARTY_PATH}/dockerfiles"
    )


@pytest.mark.parametrize("config_name", SWE_BENCH_MAX_TURNS_CONFIGS)
def test_swebench_smoke_configs_use_100_max_turns(config_name: str) -> None:
    config_path = Path(__file__).resolve().parents[3] / "config" / "custom" / "swebench_pro" / config_name
    payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    max_turns = [
        adapter.get("params", {}).get("max_turns")
        for adapter in payload.get("role_adapters", [])
        if adapter.get("params", {}).get("max_turns") is not None
    ]

    assert max_turns
    assert max_turns == [100]


@pytest.mark.parametrize("config_name", SWE_BENCH_SMOKE_CONFIGS)
def test_swebench_smoke_configs_align_runtime_defaults(config_name: str) -> None:
    config_path = Path(__file__).resolve().parents[3] / "config" / "custom" / "swebench_pro" / config_name
    payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))

    dut_adapters = [
        adapter
        for adapter in payload.get("role_adapters", [])
        if adapter.get("adapter_id") == "swebench_dut_agent"
    ]
    assert dut_adapters
    assert [adapter.get("params", {}).get("max_turns") for adapter in dut_adapters] == [100]

    agent_loop_adapters = [
        adapter
        for adapter in dut_adapters
        if adapter.get("agent_runtime_id") == "swebench_framework_loop"
    ]
    for adapter in agent_loop_adapters:
        assert adapter.get("params", {}).get("max_total_invalid_tool_calls") == 20

    for backend in payload.get("backends", []):
        config = backend.get("config", {})
        if "max_retries" in config:
            assert config.get("max_retries") == 20
        generation_parameters = config.get("generation_parameters", {})
        assert generation_parameters.get("temperature") == 0.0
        assert generation_parameters.get("max_new_tokens") == 16384

    if config_name == "swebench_pro_smoke_agent.yaml":
        prompt = payload["prompts"][0]["template"]
        assert "ls -R" not in prompt
        assert "Use `ls -R`, `grep`, or `find` to locate relevant files." not in prompt
        assert '`rg --files`' in prompt
        assert '`rg "symbol|function|class" path`' in prompt
        assert "`find <target_dir> -maxdepth 2`" in prompt

    dataset_params = payload["datasets"][0]["params"]
    preprocess_kwargs = dataset_params.get("preprocess_kwargs", {})
    if "smoke_ids_path" in preprocess_kwargs:
        assert (
            preprocess_kwargs["smoke_ids_path"]
            == f"{SWE_BENCH_PREFIXED_THIRD_PARTY_PATH}/run_scripts/smoke_instance_ids.txt"
        )

    for adapter in payload.get("role_adapters", []):
        implementation_params = adapter.get("params", {}).get("implementation_params", {})
        if "scripts_dir" in implementation_params:
            assert (
                implementation_params["scripts_dir"]
                == f"{SWE_BENCH_PREFIXED_THIRD_PARTY_PATH}/run_scripts"
            )
        if "dockerfiles_dir" in implementation_params:
            assert (
                implementation_params["dockerfiles_dir"]
                == f"{SWE_BENCH_PREFIXED_THIRD_PARTY_PATH}/dockerfiles"
            )
