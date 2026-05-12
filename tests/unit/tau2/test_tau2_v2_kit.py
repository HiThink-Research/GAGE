from __future__ import annotations

from types import SimpleNamespace

import pytest
from pydantic import ValidationError

from gage_eval.agent_eval_kits.common import validate_benchmark_kit_entry
from gage_eval.agent_eval_kits.tau2.config_schema import Tau2KitConfig
from gage_eval.agent_eval_kits.tau2.kit import load_kit
from gage_eval.agent_eval_kits.tau2.runtime import _resolve_tau2_runtime_settings
from gage_eval.agent_eval_kits.tau2.sub_workflows.framework_loop import build_workflow_bundle
from gage_eval.environment.providers.local_process import LocalProcessEnvironmentConfig


def test_tau2_kit_declares_v2_environment_and_verifier_contracts() -> None:
    kit = load_kit()

    assert kit.default_environment_provider == "local_process"
    assert kit.default_environment_profile_by_provider["local_process"] == "tau2-local-process"
    assert kit.default_environment_profile_by_provider["e2b"] == "tau2-e2b-wrapper"
    assert kit.environment_profiles["tau2-local-process"]["asset_dir"].endswith(
        "agent_eval_kits/tau2/environment/local_process"
    )
    assert kit.environment_profiles["tau2-e2b-wrapper"]["asset_dir"].endswith(
        "agent_eval_kits/tau2/environment/e2b"
    )
    assert kit.environment_profiles["tau2-e2b-wrapper"]["config"]["template_id"] == "gage-tau2-wrapper"
    assert kit.verifier_environment_policy == "reuse"
    assert kit.verifier_environment_profile_id is None
    assert kit.verifier_adapter_factory is not None
    assert kit.artifact_manifest_factory is not None
    assert kit.build_verifier_adapter().__class__.__module__ == (
        "gage_eval.agent_eval_kits.tau2.judge.adapters"
    )
    assert validate_benchmark_kit_entry(kit) is kit


def test_tau2_framework_loop_declares_respond_as_required_tool() -> None:
    workflow = build_workflow_bundle()

    loop_inputs = workflow.build_loop_inputs(
        session=SimpleNamespace(prompt_context={}, benchmark_state={}),
        sample={"messages": [{"role": "user", "content": "hello"}]},
        payload={},
    )

    assert loop_inputs["required_tool"] == "respond"


def test_tau2_kit_config_accepts_runtime_fields() -> None:
    config = Tau2KitConfig.model_validate(
        {
            "domain": "telecom",
            "data_dir": "/tmp/tau2",
            "max_steps": 200,
            "max_errors": 10,
            "respond_tool_name": "respond",
            "user_simulator": {
                "type": "litellm",
                "model": "openai/qwen",
                "api_base": "http://127.0.0.1:1234/v1",
                "api_key": "dummy",
            },
        }
    )

    assert config.data_dir == "/tmp/tau2"
    assert config.max_steps == 200
    assert config.max_errors == 10
    assert config.respond_tool_name == "respond"
    assert config.user_simulator == {
        "model": "openai/qwen",
        "model_args": {
            "api_base": "http://127.0.0.1:1234/v1",
            "api_key": "dummy",
        },
    }


def test_tau2_runtime_settings_are_read_from_benchmark_config_only() -> None:
    payload = {
        "benchmark_config": {
            "data_dir": "/tmp/from-benchmark",
            "max_steps": 5,
            "user_simulator": {"model": "benchmark-user", "model_args": {}},
        },
        "provider_config": {
            "data_dir": "/tmp/from-provider",
            "max_steps": 999,
            "user_simulator": {"model": "provider-user", "model_args": {}},
        },
    }

    settings = _resolve_tau2_runtime_settings(
        session=SimpleNamespace(runtime_context={}),
        payload=payload,
    )

    assert settings["data_dir"] == "/tmp/from-benchmark"
    assert settings["max_steps"] == 5
    assert settings["user_simulator"]["model"] == "benchmark-user"


def test_tau2_framework_loop_inputs_include_benchmark_config() -> None:
    workflow = build_workflow_bundle()
    benchmark_config = {"data_dir": "/tmp/tau2", "max_steps": 5}

    loop_inputs = workflow.build_loop_inputs(
        session=SimpleNamespace(prompt_context={}, benchmark_state={}, runtime_context={}),
        sample={"messages": [{"role": "user", "content": "hello"}]},
        payload={"benchmark_config": benchmark_config},
    )

    assert loop_inputs["benchmark_config"] == benchmark_config


def test_tau2_local_process_provider_rejects_tau2_runtime_fields() -> None:
    with pytest.raises(ValidationError):
        LocalProcessEnvironmentConfig.model_validate({"data_dir": "/tmp/tau2"})
