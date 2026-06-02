from __future__ import annotations

import pytest
from pydantic import ValidationError

from gage_eval.registry import registry
from gage_eval.role.model.backends.litellm_backend import LiteLLMBackend
from gage_eval.role.model.config.litellm import LiteLLMBackendConfig

pytestmark = pytest.mark.fast


def test_litellm_backend_registry_advertises_multimodal_modalities() -> None:
    assert registry.get("backends", "litellm") is LiteLLMBackend
    assert registry.entry("backends", "litellm").extra["modalities"] == ("text", "vision", "audio")


def test_simplified_vllm_config_parses_as_direct_endpoint() -> None:
    config = LiteLLMBackendConfig.model_validate(
        {
            "model": "hosted_vllm/Qwen/Qwen3.6-35B-A3B",
            "api_base": "http://127.0.0.1:8000/v1",
            "api_key": "dummy",
        }
    )

    assert config.model == "hosted_vllm/Qwen/Qwen3.6-35B-A3B"
    assert config.api_base == "http://127.0.0.1:8000/v1"
    assert config.route_mode == "direct"
    assert config.resolved_route_mode() == "direct"


def test_vllm_topology_profile_fields_parse_without_affecting_route_mode() -> None:
    config = LiteLLMBackendConfig.model_validate(
        {
            "model": "hosted_vllm/Qwen/Qwen3.6-35B-A3B",
            "vllm": {
                "topology": {
                    "language_model_only": True,
                    "kind": "single_node_single_gpu",
                    "gpu_ids": [0],
                },
                "reasoning_parser": "qwen3",
            },
        }
    )

    assert config.resolved_route_mode() == "direct"
    assert config.vllm.topology.language_model_only is True
    assert config.vllm.topology.kind == "single_node_single_gpu"
    assert config.vllm.topology.gpu_ids == [0]
    assert config.vllm.reasoning_parser == "qwen3"


def test_model_server_command_mode_parses_shutdown_policy_fields() -> None:
    config = LiteLLMBackendConfig.model_validate(
        {
            "model": "hosted_vllm/Qwen/Qwen3.6-35B-A3B",
            "model_server": {
                "enabled": True,
                "experimental": True,
                "startup_command": "bash framework/0521/start_vllm_8gpu.sh",
                "shutdown_policy": "terminate_on_exit",
                "pid_file": "/tmp/gage-vllm.pid",
                "health": {
                    "urls": [
                        "http://127.0.0.1:8000/health",
                        "http://127.0.0.1:8001/health",
                    ],
                    "timeout_seconds": 1800,
                    "interval_seconds": 5,
                },
            },
        }
    )

    assert config.model_server.enabled is True
    assert config.model_server.experimental is True
    assert config.model_server.startup_command == "bash framework/0521/start_vllm_8gpu.sh"
    assert config.model_server.shutdown_policy == "terminate_on_exit"
    assert config.model_server.pid_file == "/tmp/gage-vllm.pid"
    assert config.model_server.health.urls == [
        "http://127.0.0.1:8000/health",
        "http://127.0.0.1:8001/health",
    ]
    assert config.model_server.health.timeout_seconds == 1800
    assert config.model_server.health.interval_seconds == 5
    assert not hasattr(config.model_server, "kind")
    assert not hasattr(config.model_server, "cwd")


def test_thinking_and_multimodal_policy_fields_parse() -> None:
    config = LiteLLMBackendConfig.model_validate(
        {
            "model": "hosted_vllm/Qwen/Qwen3.6-35B-A3B",
            "thinking_policy": {
                "capability": "auto",
                "on_unsupported": "warn_and_continue",
                "on_mismatch": "fail_fast",
                "record_effective_state": True,
            },
            "multimodal": {
                "preserve_blocks": True,
                "audio_url_mapping": "input_audio",
                "default_audio_format": "wav",
                "max_media_bytes": 5242880,
            },
        }
    )

    assert config.thinking_policy.on_mismatch == "fail_fast"
    assert config.thinking_policy.record_effective_state is True
    assert config.multimodal.audio_url_mapping == "input_audio"
    assert config.multimodal.default_audio_format == "wav"
    assert config.multimodal.max_media_bytes == 5242880
    assert config.multimodal.fallback == "fail"


@pytest.mark.parametrize(
    ("capability", "expected"),
    [
        (True, "supported"),
        (False, "unsupported"),
    ],
)
def test_thinking_policy_capability_accepts_design_boolean_shortcuts(
    capability: bool,
    expected: str,
) -> None:
    config = LiteLLMBackendConfig.model_validate(
        {
            "model": "hosted_vllm/Qwen/Qwen3.6-35B-A3B",
            "thinking_policy": {"capability": capability},
        }
    )

    assert config.thinking_policy.capability == expected


@pytest.mark.parametrize(
    "policy_payload",
    [
        {"thinking_policy": {"capability": "automatic"}},
        {"thinking_policy": {"on_unsupported": "warn"}},
        {"thinking_policy": {"on_mismatch": "fail"}},
        {"multimodal": {"fallback": "best_effort"}},
    ],
)
def test_policy_strategy_fields_reject_unknown_values(policy_payload: dict[str, object]) -> None:
    with pytest.raises(ValidationError):
        LiteLLMBackendConfig.model_validate({"model": "hosted_vllm/local", **policy_payload})


def test_route_mode_model_list_and_router_settings_parse_with_extra_router_fields() -> None:
    config = LiteLLMBackendConfig.model_validate(
        {
            "model": "Qwen/Qwen3.6-35B-A3B",
            "route_mode": "router",
            "model_list": [
                {
                    "model_name": "Qwen/Qwen3.6-35B-A3B",
                    "litellm_params": {
                        "model": "hosted_vllm/Qwen/Qwen3.6-35B-A3B",
                        "api_base": "http://127.0.0.1:8000/v1",
                        "api_key": "dummy",
                    },
                    "rpm": 60,
                }
            ],
            "router_settings": {
                "routing_strategy": "least-busy",
                "allowed_fails": 2,
                "cooldown_time": 30,
                "num_retries": 1,
                "retry_after": 5,
            },
        }
    )

    assert config.route_mode == "router"
    assert config.resolved_route_mode() == "router"
    assert config.model_list[0].model_name == "Qwen/Qwen3.6-35B-A3B"
    assert config.model_list[0].litellm_params["api_base"] == "http://127.0.0.1:8000/v1"
    assert config.model_list[0].rpm == 60
    assert config.router_settings.routing_strategy == "least-busy"
    assert config.router_settings.retry_after == 5


def test_model_list_implies_router_when_route_mode_is_not_explicit() -> None:
    config = LiteLLMBackendConfig.model_validate(
        {
            "model": "Qwen/Qwen3.6-35B-A3B",
            "model_list": [
                {
                    "model_name": "Qwen/Qwen3.6-35B-A3B",
                    "litellm_params": {"model": "hosted_vllm/Qwen/Qwen3.6-35B-A3B"},
                }
            ],
        }
    )

    assert config.route_mode == "direct"
    assert config.resolved_route_mode() == "router"


def test_explicit_direct_route_mode_keeps_model_list_on_direct_path() -> None:
    config = LiteLLMBackendConfig.model_validate(
        {
            "model": "Qwen/Qwen3.6-35B-A3B",
            "route_mode": "direct",
            "model_list": [
                {
                    "model_name": "Qwen/Qwen3.6-35B-A3B",
                    "litellm_params": {"model": "hosted_vllm/Qwen/Qwen3.6-35B-A3B"},
                }
            ],
        }
    )

    assert config.route_mode == "direct"
    assert config.resolved_route_mode() == "direct"


def test_litellm_request_design_passthrough_fields_parse() -> None:
    config = LiteLLMBackendConfig.model_validate(
        {
            "model": "hosted_vllm/Qwen/Qwen3.6-35B-A3B",
            "litellm_request": {
                "response_format": {"type": "json_object"},
                "max_completion_tokens": 1024,
                "stream_options": {"include_usage": True},
                "parallel_tool_calls": False,
                "logit_bias": {"42": -1},
                "user": "eval-user",
                "metadata": {"run_id": "run-001"},
                "fallbacks": [{"model": "openai/fallback"}],
                "context_window_fallback_dict": {"openai/large": "openai/small"},
                "extra_body": {"top_k": 20},
                "drop_params": False,
            },
        }
    )

    request = config.litellm_request
    assert request.response_format == {"type": "json_object"}
    assert request.max_completion_tokens == 1024
    assert request.stream_options == {"include_usage": True}
    assert request.parallel_tool_calls is False
    assert request.logit_bias == {"42": -1}
    assert request.user == "eval-user"
    assert request.metadata == {"run_id": "run-001"}
    assert request.fallbacks == [{"model": "openai/fallback"}]
    assert request.context_window_fallback_dict == {"openai/large": "openai/small"}
    assert request.extra_body == {"top_k": 20}
    assert request.drop_params is False


def test_tool_calling_design_fields_parse() -> None:
    config = LiteLLMBackendConfig.model_validate(
        {
            "model": "hosted_vllm/Qwen/Qwen3-Coder",
            "tool_calling": {
                "enabled": True,
                "tool_choice": "auto",
                "parallel_tool_calls": True,
                "auto_tool_choice": True,
                "require_server_parser": True,
                "expected_server_flags": {
                    "enable_auto_tool_choice": True,
                    "tool_call_parser": "qwen3_xml",
                },
            },
        }
    )

    tool_calling = config.tool_calling
    assert tool_calling.enabled is True
    assert tool_calling.tool_choice == "auto"
    assert tool_calling.parallel_tool_calls is True
    assert tool_calling.auto_tool_choice is True
    assert tool_calling.require_server_parser is True
    assert tool_calling.expected_server_flags.enable_auto_tool_choice is True
    assert tool_calling.expected_server_flags.tool_call_parser == "qwen3_xml"


def test_legacy_litellm_fields_still_parse() -> None:
    config = LiteLLMBackendConfig.model_validate(
        {
            "model": "openai/local",
            "streaming": True,
            "custom_llm_provider": "lm_studio",
            "mock_api_base": "http://127.0.0.1:1234/v1",
            "mock_api_key": "test-key",
            "mock_model": "local-model",
        }
    )

    assert config.streaming is True
    assert config.custom_llm_provider == "lm_studio"
    assert config.mock_api_base == "http://127.0.0.1:1234/v1"
    assert config.mock_api_key == "test-key"
    assert config.mock_model == "local-model"
