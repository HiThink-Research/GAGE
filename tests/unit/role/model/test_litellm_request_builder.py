from __future__ import annotations

import pytest

from gage_eval.role.model.backends.litellm.request_builder import (
    LITELLM_REQUEST_PASSTHROUGH_FIELDS,
    LiteLLMRequestBuilder,
)
from gage_eval.role.model.backends.litellm.policies import ToolCallPolicy
from gage_eval.role.model.config.litellm import LiteLLMRequestConfig, LiteLLMToolCallingConfig

pytestmark = pytest.mark.fast


def _builder(**overrides: object) -> LiteLLMRequestBuilder:
    params = {
        "model_name": "hosted_vllm/demo-model",
        "provider": "hosted_vllm",
        "api_base": "http://127.0.0.1:8000/v1",
        "api_key": "dummy",
        "timeout": 30.0,
        "headers": {"X-Test": "1"},
        "custom_llm_provider": "hosted_vllm",
        "base_sampling": {"max_new_tokens": 256, "temperature": 0.2},
        "litellm_request": LiteLLMRequestConfig(),
        "drop_params": True,
    }
    params.update(overrides)
    return LiteLLMRequestBuilder(**params)


def test_response_format_is_only_sent_when_configured() -> None:
    kwargs, _ = _builder().build({"messages": [{"role": "user", "content": "ping"}]})

    assert "response_format" not in kwargs

    request = LiteLLMRequestConfig(response_format={"type": "json_object"})
    configured_kwargs, _ = _builder(litellm_request=request).build(
        {"messages": [{"role": "user", "content": "ping"}]}
    )

    assert configured_kwargs["response_format"] == {"type": "json_object"}


def test_extra_body_and_drop_params_are_request_level_kwargs() -> None:
    request = LiteLLMRequestConfig(extra_body={"top_k": 20}, drop_params=False)

    kwargs, _ = _builder(litellm_request=request, drop_params=True).build(
        {"messages": [{"role": "user", "content": "ping"}]}
    )

    assert kwargs["extra_body"] == {"top_k": 20}
    assert kwargs["drop_params"] is False


def test_litellm_request_overrides_generation_parameter_conflicts() -> None:
    request = LiteLLMRequestConfig(
        max_completion_tokens=128,
        metadata={"source": "litellm_request"},
        response_format={"type": "json_object"},
    )

    kwargs, _ = _builder(litellm_request=request).build(
        {
            "messages": [{"role": "user", "content": "ping"}],
            "sampling_params": {
                "max_completion_tokens": 64,
                "metadata": {"source": "generation_parameters"},
                "response_format": {"type": "text"},
            },
        }
    )

    assert kwargs["max_completion_tokens"] == 128
    assert kwargs["metadata"] == {"source": "litellm_request"}
    assert kwargs["response_format"] == {"type": "json_object"}


def test_endpoint_api_base_raises_clear_error() -> None:
    builder = _builder(api_base="http://127.0.0.1:8000/v1/chat/completions")

    with pytest.raises(ValueError, match="api_base.*base URL.*\\/v1.*not.*endpoint.*error_type=invalid_request"):
        builder.build({"messages": [{"role": "user", "content": "ping"}]})


def test_custom_llm_provider_is_forwarded() -> None:
    kwargs, _ = _builder(custom_llm_provider="hosted_vllm").build(
        {"messages": [{"role": "user", "content": "ping"}]}
    )

    assert kwargs["custom_llm_provider"] == "hosted_vllm"


def test_tools_are_formatted_and_tool_choice_is_forwarded() -> None:
    kwargs, _ = _builder().build(
        {
            "messages": [{"role": "user", "content": "ping"}],
            "tools": [
                {
                    "name": "lookup",
                    "description": "Lookup an item",
                    "parameters": {
                        "type": "object",
                        "properties": {"query": {"type": "string"}},
                    },
                }
            ],
            "tool_choice": {"type": "function", "function": {"name": "lookup"}},
        }
    )

    assert kwargs["tools"] == [
        {
            "type": "function",
            "function": {
                "name": "lookup",
                "description": "Lookup an item",
                "parameters": {
                    "type": "object",
                    "properties": {"query": {"type": "string"}},
                },
            },
        }
    ]
    assert kwargs["tool_choice"] == {"type": "function", "function": {"name": "lookup"}}


def test_openai_tool_schema_is_preserved_and_parallel_tool_calls_can_pass_through() -> None:
    request = LiteLLMRequestConfig(parallel_tool_calls=True)
    policy = ToolCallPolicy(
        LiteLLMToolCallingConfig(
            expected_server_flags={"enable_auto_tool_choice": True, "tool_call_parser": "qwen3_xml"},
        )
    )

    kwargs, _ = _builder(litellm_request=request, tool_call_policy=policy).build(
        {
            "messages": [{"role": "user", "content": "ping"}],
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": "lookup",
                        "description": "Lookup an item",
                        "parameters": {"type": "object", "properties": {}},
                        "strict": True,
                    },
                }
            ],
            "tool_choice": "auto",
        }
    )

    assert kwargs["tools"] == [
        {
            "type": "function",
            "function": {
                "name": "lookup",
                "description": "Lookup an item",
                "parameters": {"type": "object", "properties": {}},
                "strict": True,
            },
        }
    ]
    assert kwargs["tool_choice"] == "auto"
    assert kwargs["parallel_tool_calls"] is True


def test_tool_calling_config_defaults_are_used_when_inputs_omit_them() -> None:
    policy = ToolCallPolicy(
        LiteLLMToolCallingConfig(
            tool_choice="auto",
            parallel_tool_calls=True,
            expected_server_flags={"enable_auto_tool_choice": True, "tool_call_parser": "qwen3_xml"},
        )
    )

    kwargs, _ = _builder(tool_call_policy=policy).build(
        {
            "messages": [{"role": "user", "content": "ping"}],
            "tools": [{"name": "lookup", "parameters": {"type": "object", "properties": {}}}],
        }
    )

    assert kwargs["tool_choice"] == "auto"
    assert kwargs["parallel_tool_calls"] is True


def test_tool_calling_inputs_override_config_defaults() -> None:
    policy = ToolCallPolicy(
        LiteLLMToolCallingConfig(
            tool_choice="auto",
            parallel_tool_calls=True,
            expected_server_flags={"enable_auto_tool_choice": True, "tool_call_parser": "qwen3_xml"},
        )
    )

    kwargs, _ = _builder(tool_call_policy=policy).build(
        {
            "messages": [{"role": "user", "content": "ping"}],
            "tools": [{"name": "lookup", "parameters": {"type": "object", "properties": {}}}],
            "tool_choice": {"type": "function", "function": {"name": "lookup"}},
            "parallel_tool_calls": False,
        }
    )

    assert kwargs["tool_choice"] == {"type": "function", "function": {"name": "lookup"}}
    assert kwargs["parallel_tool_calls"] is False


def test_auto_tool_choice_requires_expected_server_parser() -> None:
    with pytest.raises(ValueError, match="tool_parser_missing"):
        _builder().build(
            {
                "messages": [{"role": "user", "content": "ping"}],
                "tools": [{"name": "lookup", "parameters": {"type": "object", "properties": {}}}],
                "tool_choice": "auto",
            }
        )


def test_auto_tool_choice_requires_enable_auto_tool_choice_flag() -> None:
    policy = ToolCallPolicy(
        LiteLLMToolCallingConfig(
            expected_server_flags={"tool_call_parser": "qwen3_xml"},
        )
    )

    with pytest.raises(ValueError, match="tool_parser_missing.*enable_auto_tool_choice"):
        _builder(tool_call_policy=policy).build(
            {
                "messages": [{"role": "user", "content": "ping"}],
                "tools": [{"name": "lookup", "parameters": {"type": "object", "properties": {}}}],
                "tool_choice": "auto",
            }
        )


def test_auto_tool_choice_rejects_false_enable_auto_tool_choice_flag() -> None:
    policy = ToolCallPolicy(
        LiteLLMToolCallingConfig(
            expected_server_flags={"enable_auto_tool_choice": False, "tool_call_parser": "qwen3_xml"},
        )
    )

    with pytest.raises(ValueError, match="tool_parser_missing.*enable_auto_tool_choice"):
        _builder(tool_call_policy=policy).build(
            {
                "messages": [{"role": "user", "content": "ping"}],
                "tools": [{"name": "lookup", "parameters": {"type": "object", "properties": {}}}],
                "tool_choice": "auto",
            }
        )


def test_hosted_vllm_with_declared_tool_flags_allows_false_static_function_calling_helper() -> None:
    policy = ToolCallPolicy(
        LiteLLMToolCallingConfig(
            expected_server_flags={"enable_auto_tool_choice": True, "tool_call_parser": "qwen3_xml"},
        )
    )

    kwargs, _ = _builder(tool_call_policy=policy, supports_function_calling=False).build(
        {
            "messages": [{"role": "user", "content": "ping"}],
            "tools": [{"name": "lookup", "parameters": {"type": "object", "properties": {}}}],
            "tool_choice": "auto",
        }
    )

    assert kwargs["tool_choice"] == "auto"
    assert kwargs["tools"][0]["function"]["name"] == "lookup"


def test_remote_openai_compatible_with_declared_tool_flags_allows_false_static_helper() -> None:
    policy = ToolCallPolicy(
        LiteLLMToolCallingConfig(
            expected_server_flags={"enable_auto_tool_choice": True, "tool_call_parser": "qwen3_xml"},
        )
    )

    kwargs, _ = _builder(
        model_name="openai/demo-model",
        provider="openai",
        custom_llm_provider="openai",
        api_base="https://vllm.example.com/v1",
        tool_call_policy=policy,
        supports_function_calling=False,
    ).build(
        {
            "messages": [{"role": "user", "content": "ping"}],
            "tools": [{"name": "lookup", "parameters": {"type": "object", "properties": {}}}],
            "tool_choice": "auto",
        }
    )

    assert kwargs["model"] == "openai/demo-model"
    assert kwargs["api_base"] == "https://vllm.example.com/v1"
    assert kwargs["tool_choice"] == "auto"
    assert kwargs["tools"][0]["function"]["name"] == "lookup"


def test_config_auto_tool_choice_does_not_pollute_explicit_named_tool_choice() -> None:
    policy = ToolCallPolicy(LiteLLMToolCallingConfig(auto_tool_choice=True))
    named_choice = {"type": "function", "function": {"name": "lookup"}}

    kwargs, _ = _builder(tool_call_policy=policy).build(
        {
            "messages": [{"role": "user", "content": "ping"}],
            "tools": [{"name": "lookup", "parameters": {"type": "object", "properties": {}}}],
            "tool_choice": named_choice,
        }
    )

    assert kwargs["tool_choice"] == named_choice


def test_openai_auto_tool_choice_does_not_require_vllm_server_flags() -> None:
    kwargs, _ = _builder(
        model_name="openai/gpt-4o-mini",
        provider="openai",
        custom_llm_provider="openai",
        api_base=None,
        supports_function_calling=True,
    ).build(
        {
            "messages": [{"role": "user", "content": "ping"}],
            "tools": [{"name": "lookup", "parameters": {"type": "object", "properties": {}}}],
            "tool_choice": "auto",
        }
    )

    assert kwargs["tool_choice"] == "auto"
    assert kwargs["tools"][0]["function"]["name"] == "lookup"


def test_auto_tool_choice_with_expected_server_flags_parser_builds() -> None:
    policy = ToolCallPolicy(
        LiteLLMToolCallingConfig(
            expected_server_flags={"enable_auto_tool_choice": True, "tool_call_parser": "qwen3_xml"},
        )
    )

    kwargs, _ = _builder(tool_call_policy=policy).build(
        {
            "messages": [{"role": "user", "content": "ping"}],
            "tools": [{"name": "lookup", "parameters": {"type": "object", "properties": {}}}],
            "tool_choice": "auto",
        }
    )

    assert kwargs["tool_choice"] == "auto"


def test_legacy_expected_tool_parser_satisfies_auto_tool_choice_parser_requirement() -> None:
    policy = ToolCallPolicy(
        LiteLLMToolCallingConfig(
            auto_tool_choice=True,
            expected_tool_parser="qwen3_xml",
            expected_server_flags={"enable_auto_tool_choice": True},
        )
    )

    kwargs, _ = _builder(tool_call_policy=policy).build(
        {
            "messages": [{"role": "user", "content": "ping"}],
            "tools": [{"name": "lookup", "parameters": {"type": "object", "properties": {}}}],
        }
    )

    assert kwargs["tool_choice"] == "auto"


def test_function_calling_not_supported_fails_when_tools_requested() -> None:
    with pytest.raises(ValueError, match="tool_calling_not_supported"):
        _builder(supports_function_calling=False).build(
            {
                "messages": [{"role": "user", "content": "ping"}],
                "tools": [{"name": "lookup", "parameters": {"type": "object", "properties": {}}}],
            }
        )


def test_unknown_function_calling_support_allows_tools_for_compatibility() -> None:
    kwargs, _ = _builder(supports_function_calling=None).build(
        {
            "messages": [{"role": "user", "content": "ping"}],
            "tools": [{"name": "lookup", "parameters": {"type": "object", "properties": {}}}],
        }
    )

    assert kwargs["tools"][0]["function"]["name"] == "lookup"


def test_tool_policy_fails_fast_when_required_server_parser_is_missing() -> None:
    policy = ToolCallPolicy(
        LiteLLMToolCallingConfig(
            enabled=True,
            require_server_parser=True,
            expected_tool_parser=None,
        )
    )
    builder = _builder(tool_call_policy=policy)

    with pytest.raises(ValueError, match="tool_parser_missing"):
        builder.build(
            {
                "messages": [{"role": "user", "content": "ping"}],
                "tools": [{"name": "lookup", "parameters": {"type": "object", "properties": {}}}],
            }
        )


def test_deepseek_model_is_prefixed_for_native_provider() -> None:
    kwargs, _ = _builder(
        model_name="deepseek-chat",
        provider="deepseek",
        api_base="https://api.deepseek.com",
        custom_llm_provider="deepseek",
    ).build({"messages": [{"role": "user", "content": "ping"}]})

    assert kwargs["model"] == "deepseek/deepseek-chat"
    assert kwargs["custom_llm_provider"] == "deepseek"


def test_azure_request_includes_api_type_and_version() -> None:
    kwargs, _ = _builder(
        model_name="azure:gpt-4o-mini",
        provider="azure",
        api_base="https://demo-openai.eastus.azure.com",
        custom_llm_provider="azure",
        is_azure_target=True,
        azure_api_version="2024-06-01-preview",
    ).build({"messages": [{"role": "user", "content": "ping"}]})

    assert kwargs["api_type"] == "azure"
    assert kwargs["api_version"] == "2024-06-01-preview"


def test_stop_max_tokens_and_n_are_normalized_from_sampling_params() -> None:
    kwargs, sampling_params = _builder().build(
        {
            "messages": [{"role": "user", "content": "ping"}],
            "sampling_params": {
                "max_new_tokens": 64,
                "stop": "END",
                "n": 3,
            },
        }
    )

    assert sampling_params["max_tokens"] == 64
    assert sampling_params["stop"] == "END"
    assert sampling_params["n"] == 3
    assert kwargs["max_tokens"] == 64
    assert kwargs["stop"] == ["END"]
    assert kwargs["n"] == 3


def test_all_litellm_request_passthrough_fields_are_whitelisted_and_forwarded() -> None:
    request_payload = {
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
    }
    request = LiteLLMRequestConfig(**request_payload)

    kwargs, _ = _builder(litellm_request=request).build(
        {"messages": [{"role": "user", "content": "ping"}]}
    )

    assert set(LITELLM_REQUEST_PASSTHROUGH_FIELDS) == set(request_payload)
    for field, value in request_payload.items():
        assert kwargs[field] == value


def test_litellm_request_extra_fields_are_preserved_but_not_forwarded() -> None:
    request = LiteLLMRequestConfig(foo="bar", extra_body={"top_k": 20})

    kwargs, _ = _builder(litellm_request=request).build(
        {"messages": [{"role": "user", "content": "ping"}]}
    )

    assert getattr(request, "foo") == "bar"
    assert kwargs["extra_body"] == {"top_k": 20}
    assert "foo" not in kwargs


def test_extra_body_passthrough_is_deep_copied() -> None:
    request = LiteLLMRequestConfig(
        extra_body={"chat_template_kwargs": {"enable_thinking": False, "nested": {"value": 1}}}
    )

    kwargs, _ = _builder(litellm_request=request).build(
        {"messages": [{"role": "user", "content": "ping"}]}
    )
    kwargs["extra_body"]["chat_template_kwargs"]["nested"]["value"] = 2

    assert request.extra_body["chat_template_kwargs"]["nested"]["value"] == 1


def test_runtime_sampling_unknown_high_risk_fields_are_not_forwarded() -> None:
    kwargs, _ = _builder().build(
        {
            "messages": [{"role": "user", "content": "ping"}],
            "sampling_params": {
                "max_new_tokens": 64,
                "temperature": 0.1,
                "response_format": {"type": "json_object"},
                "modalities": ["audio"],
                "audio": {"voice": "alloy"},
                "prediction": {"type": "content", "content": "cached"},
                "foo": "bar",
            },
        }
    )

    assert kwargs["max_tokens"] == 64
    assert kwargs["temperature"] == 0.1
    assert kwargs["response_format"] == {"type": "json_object"}
    assert "modalities" not in kwargs
    assert "audio" not in kwargs
    assert "prediction" not in kwargs
    assert "foo" not in kwargs
