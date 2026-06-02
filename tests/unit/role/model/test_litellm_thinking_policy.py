from __future__ import annotations

import pytest

from gage_eval.role.model.backends.litellm.policies import ThinkingControlPolicy
from gage_eval.role.model.backends.litellm.request_builder import LiteLLMRequestBuilder
from gage_eval.role.model.backends.litellm.service_profile import VLLMServiceProfile
from gage_eval.role.model.config.litellm import LiteLLMRequestConfig

pytestmark = pytest.mark.fast


def _profile(reasoning_parser: str | None = "qwen3") -> VLLMServiceProfile:
    return VLLMServiceProfile(
        mode="direct",
        model="hosted_vllm/Qwen/Qwen3-8B",
        api_base="http://127.0.0.1:8000/v1",
        reasoning_parser=reasoning_parser,
    )


def _builder(**overrides: object) -> LiteLLMRequestBuilder:
    params = {
        "model_name": "hosted_vllm/Qwen/Qwen3-8B",
        "provider": "hosted_vllm",
        "api_base": "http://127.0.0.1:8000/v1",
        "api_key": "dummy",
        "custom_llm_provider": "hosted_vllm",
        "base_sampling": {"max_new_tokens": 128},
        "litellm_request": LiteLLMRequestConfig(),
        "drop_params": True,
        "supports_reasoning": False,
        "max_context_length": None,
        "thinking_policy": ThinkingControlPolicy(service_profile=_profile()),
    }
    params.update(overrides)
    return LiteLLMRequestBuilder(**params)


def _messages() -> dict:
    return {"messages": [{"role": "user", "content": "solve 2+2"}]}


def test_qwen3_vllm_disabled_uses_extra_body_not_top_level() -> None:
    kwargs, _ = _builder().build(_messages(), thinking_config={"thinking_mode": "disabled"})

    assert "enable_thinking" not in kwargs
    assert kwargs["extra_body"]["chat_template_kwargs"]["enable_thinking"] is False


def test_qwen3_vllm_enabled_uses_extra_body_true() -> None:
    kwargs, _ = _builder().build(_messages(), thinking_config={"thinking_mode": "enabled"})

    assert "enable_thinking" not in kwargs
    assert kwargs["extra_body"]["chat_template_kwargs"]["enable_thinking"] is True


def test_thinking_mode_none_does_not_inject_enable_thinking() -> None:
    kwargs, _ = _builder().build(_messages(), thinking_config={"thinking_mode": None})

    assert "enable_thinking" not in kwargs
    assert "extra_body" not in kwargs or "enable_thinking" not in kwargs["extra_body"].get("chat_template_kwargs", {})


def test_auto_vllm_with_reasoning_parser_records_server_default_inheritance() -> None:
    policy = ThinkingControlPolicy(service_profile=_profile(reasoning_parser="qwen3"))

    resolution = policy.resolve(
        model="hosted_vllm/Qwen/Qwen3-8B",
        provider="hosted_vllm",
        custom_llm_provider="hosted_vllm",
        api_base="http://127.0.0.1:8000/v1",
        thinking_config={"thinking_mode": "auto"},
    )

    assert resolution.mode == "auto"
    assert resolution.kwargs_patch == {}
    assert resolution.metadata["thinking_inherited_from_server_default"] is True


def test_reasoning_effort_remains_top_level_litellm_kwarg() -> None:
    kwargs, _ = _builder().build(
        {
            "messages": [{"role": "user", "content": "think hard"}],
            "sampling_params": {"reasoning_effort": "high"},
        },
        thinking_config={"thinking_mode": "auto"},
    )

    assert kwargs["reasoning_effort"] == "high"


def test_disabled_thinking_with_reasoning_model_does_not_expand_max_tokens() -> None:
    kwargs, _ = _builder(supports_reasoning=True).build(
        {
            "messages": [{"role": "user", "content": "short answer"}],
            "sampling_params": {"max_new_tokens": 128},
        },
        thinking_config={"thinking_mode": "disabled"},
    )

    assert kwargs["max_tokens"] == 128


def test_enabled_thinking_with_reasoning_model_may_expand_max_tokens() -> None:
    kwargs, _ = _builder(supports_reasoning=True, max_context_length=1000).build(
        {
            "messages": [{"role": "user", "content": "show reasoning"}],
            "sampling_params": {"max_new_tokens": 128},
        },
        thinking_config={"thinking_mode": "enabled"},
    )

    assert kwargs["max_tokens"] == 1000


def test_enabled_vllm_without_reasoning_parser_fail_fast_raises() -> None:
    policy = ThinkingControlPolicy(service_profile=_profile(reasoning_parser=None), on_unsupported="fail_fast")

    with pytest.raises(ValueError, match="thinking.*reasoning_parser"):
        policy.resolve(
            model="hosted_vllm/Qwen/Qwen3-8B",
            provider="hosted_vllm",
            custom_llm_provider="hosted_vllm",
            thinking_config={"thinking_mode": "enabled"},
        )


def test_enabled_vllm_without_reasoning_parser_default_warns_but_builds() -> None:
    builder = _builder(thinking_policy=ThinkingControlPolicy(service_profile=_profile(reasoning_parser=None)))

    kwargs, _ = builder.build(_messages(), thinking_config={"thinking_mode": "enabled"})

    assert kwargs["extra_body"]["chat_template_kwargs"]["enable_thinking"] is True


def test_capability_unsupported_enabled_fail_fast_raises() -> None:
    policy = ThinkingControlPolicy(
        service_profile=_profile(),
        capability="unsupported",
        on_unsupported="fail_fast",
    )

    with pytest.raises(ValueError, match="thinking.*unsupported"):
        policy.resolve(
            model="hosted_vllm/Qwen/Qwen3-8B",
            provider="hosted_vllm",
            custom_llm_provider="hosted_vllm",
            thinking_config={"thinking_mode": "enabled"},
        )


def test_capability_unsupported_enabled_default_builds_without_vllm_enable_thinking() -> None:
    builder = _builder(thinking_policy=ThinkingControlPolicy(service_profile=_profile(), capability="unsupported"))

    kwargs, _ = builder.build(_messages(), thinking_config={"thinking_mode": "enabled"})

    assert "extra_body" not in kwargs or "enable_thinking" not in kwargs["extra_body"].get("chat_template_kwargs", {})


def test_capability_supported_hosted_vllm_enabled_without_parser_injects_without_fail_fast() -> None:
    builder = _builder(
        thinking_policy=ThinkingControlPolicy(
            service_profile=_profile(reasoning_parser=None),
            capability="supported",
            on_unsupported="fail_fast",
        )
    )

    kwargs, _ = builder.build(_messages(), thinking_config={"thinking_mode": "enabled"})

    assert kwargs["extra_body"]["chat_template_kwargs"]["enable_thinking"] is True


def test_explicit_litellm_request_extra_body_overrides_policy_enable_thinking() -> None:
    request = LiteLLMRequestConfig(extra_body={"chat_template_kwargs": {"enable_thinking": False}})

    kwargs, _ = _builder(litellm_request=request).build(_messages(), thinking_config={"thinking_mode": "enabled"})

    assert kwargs["extra_body"]["chat_template_kwargs"]["enable_thinking"] is False


def test_explicit_litellm_request_disable_controls_reasoning_max_tokens() -> None:
    request = LiteLLMRequestConfig(extra_body={"chat_template_kwargs": {"enable_thinking": False}})

    kwargs, _ = _builder(litellm_request=request, supports_reasoning=True).build(
        {
            "messages": [{"role": "user", "content": "short answer"}],
            "sampling_params": {"max_new_tokens": 128},
        },
        thinking_config={"thinking_mode": "enabled"},
    )

    assert kwargs["extra_body"]["chat_template_kwargs"]["enable_thinking"] is False
    assert kwargs["max_tokens"] == 128


def test_explicit_litellm_request_enable_controls_reasoning_max_tokens() -> None:
    request = LiteLLMRequestConfig(extra_body={"chat_template_kwargs": {"enable_thinking": True}})

    kwargs, _ = _builder(litellm_request=request, supports_reasoning=True, max_context_length=1000).build(
        {
            "messages": [{"role": "user", "content": "show reasoning"}],
            "sampling_params": {"max_new_tokens": 128},
        },
        thinking_config={"thinking_mode": "disabled"},
    )

    assert kwargs["extra_body"]["chat_template_kwargs"]["enable_thinking"] is True
    assert kwargs["max_tokens"] == 1000


def test_runtime_extra_body_disable_controls_reasoning_max_tokens() -> None:
    kwargs, _ = _builder(supports_reasoning=True).build(
        {
            "messages": [{"role": "user", "content": "short answer"}],
            "sampling_params": {
                "max_new_tokens": 128,
                "extra_body": {"chat_template_kwargs": {"enable_thinking": False}},
            },
        },
        thinking_config={"thinking_mode": "enabled"},
    )

    assert kwargs["extra_body"]["chat_template_kwargs"]["enable_thinking"] is False
    assert kwargs["max_tokens"] == 128


def test_runtime_extra_body_enable_controls_reasoning_max_tokens() -> None:
    kwargs, _ = _builder(supports_reasoning=True, max_context_length=1000).build(
        {
            "messages": [{"role": "user", "content": "show reasoning"}],
            "sampling_params": {
                "max_new_tokens": 128,
                "extra_body": {"chat_template_kwargs": {"enable_thinking": True}},
            },
        },
        thinking_config={"thinking_mode": "disabled"},
    )

    assert kwargs["extra_body"]["chat_template_kwargs"]["enable_thinking"] is True
    assert kwargs["max_tokens"] == 1000


def test_openai_provider_with_openai_api_base_does_not_inject_vllm_extra_body() -> None:
    kwargs, _ = _builder(
        model_name="gpt-4o-mini",
        provider="openai",
        custom_llm_provider="openai",
        api_base="https://api.openai.com/v1",
        thinking_policy=ThinkingControlPolicy(service_profile=_profile(reasoning_parser="qwen3")),
    ).build(_messages(), thinking_config={"thinking_mode": "enabled"})

    assert "extra_body" not in kwargs


def test_openai_provider_with_local_api_base_uses_profile_parser_as_vllm_signal() -> None:
    kwargs, _ = _builder(
        model_name="Qwen/Qwen3-8B",
        provider="openai",
        custom_llm_provider="openai",
        api_base="http://127.0.0.1:4000/v1",
        thinking_policy=ThinkingControlPolicy(service_profile=_profile(reasoning_parser="qwen3")),
    ).build(_messages(), thinking_config={"thinking_mode": "enabled"})

    assert kwargs["extra_body"]["chat_template_kwargs"]["enable_thinking"] is True


def test_openai_api_base_without_provider_does_not_use_profile_parser_as_vllm_signal() -> None:
    kwargs, _ = _builder(
        model_name="gpt-4o-mini",
        provider=None,
        custom_llm_provider=None,
        api_base="https://api.openai.com/v1",
        thinking_policy=ThinkingControlPolicy(service_profile=_profile(reasoning_parser="qwen3")),
    ).build(_messages(), thinking_config={"thinking_mode": "enabled"})

    assert "extra_body" not in kwargs


def test_azure_openai_api_base_without_provider_does_not_use_profile_parser_as_vllm_signal() -> None:
    kwargs, _ = _builder(
        model_name="gpt-4o-mini",
        provider=None,
        custom_llm_provider=None,
        api_base="https://demo.openai.azure.com/openai/deployments/gpt-4o",
        thinking_policy=ThinkingControlPolicy(service_profile=_profile(reasoning_parser="qwen3")),
    ).build(_messages(), thinking_config={"thinking_mode": "enabled"})

    assert "extra_body" not in kwargs


def test_runtime_sampling_thinking_mode_overrides_backend_thinking_config() -> None:
    kwargs, _ = _builder().build(
        {
            "messages": [{"role": "user", "content": "short answer"}],
            "sampling_params": {"thinking_mode": "disabled"},
        },
        thinking_config={"thinking_mode": "enabled"},
    )

    assert kwargs["extra_body"]["chat_template_kwargs"]["enable_thinking"] is False
