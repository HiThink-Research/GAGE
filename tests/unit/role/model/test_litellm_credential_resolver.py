from __future__ import annotations

from unittest import mock

import pytest

from gage_eval.role.model.backends.litellm.credential_resolver import ProviderCredentialResolver
from gage_eval.role.model.config.litellm import LiteLLMBackendConfig

pytestmark = pytest.mark.fast


def _resolve(config: dict[str, object], env: dict[str, str]) -> str | None:
    cfg = LiteLLMBackendConfig.model_validate(config)
    with mock.patch.dict("os.environ", env, clear=True):
        return ProviderCredentialResolver(cfg).resolve()


def test_non_openai_provider_does_not_use_openai_api_key() -> None:
    assert _resolve({"provider": "deepseek", "model": "deepseek-chat"}, {"OPENAI_API_KEY": "openai-key"}) is None


def test_known_provider_does_not_use_litellm_api_key_for_gateway_like_direct_host() -> None:
    assert (
        _resolve(
            {
                "provider": "deepseek",
                "model": "deepseek-chat",
                "api_base": "https://api-gateway.company.com/v1",
            },
            {"LITELLM_API_KEY": "gateway-key"},
        )
        is None
    )


def test_explicit_litellm_gateway_provider_can_use_litellm_api_key() -> None:
    assert _resolve({"provider": "gateway", "model": "qwen3"}, {"LITELLM_API_KEY": "gateway-key"}) == "gateway-key"


def test_external_gateway_topology_uses_litellm_api_key_before_openai_api_key() -> None:
    assert (
        _resolve(
            {
                "model": "openai/qwen3-group",
                "vllm": {"topology": {"kind": "external_gateway"}},
            },
            {"OPENAI_API_KEY": "openai-key", "LITELLM_API_KEY": "gateway-key"},
        )
        == "gateway-key"
    )


def test_default_local_litellm_gateway_does_not_fall_back_to_openai_api_key() -> None:
    assert (
        _resolve(
            {
                "model": "openai/qwen3-group",
                "api_base": "http://127.0.0.1:4000/v1",
            },
            {"OPENAI_API_KEY": "openai-key"},
        )
        == "dummy"
    )


def test_default_local_litellm_gateway_prefers_litellm_api_key() -> None:
    assert (
        _resolve(
            {
                "model": "openai/qwen3-group",
                "api_base": "http://127.0.0.1:4000/v1",
            },
            {"OPENAI_API_KEY": "openai-key", "LITELLM_API_KEY": "gateway-key"},
        )
        == "gateway-key"
    )


@pytest.mark.parametrize(
    ("config", "env", "expected"),
    [
        ({"provider": "deepseek", "model": "deepseek-chat"}, {"DEEPSEEK_API_KEY": "deepseek-key"}, "deepseek-key"),
        ({"provider": "kimi", "model": "moonshot-v1-8k"}, {"KIMI_API_KEY": "kimi-key"}, "kimi-key"),
        ({"provider": "kimi", "model": "moonshot-v1-8k"}, {"MOONSHOT_API_KEY": "moonshot-key"}, "moonshot-key"),
        ({"provider": "grok", "model": "grok-2"}, {"XAI_API_KEY": "xai-key"}, "xai-key"),
        ({"provider": "grok", "model": "grok-2"}, {"GROK_API_KEY": "grok-key"}, "grok-key"),
        ({"provider": "azure", "model": "azure:gpt-4o"}, {"AZURE_API_KEY": "azure-key"}, "azure-key"),
        (
            {"provider": "azure", "model": "azure:gpt-4o"},
            {"AZURE_OPENAI_API_KEY": "azure-openai-key"},
            "azure-openai-key",
        ),
        ({"provider": "openai", "model": "gpt-4o-mini"}, {"OPENAI_API_KEY": "openai-key"}, "openai-key"),
    ],
)
def test_provider_specific_env_precedence(config: dict[str, object], env: dict[str, str], expected: str) -> None:
    assert _resolve(config, {**env, "LITELLM_API_KEY": "gateway-key", "OPENAI_API_KEY": "openai-key"}) == expected


def test_explicit_api_key_has_highest_priority() -> None:
    assert (
        _resolve(
            {"provider": "deepseek", "model": "deepseek-chat", "api_key": "explicit-key"},
            {"DEEPSEEK_API_KEY": "deepseek-key"},
        )
        == "explicit-key"
    )


def test_hosted_vllm_without_key_uses_safe_dummy_key() -> None:
    assert _resolve({"model": "hosted_vllm/qwen3", "api_base": "http://127.0.0.1:8000/v1"}, {}) == "dummy"
