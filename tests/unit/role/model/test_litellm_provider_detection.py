from __future__ import annotations

import pytest

from gage_eval.role.model.backends.litellm.provider_detection import (
    infer_provider_from_model,
    is_default_local_litellm_gateway,
    is_explicit_litellm_gateway,
    looks_like_azure,
    looks_like_deepseek,
    looks_like_grok,
    looks_like_kimi,
    normalize_custom_provider,
    normalize_provider_name,
)

pytestmark = pytest.mark.fast


def test_infer_provider_from_model_preserves_backend_and_credential_resolver_modes() -> None:
    assert infer_provider_from_model("deepseek-chat") == "deepseek"
    assert infer_provider_from_model("hosted_vllm/qwen3") == "hosted_vllm"
    assert infer_provider_from_model("gpt-4o-mini") is None
    assert infer_provider_from_model("gpt-4o-mini", include_openai_family=True) == "openai"


def test_provider_normalization_and_custom_provider_aliases_match_existing_contract() -> None:
    assert normalize_provider_name("azure-openai") == "azure_openai"
    assert normalize_custom_provider("kimi") == "moonshot"
    assert normalize_custom_provider("grok") == "xai"
    assert normalize_custom_provider("azure_openai") == "azure"


def test_provider_family_detection_matches_existing_backend_heuristics() -> None:
    assert looks_like_deepseek("openai", "deepseek-chat")
    assert looks_like_deepseek(None, "chat", "https://api.deepseek.com")
    assert looks_like_kimi(None, "moonshot-v1-8k")
    assert looks_like_grok(None, "grok-2")
    assert looks_like_azure(None, "azure:gpt-4o")


def test_litellm_gateway_detection_requires_explicit_profile_or_default_local_port() -> None:
    assert is_explicit_litellm_gateway(topology_kind="external_gateway")
    assert is_default_local_litellm_gateway("http://127.0.0.1:4000/v1")
    assert not is_default_local_litellm_gateway("https://api-gateway.company.com/v1")
