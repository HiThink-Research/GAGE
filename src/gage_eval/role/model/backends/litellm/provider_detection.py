"""Provider detection helpers for LiteLLM-compatible backends."""

from __future__ import annotations

from urllib.parse import urlsplit


LOCAL_API_HOSTS = frozenset({"127.0.0.1", "localhost", "0.0.0.0", "::1"})  # nosec
LITELLM_GATEWAY_PROFILES = frozenset(
    {
        "external_gateway",
        "external_litellm_gateway",
        "gateway",
        "litellm",
        "litellm_gateway",
        "proxy",
    }
)


def normalize_provider_name(value: str | None) -> str:
    return (value or "").strip().lower().replace("-", "_")


def normalize_custom_provider(provider: str | None) -> str | None:
    if not provider:
        return None
    lower = provider.lower()
    alias_map = {
        "deepseek": "deepseek",
        "kimi": "moonshot",
        "moonshot": "moonshot",
        "grok": "xai",
        "xai": "xai",
        "google": "gemini",
        "gemini": "gemini",
        "azure_openai": "azure",
        "google_genai": "google",
    }
    return alias_map.get(lower, lower)


def infer_provider_from_model(
    model: str | None,
    *,
    include_openai_family: bool = False,
) -> str | None:
    if not model:
        return None
    if "/" in model:
        return model.split("/", 1)[0]
    lower = model.lower()
    if lower.startswith("deepseek"):
        return "deepseek"
    if lower.startswith("grok"):
        return "grok"
    if lower.startswith("moonshot"):
        return "kimi"
    if lower.startswith("azure:"):
        return "azure"
    if include_openai_family and lower.startswith(("gpt-", "o1", "o3", "o4")):
        return "openai"
    return None


def looks_like_deepseek(provider: str | None, model: str, api_base: str | None = None) -> bool:
    target = (provider or "").lower()
    model_lower = (model or "").lower()
    base = (api_base or "").lower()
    return target == "deepseek" or model_lower.startswith("deepseek") or "deepseek.com" in base


def looks_like_kimi(provider: str | None, model: str, api_base: str | None = None) -> bool:
    target = (provider or "").lower()
    model_lower = (model or "").lower()
    base = (api_base or "").lower()
    return target in {"kimi", "moonshot"} or model_lower.startswith("moonshot") or "moonshot" in base or "kimi" in base


def looks_like_grok(provider: str | None, model: str, api_base: str | None = None) -> bool:
    target = (provider or "").lower()
    model_lower = (model or "").lower()
    base = (api_base or "").lower()
    return target in {"grok", "xai"} or model_lower.startswith("grok") or "api.x.ai" in base or base.endswith("x.ai")


def looks_like_azure(provider: str | None, model: str, api_base: str | None = None) -> bool:
    target = (provider or "").lower()
    model_lower = (model or "").lower()
    base = (api_base or "").lower()
    return target in {"azure", "azure_openai"} or "openai.azure.com" in base or "azure" in base or model_lower.startswith("azure:")


def is_explicit_litellm_gateway(
    provider: str | None = None,
    *,
    custom_llm_provider: str | None = None,
    topology_kind: str | None = None,
) -> bool:
    """Return whether config explicitly declares an external LiteLLM Gateway."""

    return any(
        normalize_provider_name(value) in LITELLM_GATEWAY_PROFILES
        for value in (provider, custom_llm_provider, topology_kind)
    )


def is_default_local_litellm_gateway(api_base: str | None) -> bool:
    """Return whether ``api_base`` matches LiteLLM Gateway's common local endpoint."""

    if not api_base:
        return False
    parsed = urlsplit(api_base)
    host = (parsed.hostname or "").lower()
    return host in LOCAL_API_HOSTS and parsed.port == 4000


def is_local_api_base(api_base: str | None) -> bool:
    if not api_base:
        return False
    host = (urlsplit(api_base).hostname or "").lower()
    return host in LOCAL_API_HOSTS
