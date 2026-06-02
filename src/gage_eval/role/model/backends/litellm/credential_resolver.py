"""Provider-aware credential resolution for LiteLLM backend."""

from __future__ import annotations

import os
from typing import Iterable

from gage_eval.role.model.backends.litellm.provider_detection import (
    infer_provider_from_model,
    is_default_local_litellm_gateway,
    is_explicit_litellm_gateway,
    is_local_api_base,
    normalize_provider_name,
)
from gage_eval.role.model.config.litellm import LiteLLMBackendConfig


SAFE_DUMMY_CREDENTIAL = "dummy"


class ProviderCredentialResolver:
    """Resolve API keys without leaking one provider's env key to another."""

    _PROVIDER_ENV_KEYS = {
        "deepseek": ("DEEPSEEK_API_KEY",),
        "kimi": ("KIMI_API_KEY", "MOONSHOT_API_KEY"),
        "moonshot": ("KIMI_API_KEY", "MOONSHOT_API_KEY"),
        "grok": ("XAI_API_KEY", "GROK_API_KEY"),
        "xai": ("XAI_API_KEY", "GROK_API_KEY"),
        "azure": ("AZURE_API_KEY", "AZURE_OPENAI_API_KEY"),
        "azure_openai": ("AZURE_API_KEY", "AZURE_OPENAI_API_KEY"),
        "gemini": ("GEMINI_API_KEY", "GOOGLE_API_KEY"),
        "google": ("GEMINI_API_KEY", "GOOGLE_API_KEY"),
        "anthropic": ("ANTHROPIC_API_KEY",),
        "openai": ("OPENAI_API_KEY",),
    }

    _GENERIC_ENV_KEYS = ("LITELLM_API_KEY",)
    _OPENAI_COMPATIBLE_PROVIDERS = {"hosted_vllm", "vllm", "openai_compatible"}

    def __init__(
        self,
        cfg: LiteLLMBackendConfig,
        *,
        provider: str | None = None,
        custom_llm_provider: str | None = None,
        topology_kind: str | None = None,
    ) -> None:
        self._cfg = cfg
        self._provider = normalize_provider_name(
            provider or cfg.provider or infer_provider_from_model(cfg.model, include_openai_family=True)
        )
        self._custom_llm_provider = normalize_provider_name(custom_llm_provider or cfg.custom_llm_provider)
        topology = getattr(getattr(cfg, "vllm", None), "topology", None)
        self._topology_kind = topology_kind if topology_kind is not None else getattr(topology, "kind", None)

    def resolve(self) -> str | None:
        """Return the effective key, a safe dummy key, or ``None`` when absent."""

        explicit = self._first_present((self._cfg.api_key, self._cfg.mock_api_key))
        if explicit:
            return explicit

        env_key = self._first_env(self._env_candidates())
        if env_key:
            return env_key

        if self._allows_dummy_key():
            return SAFE_DUMMY_CREDENTIAL
        return None

    def _env_candidates(self) -> tuple[str, ...]:
        provider = self._effective_provider()
        if is_explicit_litellm_gateway(
            provider,
            custom_llm_provider=self._custom_llm_provider,
            topology_kind=self._topology_kind,
        ) or is_default_local_litellm_gateway(self._cfg.api_base or self._cfg.mock_api_base):
            return self._GENERIC_ENV_KEYS

        candidates: list[str] = []
        candidates.extend(self._PROVIDER_ENV_KEYS.get(provider, ()))
        if provider == "":
            candidates.extend(self._GENERIC_ENV_KEYS)
        elif provider in self._OPENAI_COMPATIBLE_PROVIDERS:
            candidates.extend(self._GENERIC_ENV_KEYS)
        elif provider not in self._PROVIDER_ENV_KEYS:
            candidates.extend(self._GENERIC_ENV_KEYS)
        return tuple(dict.fromkeys(candidates))

    def _allows_dummy_key(self) -> bool:
        provider = self._effective_provider()
        if provider in self._OPENAI_COMPATIBLE_PROVIDERS:
            return True
        model = (self._cfg.model or "").lower()
        if model.startswith("hosted_vllm/"):
            return True
        return self._is_local_api_base()

    def _effective_provider(self) -> str:
        return (
            self._custom_llm_provider
            or self._provider
            or normalize_provider_name(infer_provider_from_model(self._cfg.model, include_openai_family=True))
        )

    def _is_local_api_base(self) -> bool:
        return is_local_api_base(self._cfg.api_base or self._cfg.mock_api_base)

    @staticmethod
    def _first_present(values: Iterable[str | None]) -> str | None:
        return next((value for value in values if value), None)

    @staticmethod
    def _first_env(names: Iterable[str]) -> str | None:
        return next((os.getenv(name) for name in names if os.getenv(name)), None)
