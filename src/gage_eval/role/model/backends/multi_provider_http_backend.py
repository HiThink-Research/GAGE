"""HTTP backend that proxies requests via HuggingFace Inference Providers."""

from __future__ import annotations

import asyncio
import os
from typing import Any, Dict, List

from huggingface_hub import AsyncInferenceClient
from huggingface_hub.errors import HfHubHTTPError
from loguru import logger

from gage_eval.role.model.backends.base_backend import EngineBackend
from gage_eval.role.model.config.inference_providers import InferenceProvidersBackendConfig
from gage_eval.registry import registry


@registry.asset(
    "backends",
    "multi_provider_http",
    desc="Multi-provider HTTP inference backend (HuggingFace Inference Providers)",
    tags=("llm", "remote", "provider"),
    modalities=("text",),
    config_schema_ref="gage_eval.role.model.config.inference_providers:InferenceProvidersBackendConfig",
)
class MultiProviderHTTPBackend(EngineBackend):
    """Backend that fans out chat-completion requests to HuggingFace Inference Providers."""

    def __init__(self, config: Dict[str, Any]) -> None:
        self.transport = "http"
        self.http_retry_params: Dict[str, Any] = {}
        self._semaphore: asyncio.Semaphore | None = None
        super().__init__(config)

    def load_model(self, config: Dict[str, Any]):
        cfg = InferenceProvidersBackendConfig(**config)
        token = cfg.token or os.getenv("HF_API_TOKEN")
        if not token:
            logger.warning("HF_API_TOKEN not set; AsyncInferenceClient will rely on default login context")

        self.provider = cfg.provider
        self.model_name = cfg.model_name
        self.parallel_calls = max(1, int(cfg.parallel_calls_count))
        self._client = AsyncInferenceClient(
            provider=cfg.provider,
            token=token,
            timeout=cfg.timeout,
            proxies=cfg.proxies,
            bill_to=cfg.org_to_bill,
        )
        # optional chat template
        from transformers import AutoTokenizer

        tokenizer_name = cfg.tokenizer_name or self.model_name
        try:
            self._tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        except (HfHubHTTPError, ValueError, OSError) as exc:
            self._tokenizer = None
            logger.warning(
                "Tokenizer {} not available or requires trust_remote_code ({}); prompts will be sent as raw text",
                tokenizer_name,
                exc,
            )

        self._generation = cfg.generation_parameters
        self.http_retry_params = self._prepare_http_retry_params(cfg.http_retry_params)

    async def ainvoke(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        semaphore = self._lazy_semaphore()
        async with semaphore:
            prompt = self._build_messages(payload)
            response = await self._call_api(prompt, payload.get("num_samples", 1))
            choices = self._extract_choices(response)
            answer = choices[0] if choices else ""
            return {
                "answer": answer,
                "raw_response": response,
            }

    async def _call_api(self, messages: List[Dict[str, str]], num_samples: int):
        retries = int(self.http_retry_params.get("max_retries", 5))
        base_sleep = float(self.http_retry_params.get("base_sleep", 3))
        multiplier = float(self.http_retry_params.get("multiplier", 2))

        kwargs = {
            "model": self.model_name,
            "messages": messages,
            "n": num_samples,
        }
        kwargs.update(self._generation.to_inference_providers_dict())

        if kwargs.get("temperature") == 0.0 and num_samples > 1:
            raise ValueError(
                "Temperature is set to 0.0, but num_samples > 1. "
                "This is not supported by the inference providers API."
            )

        for attempt in range(retries + 1):
            try:
                return await self._chat_completion(kwargs)
            except Exception as exc:  # pragma: no cover - network/runtime failures
                if attempt == retries:
                    raise
                wait = min(64, base_sleep * (multiplier**attempt))
                logger.warning("Inference provider call failed (attempt {}): {}. Retrying in {}s", attempt + 1, exc, wait)
                await asyncio.sleep(wait)

    async def _chat_completion(self, kwargs: Dict[str, Any]):
        # huggingface_hub >=0.24 exposes chat.completions.create
        chat = getattr(self._client, "chat", None)
        completions = getattr(chat, "completions", None) if chat else None
        create_fn = getattr(completions, "create", None) if completions else None
        if create_fn:
            return await create_fn(**kwargs)
        # fallback to legacy name
        if hasattr(self._client, "chat_completion"):
            return await self._client.chat_completion(**kwargs)
        raise AttributeError("AsyncInferenceClient has no chat.completions.create or chat_completion method")

    def _extract_choices(self, response: Any) -> List[str]:
        # Support ChatCompletionOutput or raw dict
        if hasattr(response, "choices"):
            choices = response.choices
            if choices and hasattr(choices[0], "message"):
                return [c.message.content for c in choices]
        if isinstance(response, dict):
            choices = response.get("choices") or []
            return [c.get("message", {}).get("content", "") for c in choices]
        return []

    def _build_messages(self, payload: Dict[str, Any]) -> List[Dict[str, str]]:
        sample = payload.get("sample", {})
        messages = payload.get("messages") or []
        if messages:
            return messages
        prompt = sample.get("prompt") or payload.get("prompt") or ""
        return [{"role": "user", "content": prompt}]

    def _lazy_semaphore(self) -> asyncio.Semaphore:
        if self._semaphore is None:
            self._semaphore = asyncio.Semaphore(self.parallel_calls)
        return self._semaphore

    def _prepare_http_retry_params(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Merge user-provided retry settings with safe defaults required by wrappers."""

        defaults = {
            "attempts": 3,
            "interval": 1.0,
            "max_retries": 5,
            "base_sleep": 3,
            "multiplier": 2,
        }
        merged = {**defaults, **(params or {})}
        merged["attempts"] = int(merged.get("attempts", defaults["attempts"]))
        merged["max_retries"] = int(merged.get("max_retries", defaults["max_retries"]))
        merged["interval"] = float(merged.get("interval", defaults["interval"]))
        merged["base_sleep"] = float(merged.get("base_sleep", defaults["base_sleep"]))
        merged["multiplier"] = float(merged.get("multiplier", defaults["multiplier"]))
        return merged
