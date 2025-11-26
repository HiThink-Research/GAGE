"""HTTP backend that proxies requests via HuggingFace Inference Providers."""

from __future__ import annotations

import asyncio
import os
from typing import Any, Dict, List

from huggingface_hub import AsyncInferenceClient
from huggingface_hub.errors import HfHubHTTPError
from loguru import logger

from gage_eval.role.model.backends.base_backend import EngineBackend
from gage_eval.role.model.config.generations import GenerationParameters
from gage_eval.registry import registry


@registry.asset(
    "backends",
    "multi_provider_http",
    desc="多 provider HTTP 推理后端（基于 HuggingFace Inference Providers）",
    tags=("llm", "remote", "provider"),
    modalities=("text",),
)
class MultiProviderHTTPBackend(EngineBackend):
    """Backend that fans out chat-completion requests to HuggingFace Inference Providers."""

    def __init__(self, config: Dict[str, Any]) -> None:
        self.transport = "http"
        self.http_retry_params = config.get("http_retry_params", {})
        self._semaphore: asyncio.Semaphore | None = None
        super().__init__(config)

    def load_model(self, config: Dict[str, Any]):
        provider = config.get("provider")
        if not provider:
            raise ValueError("MultiProviderHTTPBackend requires 'provider'")
        model_name = config.get("model_name")
        if not model_name:
            raise ValueError("MultiProviderHTTPBackend requires 'model_name'")
        token = config.get("token") or os.getenv("HF_API_TOKEN")
        if not token:
            logger.warning("HF_API_TOKEN not set; AsyncInferenceClient will rely on default login context")

        self.provider = provider
        self.model_name = model_name
        self.parallel_calls = max(1, int(config.get("parallel_calls_count", 4)))
        self.temperature = config.get("generation_parameters", {}).get("temperature", 0.7)
        self._client = AsyncInferenceClient(
            provider=provider,
            token=token,
            timeout=config.get("timeout"),
            proxies=config.get("proxies"),
            bill_to=config.get("org_to_bill"),
        )
        # optional chat template
        from transformers import AutoTokenizer

        tokenizer_name = config.get("tokenizer_name") or model_name
        try:
            self._tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        except HfHubHTTPError:
            self._tokenizer = None
            logger.warning("Tokenizer %s not available; prompts will be sent as raw text", tokenizer_name)

        self._generation = GenerationParameters(**config.get("generation_parameters", {}))

    async def ainvoke(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        semaphore = self._lazy_semaphore()
        async with semaphore:
            prompt = self._build_messages(payload)
            kwargs = {
                "model": self.model_name,
                "messages": prompt,
                "n": payload.get("num_samples", 1),
            }
            kwargs.update(self._generation.to_inference_providers_dict())
            response = await self._client.chat_completion(**kwargs)
            choices = response.get("choices") or []
            answer = choices[0]["message"]["content"] if choices else ""
            return {
                "answer": answer,
                "raw_response": response,
            }

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
