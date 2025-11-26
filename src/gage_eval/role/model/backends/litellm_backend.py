"""LiteLLM backend adaptor."""

from __future__ import annotations

import os
from typing import Any, Dict

from gage_eval.role.model.backends.base_backend import EngineBackend
from gage_eval.registry import registry


@registry.asset(
    "backends",
    "litellm",
    desc="通过 LiteLLM 统一接入各类推理服务",
    tags=("llm", "remote", "api"),
    modalities=("text",),
)
class LiteLLMBackend(EngineBackend):
    def load_model(self, config: Dict[str, Any]):
        try:  # pragma: no cover - optional dependency
            import litellm  # type: ignore
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError("liteLLM is not installed") from exc
        api_key = config.get("api_key") or os.getenv("LITELLM_API_KEY") or os.getenv("OPENAI_API_KEY")
        if api_key:
            litellm.api_key = api_key
        api_base = config.get("api_base")
        if api_base:
            litellm.api_base = api_base
        headers = config.get("extra_headers")
        if headers:
            litellm.headers = headers
        self._litellm = litellm
        return None

    def generate(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        messages = inputs.get("messages")
        if not messages:
            prompt = inputs.get("prompt") or ""
            messages = [{"role": "user", "content": prompt}]
        kwargs = dict(model=self.config.get("model", "gpt-4o-mini"), messages=messages)
        sampling = inputs.get("sampling_params") or {}
        kwargs.update({k: v for k, v in sampling.items() if v is not None})

        # liteLLM exposes both sync and async; we keep sync for EngineBackend.
        completion = self._litellm.completion(**kwargs)
        choice = (completion.get("choices") or [{}])[0]
        message = choice.get("message") or {}
        return {
            "answer": message.get("content"),
            "raw": completion,
        }
