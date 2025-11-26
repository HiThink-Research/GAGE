"""Prompt-aware helper mixin for text generation backends."""

from __future__ import annotations

from typing import Any, Dict, Optional


class TextGenerationMixin:
    """Provides `prepare_backend_request` logic shared by text LLM adapters."""

    def prepare_backend_request(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        sample = payload.get("sample", {})
        rendered = self.render_prompt(payload)
        sample_messages = sample.get("messages") or payload.get("messages") or []
        messages = rendered.messages or sample_messages
        prompt = rendered.prompt or sample.get("prompt") or sample.get("text") or ""
        if not prompt and messages:
            # Leave prompt empty but preserve messages so backend can render chat prompts.
            prompt = ""
        sampling_params = self._compose_sampling_params(
            sample_params=sample.get("sampling_params"),
            runtime_params=payload.get("sampling_params"),
            legacy_params=sample.get("generation_params"),
        )
        request = {
            "sample": sample,
            "prompt": prompt,
            "messages": messages,
            "inputs": sample.get("inputs"),
            "sampling_params": sampling_params,
        }
        if rendered.metadata:
            request["prompt_meta"] = rendered.metadata
        cache_namespace = self._runtime_namespace(payload)
        if cache_namespace:
            request["cache_namespace"] = cache_namespace
        return request

    def _runtime_namespace(self, payload: Dict[str, Any]) -> Optional[str]:
        usage = payload.get("usage")
        if usage:
            return str(usage)
        return None
