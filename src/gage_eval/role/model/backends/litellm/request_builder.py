"""Build LiteLLM completion request kwargs."""

from __future__ import annotations

import copy
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import urlsplit

from loguru import logger

from gage_eval.role.model.backends.litellm.errors import INVALID_REQUEST
from gage_eval.role.model.backends.litellm.policies import (
    ThinkingControlPolicy,
    ThinkingResolution,
    ToolCallPolicy,
)
from gage_eval.role.model.backends.litellm.provider_detection import (
    looks_like_deepseek,
    normalize_custom_provider,
)
from gage_eval.role.model.config.litellm import LiteLLMRequestConfig

LITELLM_REQUEST_PASSTHROUGH_FIELDS: Tuple[str, ...] = (
    "response_format",
    "max_completion_tokens",
    "stream_options",
    "parallel_tool_calls",
    "logit_bias",
    "user",
    "metadata",
    "fallbacks",
    "context_window_fallback_dict",
    "extra_body",
    "drop_params",
)


class LiteLLMRequestBuilder:
    """Build kwargs for ``litellm.completion`` without invoking LiteLLM."""

    def __init__(
        self,
        *,
        model_name: str,
        provider: str | None = None,
        api_base: str | None = None,
        api_key: str | None = None,
        timeout: float | None = None,
        headers: Dict[str, str] | None = None,
        custom_llm_provider: str | None = None,
        base_sampling: Dict[str, Any] | None = None,
        litellm_request: LiteLLMRequestConfig | None = None,
        drop_params: bool = True,
        is_azure_target: bool = False,
        azure_api_version: str | None = None,
        max_context_length: int | None = None,
        supports_reasoning: bool = False,
        supports_function_calling: bool | None = None,
        thinking_policy: ThinkingControlPolicy | None = None,
        tool_call_policy: ToolCallPolicy | None = None,
    ) -> None:
        self.model_name = model_name
        self.provider = provider
        self.api_base = api_base
        self.api_key = api_key
        self.timeout = timeout
        self.headers = dict(headers or {})
        self.custom_llm_provider = custom_llm_provider
        self.base_sampling = copy.deepcopy(base_sampling or {})
        self.litellm_request = litellm_request or LiteLLMRequestConfig()
        self.drop_params = bool(drop_params)
        self.is_azure_target = is_azure_target
        self.azure_api_version = azure_api_version
        self.max_context_length = max_context_length
        self.supports_reasoning = supports_reasoning
        self.supports_function_calling = supports_function_calling
        self.thinking_policy = thinking_policy or ThinkingControlPolicy()
        self.tool_call_policy = tool_call_policy or ToolCallPolicy()

    def build(
        self,
        inputs: Dict[str, Any],
        *,
        thinking_config: Dict[str, Any] | None = None,
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Return LiteLLM kwargs and normalized sampling parameters."""

        self._validate_api_base()
        raw_sampling_params = inputs.get("sampling_params") or {}
        runtime_sampling_params = inputs.get("runtime_sampling_params")
        if not isinstance(runtime_sampling_params, dict):
            runtime_sampling_params = raw_sampling_params
        request_model = inputs.get("model") or self.model_name
        thinking_resolution = self._resolve_thinking(
            runtime_sampling_params,
            thinking_config or {},
            request_model=request_model,
        )
        sampling_params = self._normalize_sampling_params(
            raw_sampling_params,
            inputs.get("num_samples"),
            thinking_resolution=thinking_resolution,
        )
        stop_sequences = self._prepare_stop_sequences(sampling_params.get("stop"))

        kwargs: Dict[str, Any] = {
            "model": self._normalize_request_model(inputs.get("model") or self.model_name),
            "messages": inputs.get("messages") or [],
            "stream": inputs.get("stream", False),
            "timeout": self.timeout,
        }
        if self.api_base:
            kwargs["base_url"] = self.api_base
            kwargs["api_base"] = self.api_base
        if self.api_key:
            kwargs["api_key"] = self.api_key
        if self.custom_llm_provider:
            kwargs["custom_llm_provider"] = self.custom_llm_provider
        if self.is_azure_target:
            kwargs["api_type"] = "azure"
            if self.azure_api_version:
                kwargs["api_version"] = self.azure_api_version
        if self.headers:
            kwargs["headers"] = dict(self.headers)

        self._add_tools(kwargs, inputs)
        kwargs.update({k: v for k, v in sampling_params.items() if v is not None})
        if stop_sequences:
            kwargs["stop"] = stop_sequences
        kwargs.setdefault("n", inputs.get("num_samples"))

        self._apply_thinking_kwargs(kwargs, sampling_params, thinking_resolution)
        self._apply_runtime_passthrough(kwargs, raw_sampling_params)
        self._apply_litellm_request_passthrough(kwargs)
        return kwargs, sampling_params

    def _validate_api_base(self) -> None:
        if not self.api_base:
            return
        path = urlsplit(self.api_base).path.rstrip("/").lower()
        if path.endswith("/chat/completions") or path.endswith("/completions"):
            raise ValueError(
                "api_base must be an OpenAI-compatible base URL ending at /v1, "
                "not a concrete endpoint path such as /chat/completions. "
                f"(error_type={INVALID_REQUEST})"
            )

    def _add_tools(self, kwargs: Dict[str, Any], inputs: Dict[str, Any]) -> None:
        self.tool_call_policy.apply(
            kwargs,
            inputs,
            supports_function_calling=self.supports_function_calling,
            is_vllm_tool_target=self._is_vllm_tool_target(kwargs.get("model")),
        )

    def _apply_thinking_kwargs(
        self,
        kwargs: Dict[str, Any],
        sampling_params: Dict[str, Any],
        thinking_resolution: ThinkingResolution,
    ) -> None:
        self._merge_kwargs_patch(kwargs, thinking_resolution.kwargs_patch)
        reasoning_effort = sampling_params.get("reasoning_effort") or thinking_resolution.kwargs_patch.get(
            "reasoning_effort"
        )
        if reasoning_effort:
            kwargs["reasoning_effort"] = reasoning_effort

    def _apply_runtime_passthrough(self, kwargs: Dict[str, Any], raw_sampling_params: Dict[str, Any]) -> None:
        for field in LITELLM_REQUEST_PASSTHROUGH_FIELDS:
            if field in raw_sampling_params and raw_sampling_params[field] is not None:
                self._set_passthrough_field(kwargs, field, raw_sampling_params[field])

    def _apply_litellm_request_passthrough(self, kwargs: Dict[str, Any]) -> None:
        for field in LITELLM_REQUEST_PASSTHROUGH_FIELDS:
            if field == "drop_params":
                value = self.litellm_request.drop_params
                kwargs[field] = self.drop_params if value is None else value
                continue
            value = getattr(self.litellm_request, field)
            if value is None:
                continue
            if value == {} or value == []:
                continue
            self._set_passthrough_field(kwargs, field, value)

    def _normalize_sampling_params(
        self,
        params: Dict[str, Any],
        num_samples: Optional[int],
        *,
        thinking_resolution: ThinkingResolution,
    ) -> Dict[str, Any]:
        sampling = {k: copy.deepcopy(v) for k, v in params.items() if v is not None}
        merged_sampling = copy.deepcopy(self.base_sampling)
        merged_sampling.update(sampling)
        normalized: Dict[str, Any] = {}

        max_tokens = (
            merged_sampling.get("max_tokens")
            or merged_sampling.get("max_new_tokens")
            or self.base_sampling.get("max_new_tokens")
        )
        normalized["max_tokens"] = self._prepare_max_tokens(max_tokens, thinking_resolution=thinking_resolution)
        stop_sequences = (
            merged_sampling.get("stop_sequences")
            or merged_sampling.get("stop")
            or self.base_sampling.get("stop")
        )
        if stop_sequences:
            normalized["stop"] = stop_sequences

        for key in (
            "temperature",
            "top_p",
            "presence_penalty",
            "frequency_penalty",
            "repetition_penalty",
            "logprobs",
            "top_k",
            "min_p",
            "seed",
            "reasoning_effort",
        ):
            if merged_sampling.get(key) is not None:
                normalized[key] = merged_sampling[key]

        normalized["n"] = merged_sampling.get("n") or merged_sampling.get("num_samples") or num_samples
        return {k: v for k, v in normalized.items() if v is not None}

    def _prepare_stop_sequences(self, stop: Any) -> List[str]:
        if not stop:
            return []
        if isinstance(stop, str):
            sequences = [stop]
        elif isinstance(stop, list):
            sequences = [s for s in stop if isinstance(s, str)]
        else:
            return []
        if (self.provider or "").lower() == "anthropic":
            sequences = [s for s in sequences if s and s.strip()]
        return sequences

    def _prepare_max_tokens(
        self,
        max_tokens: Any,
        *,
        thinking_resolution: ThinkingResolution,
    ) -> Optional[int]:
        if max_tokens is None:
            return None
        try:
            max_tokens = int(max_tokens)
        except (TypeError, ValueError):
            return None
        if max_tokens <= 0:
            return None
        if self.supports_reasoning and thinking_resolution.mode == "enabled":
            target = max_tokens * 10
            if self.max_context_length:
                target = min(target, self.max_context_length)
            logger.warning("Reasoning 模型 {} 调整 max_tokens 至 {}", self.model_name, target)
            return target
        if self.max_context_length:
            return min(max_tokens, self.max_context_length)
        return max_tokens

    def _resolve_thinking(
        self,
        raw_sampling_params: Dict[str, Any],
        thinking_config: Dict[str, Any],
        *,
        request_model: str,
    ) -> ThinkingResolution:
        merged_config: Dict[str, Any] = {}
        for source in (self.base_sampling, thinking_config, raw_sampling_params):
            self._merge_thinking_source(merged_config, source)

        extra_body = self.litellm_request.extra_body
        if isinstance(extra_body, dict):
            self._merge_thinking_source(
                merged_config,
                {
                    "chat_template_kwargs": extra_body.get("chat_template_kwargs"),
                },
            )
        return self.thinking_policy.resolve(
            model=self._normalize_request_model(request_model),
            provider=self.provider,
            custom_llm_provider=self.custom_llm_provider,
            api_base=self.api_base,
            thinking_config=merged_config,
        )

    def _merge_thinking_source(self, merged_config: Dict[str, Any], source: Dict[str, Any]) -> None:
        for key in ("thinking_mode", "chat_template_kwargs", "reasoning_effort", "enable_thinking", "extra_body"):
            if key not in source or source[key] is None:
                continue
            value = source[key]
            if key == "chat_template_kwargs" and isinstance(value, dict):
                existing = merged_config.get("chat_template_kwargs")
                merged_config["chat_template_kwargs"] = self._deep_merge_dicts(
                    existing if isinstance(existing, dict) else {},
                    value,
                )
                enable_thinking = value.get("enable_thinking")
                if enable_thinking is True or enable_thinking is False:
                    merged_config["enable_thinking"] = enable_thinking
                    merged_config["thinking_mode"] = "enabled" if enable_thinking else "disabled"
                continue
            if key == "extra_body" and isinstance(value, dict):
                chat_template_kwargs = value.get("chat_template_kwargs")
                if isinstance(chat_template_kwargs, dict):
                    self._merge_thinking_source(
                        merged_config,
                        {"chat_template_kwargs": chat_template_kwargs},
                    )
                continue
            merged_config[key] = copy.deepcopy(value)
            if key == "enable_thinking" and (value is True or value is False):
                merged_config["thinking_mode"] = "enabled" if value else "disabled"

    def _set_passthrough_field(self, kwargs: Dict[str, Any], field: str, value: Any) -> None:
        if field == "extra_body" and isinstance(value, dict):
            existing = kwargs.get("extra_body")
            kwargs["extra_body"] = self._deep_merge_dicts(existing if isinstance(existing, dict) else {}, value)
            return
        kwargs[field] = copy.deepcopy(value)

    def _merge_kwargs_patch(self, kwargs: Dict[str, Any], patch: Dict[str, Any]) -> None:
        for key, value in patch.items():
            self._set_passthrough_field(kwargs, key, value)

    @classmethod
    def _deep_merge_dicts(cls, base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
        merged = copy.deepcopy(base)
        for key, value in override.items():
            existing = merged.get(key)
            if isinstance(existing, dict) and isinstance(value, dict):
                merged[key] = cls._deep_merge_dicts(existing, value)
            else:
                merged[key] = copy.deepcopy(value)
        return merged

    def _normalize_request_model(self, model_name: str) -> str:
        if looks_like_deepseek(self.provider, model_name, self.api_base) and "/" not in model_name:
            return f"deepseek/{model_name}"
        return model_name

    def _is_vllm_tool_target(self, request_model: str | None) -> bool:
        model_lower = (request_model or "").lower()
        providers = {
            (self.provider or "").lower().replace("-", "_"),
            (self.custom_llm_provider or "").lower().replace("-", "_"),
        }
        if model_lower.startswith("hosted_vllm/"):
            return True
        if providers & {"hosted_vllm", "vllm", "openai_compatible"}:
            return True
        if self.tool_call_policy.has_declared_vllm_tool_support():
            return True
        return False

    normalize_custom_provider = staticmethod(normalize_custom_provider)
