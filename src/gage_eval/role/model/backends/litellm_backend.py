"""LiteLLM backend adaptor."""

from __future__ import annotations

import json
import os
import time
from typing import Any, Dict, List, Optional, Tuple

from loguru import logger

from gage_eval.role.model.backends.base_backend import EngineBackend
from gage_eval.role.model.config.litellm import LiteLLMBackendConfig
from gage_eval.registry import registry
from gage_eval.utils.messages import normalize_messages_for_template, stringify_message_content


@registry.asset(
    "backends",
    "litellm",
    desc="LiteLLM backend for unified provider access (Grok/Kimi base URLs + param normalization)",
    tags=("llm", "remote", "api"),
    modalities=("text",),
)
class LiteLLMBackend(EngineBackend):
    """LiteLLM backend with provider inference and sampling normalization."""

    def __init__(self, config: Dict[str, Any]) -> None:
        self.http_retry_mode = "native"
        self.transport = "http"
        self._litellm = None
        self._supports_reasoning_fn = None
        self._custom_llm_provider = None
        super().__init__(config)

    # ------------------------------------------------------------------ #
    # Engine interface                                                   #
    # ------------------------------------------------------------------ #
    def load_model(self, config_dict: Dict[str, Any]):
        # STEP 1: Resolve provider-specific routing, credentials, and sampling defaults.
        self._cfg = LiteLLMBackendConfig(**config_dict)
        self._tool_choice_default = config_dict.get("tool_choice")
        self.model_name = self._cfg.model
        self.provider = self._cfg.provider or self._infer_provider(self.model_name)
        self.api_base = self._cfg.api_base
        self._is_deepseek_target = self._looks_like_deepseek(self.provider, self.model_name, self.api_base)
        self.api_key = None
        self.headers = dict(self._cfg.extra_headers or {})
        self._timeout = self._cfg.timeout
        self._max_retries = max(1, int(self._cfg.max_retries))
        self._retry_sleep = float(self._cfg.retry_sleep)
        self._retry_multiplier = max(1.0, float(self._cfg.retry_multiplier))
        self._max_context_length = self._cfg.max_model_length
        self._base_sampling = self._cfg.generation_parameters.to_dict()
        self._is_kimi_target = self._looks_like_kimi(self.provider, self.model_name, self.api_base)
        self._is_grok_target = self._looks_like_grok(self.provider, self.model_name, self.api_base)
        self._is_azure_target = self._looks_like_azure(self.provider, self.model_name, self.api_base)
        if self._is_deepseek_target and (not self.provider or self.provider.lower() == "openai"):
            self.provider = "deepseek"
        if not self.provider and self._is_grok_target:
            self.provider = "grok"
        if not self.provider and self._is_kimi_target:
            self.provider = "kimi"
        if not self.provider and self._is_azure_target:
            self.provider = "azure"
        if not self.api_base:
            if self._is_deepseek_target:
                self.api_base = "https://api.deepseek.com"
            elif self._is_kimi_target:
                self.api_base = "https://api.moonshot.cn/v1"
            elif self._is_grok_target:
                self.api_base = "https://api.x.ai/v1"
            elif self._is_azure_target:
                self.api_base = os.getenv("AZURE_OPENAI_ENDPOINT") or os.getenv("AZURE_ENDPOINT")
                if self.api_base:
                    self.api_base = self.api_base.rstrip("/")

        self.api_key = self._resolve_api_key()
        self._azure_api_version = self._cfg.azure_api_version or os.getenv("AZURE_OPENAI_API_VERSION")
        if self._is_azure_target and not self._azure_api_version:
            self._azure_api_version = "2024-02-15-preview"

        if self._cfg.force_kimi_direct or self._cfg.prefer_litellm_kimi:
            logger.warning("force_kimi_direct/prefer_litellm_kimi 已废弃，LiteLLM 现统一走 litellm 调用路径")

        # NOTE: Normalize `custom_llm_provider`: prefer an explicit user value, then
        # fall back to provider inference.
        self._custom_llm_provider = self._normalize_custom_provider(
            getattr(self._cfg, "custom_llm_provider", None) or self.provider
        )

        try:  # pragma: no cover - optional dependency
            import litellm  # type: ignore
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError("liteLLM is not installed") from exc

        # STEP 2: Initialize LiteLLM module behavior without mutating credential globals.
        self._litellm = litellm
        self._supports_reasoning_fn = getattr(litellm, "supports_reasoning", None)
        litellm.drop_params = True
        litellm.verbose = bool(self._cfg.verbose)
        return None

    def prepare_inputs(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        sample = payload.get("sample") or {}
        messages = payload.get("messages") or sample.get("messages")
        if not messages:
            system_prompt = payload.get("system_prompt") or sample.get("system_prompt")
            prompt = payload.get("prompt") or sample.get("prompt") or sample.get("text") or ""
            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": prompt})

        messages = self._normalize_messages_for_provider(messages)

        sampling_params = dict(self._base_sampling)
        sampling_params.update(sample.get("sampling_params") or {})
        sampling_params.update(payload.get("sampling_params") or {})
        tool_defs = payload.get("tools") or sample.get("tools")
        tool_choice = payload.get("tool_choice") or sample.get("tool_choice") or self._tool_choice_default

        return {
            "model": payload.get("model") or self.model_name,
            "messages": messages,
            "sampling_params": sampling_params,
            "tools": tool_defs,
            "tool_choice": tool_choice,
            "stream": bool(payload.get("stream", self._cfg.streaming)),
            "num_samples": payload.get("num_samples") or sampling_params.get("n"),
        }

    def generate(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        return self._generate_litellm(inputs)

    # ------------------------------------------------------------------ #
    # LiteLLM call path                                                 #
    # ------------------------------------------------------------------ #
    def _generate_litellm(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        # STEP 1: Build isolated request kwargs and execute the LiteLLM call path.
        kwargs, _ = self._build_litellm_kwargs(inputs)
        stream = kwargs.pop("stream", False)
        start = time.time()

        def _call():
            return self._litellm.completion(stream=stream, **kwargs)

        completion = self._call_with_retries(_call)

        # STEP 2: Normalize the response for downstream consumers and safe diagnostics.
        if stream:
            answer, raw_response = self._collect_stream(completion)
        else:
            answer = self._extract_answer(completion)
            raw_response = completion
        raw_response = self._to_jsonable(raw_response)
        result: Dict[str, Any] = {"answer": answer, "raw_response": raw_response}
        if not stream and hasattr(completion, "usage"):
            usage = getattr(completion, "usage")
            if hasattr(usage, "model_dump"):
                result["usage"] = usage.model_dump()
        result.setdefault("latency_ms", (time.time() - start) * 1000)
        logger.info(
            "LiteLLM response summary: {}",
            self._format_response_json(
                self._build_response_log_summary(
                    raw_response,
                    answer=answer,
                    usage=result.get("usage"),
                    stream=stream,
                    request_model=kwargs.get("model"),
                    latency_ms=result.get("latency_ms"),
                )
            ),
        )
        return result

    def _build_litellm_kwargs(self, inputs: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        sampling_params = self._normalize_sampling_params(inputs.get("sampling_params") or {}, inputs.get("num_samples"))
        stop_sequences = self._prepare_stop_sequences(sampling_params.get("stop"))

        kwargs: Dict[str, Any] = {
            "model": self._normalize_request_model(inputs.get("model") or self.model_name),
            "messages": inputs.get("messages") or [],
            "response_format": {"type": "text"},
            "stream": inputs.get("stream", False),
            "timeout": self._timeout,
        }
        if self.api_base:
            kwargs["base_url"] = self.api_base
            kwargs["api_base"] = self.api_base
        if self.api_key:
            kwargs["api_key"] = self.api_key
        if self._custom_llm_provider:
            kwargs["custom_llm_provider"] = self._custom_llm_provider
        if self._is_azure_target:
            kwargs["api_type"] = "azure"
            if self._azure_api_version:
                kwargs["api_version"] = self._azure_api_version
        if self.headers:
            kwargs["headers"] = dict(self.headers)
        tool_defs = inputs.get("tools")
        if tool_defs:
            formatted_tools = self._format_tools(tool_defs)
            if formatted_tools:
                kwargs["tools"] = formatted_tools
                tool_choice = inputs.get("tool_choice")
                if tool_choice is not None:
                    kwargs["tool_choice"] = tool_choice
        kwargs.update({k: v for k, v in sampling_params.items() if v is not None})
        if stop_sequences:
            kwargs["stop"] = stop_sequences
        kwargs.setdefault("n", inputs.get("num_samples"))

        # STEP: Inject thinking-related parameters from base class config
        thinking_config = self.get_thinking_config()
        if "enable_thinking" in thinking_config:
            kwargs["enable_thinking"] = thinking_config["enable_thinking"]
        # Support reasoning_effort from config or per-request sampling params
        reasoning_effort = (
            sampling_params.get("reasoning_effort")
            or thinking_config.get("reasoning_effort")
        )
        if reasoning_effort:
            kwargs["reasoning_effort"] = reasoning_effort

        return kwargs, sampling_params

    def _extract_answer(self, completion: Any) -> str:
        if completion is None:
            return ""
        choices = getattr(completion, "choices", None)
        if choices:
            choice = choices[0]
            message = getattr(choice, "message", None) or getattr(choice, "delta", None)
            content = getattr(message, "content", None) if message else None
            if isinstance(content, list):
                return "".join([self._content_piece(part) for part in content])
            return content or ""
        if isinstance(completion, dict):
            choices = completion.get("choices") or []
            if choices:
                message = choices[0].get("message") or choices[0].get("delta") or {}
                content = message.get("content")
                if isinstance(content, list):
                    return "".join([self._content_piece(part) for part in content])
                return content or ""
        return str(completion)

    def _collect_stream(self, stream_response: Any) -> Tuple[str, List[Any]]:
        """Collect streaming chunks into a single text answer."""

        chunks: List[Any] = []
        parts: List[str] = []
        for chunk in stream_response:
            chunks.append(chunk)
            choices = getattr(chunk, "choices", None)
            if not choices and isinstance(chunk, dict):
                choices = chunk.get("choices")
            if not choices:
                continue
            delta = getattr(choices[0], "delta", None) or getattr(choices[0], "message", None)
            if delta is None and isinstance(choices[0], dict):
                delta = choices[0].get("delta") or choices[0].get("message")
            content = getattr(delta, "content", None) if delta else None
            if isinstance(content, list):
                parts.append("".join([self._content_piece(part) for part in content]))
            elif content:
                parts.append(str(content))
        return "".join(parts), chunks

    # ------------------------------------------------------------------ #
    # Helpers                                                            #
    # ------------------------------------------------------------------ #
    def _normalize_messages_for_provider(self, messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Normalize message content according to the target provider's input contract."""

        if self._should_flatten_multimodal_messages():
            return normalize_messages_for_template(messages, image_placeholder="<image>")
        if self._should_sanitize_openai_messages():
            return self._sanitize_openai_messages(messages)
        return messages

    def _should_flatten_multimodal_messages(self) -> bool:
        provider = (self._custom_llm_provider or self.provider or "").lower()
        return provider == "deepseek" or self._looks_like_deepseek(self.provider, self.model_name, self.api_base)

    def _should_sanitize_openai_messages(self) -> bool:
        provider = (self._custom_llm_provider or self.provider or "").lower()
        return provider in {"openai", "azure"}

    def _sanitize_openai_messages(self, messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        sanitized: List[Dict[str, Any]] = []
        for msg in messages or []:
            new_msg = dict(msg)
            content = msg.get("content")
            if isinstance(content, list):
                new_content: List[Dict[str, Any]] = []
                for item in content:
                    if isinstance(item, dict):
                        itype = item.get("type")
                        if itype == "text":
                            text = item.get("text")
                            if text is not None:
                                new_content.append({"type": "text", "text": str(text)})
                        elif itype == "image_url":
                            url = None
                            val = item.get("image_url")
                            if isinstance(val, dict):
                                url = val.get("url")
                            elif isinstance(val, str):
                                url = val
                            if not url:
                                url = item.get("url") or item.get("image")
                            if url:
                                new_content.append({"type": "image_url", "image_url": {"url": url}})
                        else:
                            text = item.get("text")
                            if text is not None:
                                new_content.append({"type": "text", "text": str(text)})
                    elif item is not None:
                        new_content.append({"type": "text", "text": str(item)})
                if not new_content:
                    fallback_text = stringify_message_content(content, image_placeholder=None)
                    new_msg["content"] = fallback_text
                else:
                    new_msg["content"] = new_content
            sanitized.append(new_msg)
        return sanitized

    def _format_tools(self, tools: Any) -> List[Dict[str, Any]]:
        if not tools:
            return []
        if isinstance(tools, dict):
            tools = [tools]
        if not isinstance(tools, list):
            return []
        formatted = []
        for tool in tools:
            if not isinstance(tool, dict):
                continue
            if tool.get("type") == "function":
                formatted.append(
                    {
                        "type": "function",
                        "function": {
                            "name": tool["function"]["name"],
                            "description": tool["function"].get("description", ""),
                            "parameters": tool["function"].get("parameters", {"type": "object", "properties": {}}),
                        },
                    }
                )
            elif "name" in tool and "parameters" in tool:
                formatted.append(
                    {
                        "type": "function",
                        "function": {
                            "name": tool.get("name"),
                            "description": tool.get("description", ""),
                            "parameters": tool.get("parameters") or {"type": "object", "properties": {}},
                        },
                    }
                )
            else:
                formatted.append(tool)
        return formatted

    def _normalize_sampling_params(self, params: Dict[str, Any], num_samples: Optional[int]) -> Dict[str, Any]:
        sampling = {k: v for k, v in params.items() if v is not None}
        normalized: Dict[str, Any] = {}

        max_tokens = sampling.get("max_tokens") or sampling.get("max_new_tokens") or self._base_sampling.get("max_new_tokens")
        normalized["max_tokens"] = self._prepare_max_tokens(max_tokens)
        stop_sequences = sampling.get("stop_sequences") or sampling.get("stop") or self._base_sampling.get("stop")
        if stop_sequences:
            normalized["stop"] = stop_sequences

        for key in ("temperature", "top_p", "presence_penalty", "frequency_penalty", "repetition_penalty", "logprobs", "top_k", "min_p", "seed"):
            if sampling.get(key) is not None:
                normalized[key] = sampling[key]
            elif key in self._base_sampling and self._base_sampling[key] is not None:
                normalized.setdefault(key, self._base_sampling[key])

        normalized["n"] = sampling.get("n") or sampling.get("num_samples") or num_samples
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

    def _prepare_max_tokens(self, max_tokens: Any) -> Optional[int]:
        if max_tokens is None:
            return None
        try:
            max_tokens = int(max_tokens)
        except (TypeError, ValueError):
            return None
        if max_tokens <= 0:
            return None
        if self._supports_reasoning_fn and self._supports_reasoning_fn(self.model_name):
            target = max_tokens * 10
            if self._max_context_length:
                target = min(target, self._max_context_length)
            logger.warning("Reasoning 模型 {} 调整 max_tokens 至 {}", self.model_name, target)
            return target
        if self._max_context_length:
            return min(max_tokens, self._max_context_length)
        return max_tokens

    def _call_with_retries(self, func):
        last_exc: Exception | None = None
        for attempt in range(self._max_retries):
            try:
                return func()
            except Exception as exc:  # pragma: no cover - network/third-party errors
                last_exc = exc
                if self._is_non_retryable_error(exc):
                    logger.debug("LiteLLM call aborted without retry due to non-retryable error: {}", exc)
                    break
                if attempt == self._max_retries - 1:
                    break
                wait = min(64, self._retry_sleep * (self._retry_multiplier**attempt))
                logger.warning("LiteLLM 调用失败，重试 {}/{}，等待 {:.1f}s: {}", attempt + 1, self._max_retries, wait, exc)
                time.sleep(wait)
        assert last_exc is not None
        raise last_exc

    @staticmethod
    def _is_non_retryable_error(exc: Exception) -> bool:
        """Return True when an error should bypass retry loops."""

        message = str(exc).strip().lower()
        if not message:
            return False
        non_retryable_markers = (
            "cannot schedule new futures after shutdown",
            "rolepool",
            "is shut down",
        )
        return any(marker in message for marker in non_retryable_markers)

    @staticmethod
    def _infer_provider(model_name: str | None) -> Optional[str]:
        if not model_name:
            return None
        if model_name.startswith("deepseek/"):
            return "deepseek"
        if "/" in model_name:
            return model_name.split("/")[0]
        lower = model_name.lower()
        if lower.startswith("deepseek"):
            return "deepseek"
        if lower.startswith("grok"):
            return "grok"
        if lower.startswith("moonshot"):
            return "kimi"
        if lower.startswith("azure:"):
            return "azure"
        return None

    @staticmethod
    def _looks_like_deepseek(provider: Optional[str], model: str, api_base: Optional[str] = None) -> bool:
        target = (provider or "").lower()
        model_lower = (model or "").lower()
        base = (api_base or "").lower()
        return target == "deepseek" or model_lower.startswith("deepseek") or "deepseek.com" in base

    @staticmethod
    def _looks_like_kimi(provider: Optional[str], model: str, api_base: Optional[str] = None) -> bool:
        target = (provider or "").lower()
        model_lower = (model or "").lower()
        base = (api_base or "").lower()
        return target in {"kimi", "moonshot"} or model_lower.startswith("moonshot") or "moonshot" in base or "kimi" in base

    @staticmethod
    def _looks_like_grok(provider: Optional[str], model: str, api_base: Optional[str] = None) -> bool:
        target = (provider or "").lower()
        model_lower = (model or "").lower()
        base = (api_base or "").lower()
        return target in {"grok", "xai"} or model_lower.startswith("grok") or "api.x.ai" in base or base.endswith("x.ai")

    @staticmethod
    def _looks_like_azure(provider: Optional[str], model: str, api_base: Optional[str] = None) -> bool:
        target = (provider or "").lower()
        model_lower = (model or "").lower()
        base = (api_base or "").lower()
        return target in {"azure", "azure_openai"} or "openai.azure.com" in base or "azure" in base or model_lower.startswith("azure:")

    @staticmethod
    def _normalize_custom_provider(provider: Optional[str]) -> Optional[str]:
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

    @staticmethod
    def _content_piece(part: Any) -> str:
        if isinstance(part, dict) and "text" in part:
            return str(part["text"])
        return str(part)

    def _normalize_request_model(self, model_name: str) -> str:
        """Return the provider-qualified model name when LiteLLM expects one."""

        if self._looks_like_deepseek(self.provider, model_name, self.api_base) and "/" not in model_name:
            return f"deepseek/{model_name}"
        return model_name

    def _resolve_api_key(self) -> Optional[str]:
        """Resolve API credentials without leaking DeepSeek keys to other providers."""

        candidates: List[Optional[str]] = [self._cfg.api_key]
        if self._is_deepseek_target:
            candidates.append(os.getenv("DEEPSEEK_API_KEY"))
        candidates.extend(
            [
                os.getenv("LITELLM_API_KEY"),
                os.getenv("OPENAI_API_KEY"),
                os.getenv("XAI_API_KEY"),
                os.getenv("GROK_API_KEY"),
                os.getenv("AZURE_API_KEY"),
                os.getenv("AZURE_OPENAI_API_KEY"),
                os.getenv("KIMI_API_KEY"),
                os.getenv("MOONSHOT_API_KEY"),
            ]
        )
        return next((candidate for candidate in candidates if candidate), None)

    @staticmethod
    def _to_jsonable(obj: Any) -> Any:
        """Best-effort convert litellm ModelResponse / stream chunks to JSON-safe objects."""
        try:
            if obj is None:
                return None
            if isinstance(obj, (str, int, float, bool)):
                return obj
            if isinstance(obj, dict):
                return {k: LiteLLMBackend._to_jsonable(v) for k, v in obj.items()}
            if isinstance(obj, list):
                return [LiteLLMBackend._to_jsonable(v) for v in obj]
            if hasattr(obj, "model_dump"):
                return obj.model_dump()
            if hasattr(obj, "__dict__"):
                return {k: LiteLLMBackend._to_jsonable(v) for k, v in obj.__dict__.items()}
            return str(obj)
        except Exception:  # pragma: no cover - defensive
            return str(obj)

    @staticmethod
    def _build_response_log_summary(
        raw_response: Any,
        *,
        answer: str,
        usage: Optional[Dict[str, Any]],
        stream: bool,
        request_model: Optional[str],
        latency_ms: Optional[float],
    ) -> Dict[str, Any]:
        """Builds a non-sensitive response summary for logs.

        Args:
            raw_response: JSON-safe LiteLLM response payload.
            answer: Extracted answer text returned to callers.
            usage: Usage payload captured from the completion object.
            stream: Whether the request used streaming mode.
            request_model: Model name sent to LiteLLM for this request.
            latency_ms: End-to-end request latency in milliseconds.

        Returns:
            A compact summary that excludes response content and tool payloads.
        """

        summary: Dict[str, Any] = {
            "stream": stream,
            "request_model": request_model,
            "latency_ms": round(float(latency_ms), 3) if latency_ms is not None else None,
            "answer_chars": len(answer or ""),
            "response_type": type(raw_response).__name__,
        }

        if isinstance(raw_response, dict):
            choices = raw_response.get("choices") or []
            first_choice = choices[0] if choices else {}
            summary.update(
                {
                    "response_model": raw_response.get("model"),
                    "response_object": raw_response.get("object"),
                    "choice_count": len(choices),
                    "finish_reason": LiteLLMBackend._extract_finish_reason(first_choice),
                    "has_tool_calls": LiteLLMBackend._choice_has_tool_calls(first_choice),
                }
            )
        elif isinstance(raw_response, list):
            first_chunk = next((chunk for chunk in raw_response if isinstance(chunk, dict)), {})
            finish_choice = LiteLLMBackend._find_stream_finish_choice(raw_response)
            summary.update(
                {
                    "response_object": first_chunk.get("object"),
                    "response_model": first_chunk.get("model") or request_model,
                    "chunk_count": len(raw_response),
                    "finish_reason": LiteLLMBackend._extract_finish_reason(finish_choice),
                    "has_tool_calls": LiteLLMBackend._choice_has_tool_calls(finish_choice),
                }
            )

        resolved_usage = usage
        if resolved_usage is None and isinstance(raw_response, dict):
            resolved_usage = raw_response.get("usage")
        if resolved_usage:
            summary["usage"] = LiteLLMBackend._to_jsonable(resolved_usage)

        return {key: value for key, value in summary.items() if value is not None}

    @staticmethod
    def _find_stream_finish_choice(raw_response: List[Any]) -> Dict[str, Any]:
        """Returns the most informative stream choice for logging metadata."""

        for chunk in reversed(raw_response):
            if not isinstance(chunk, dict):
                continue
            choices = chunk.get("choices") or []
            if choices:
                return choices[0]
        return {}

    @staticmethod
    def _extract_finish_reason(choice: Any) -> Optional[str]:
        """Extracts finish_reason from a choice payload without touching content."""

        if isinstance(choice, dict):
            finish_reason = choice.get("finish_reason")
            if finish_reason is not None:
                return str(finish_reason)
        return None

    @staticmethod
    def _choice_has_tool_calls(choice: Any) -> bool:
        """Returns whether the choice contains tool-call metadata."""

        if not isinstance(choice, dict):
            return False
        message = choice.get("message") or choice.get("delta") or {}
        if not isinstance(message, dict):
            return False
        tool_calls = message.get("tool_calls")
        return bool(tool_calls)

    @staticmethod
    def _format_response_json(raw_response: Any) -> str:
        try:
            return json.dumps(raw_response, ensure_ascii=True)
        except TypeError:
            return str(raw_response)
