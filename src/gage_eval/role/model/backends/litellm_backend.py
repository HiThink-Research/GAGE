"""LiteLLM backend adaptor."""

from __future__ import annotations

import time
import os
from typing import Any, Dict, List, Optional, Tuple

import requests
from loguru import logger

from gage_eval.role.model.backends.base_backend import EngineBackend
from gage_eval.role.model.config.litellm import LiteLLMBackendConfig
from gage_eval.registry import registry


@registry.asset(
    "backends",
    "litellm",
    desc="通过 LiteLLM 统一接入各类推理服务，含 Kimi 兜底直连",
    tags=("llm", "remote", "api"),
    modalities=("text",),
)
class LiteLLMBackend(EngineBackend):
    """LiteLLM 统一接口 + Kimi 直连兜底的后端。

    - 优先通过 liteLLM 适配多厂商，带重试与 stop 序列清洗。
    - 当目标为 Kimi 且 liteLLM 不可用/不支持时，退化为 OpenAI 兼容的 HTTP 调用。
    """

    def __init__(self, config: Dict[str, Any]) -> None:
        self.transport = "http"
        self.http_retry_params: Dict[str, Any] = {}
        self._litellm = None
        self._supports_reasoning_fn = None
        self._mock_mode = False
        super().__init__(config)

    # ------------------------------------------------------------------ #
    # Engine interface                                                   #
    # ------------------------------------------------------------------ #
    def load_model(self, config_dict: Dict[str, Any]):
        self._cfg = LiteLLMBackendConfig(**config_dict)
        self.model_name = self._cfg.model
        self.provider = self._cfg.provider or self._infer_provider(self.model_name)
        self.api_base = self._cfg.api_base
        self.api_key = (
            self._cfg.api_key
            or os.getenv("LITELLM_API_KEY")
            or os.getenv("OPENAI_API_KEY")
            or os.getenv("KIMI_API_KEY")
            or os.getenv("MOONSHOT_API_KEY")
        )
        self.headers = dict(self._cfg.extra_headers or {})
        self._timeout = self._cfg.timeout
        self._max_retries = max(1, int(self._cfg.max_retries))
        self._retry_sleep = float(self._cfg.retry_sleep)
        self._retry_multiplier = max(1.0, float(self._cfg.retry_multiplier))
        self._max_context_length = self._cfg.max_model_length
        self._force_kimi_direct = bool(self._cfg.force_kimi_direct)
        self._prefer_litellm_kimi = bool(self._cfg.prefer_litellm_kimi)
        self._base_sampling = self._cfg.generation_parameters.to_dict()
        self.http_retry_params = {"attempts": self._max_retries, "interval": self._retry_sleep}
        self._is_kimi_target = self._looks_like_kimi(self.provider, self.model_name, self.api_base)
        if self._is_kimi_target and not self.api_base:
            self.api_base = "https://api.moonshot.cn/v1"

        # 可选：无额度时使用本地/自建 OpenAI 兼容服务（例如本地 qwen）模拟调用
        mock_base = self._cfg.mock_api_base or os.getenv("LITELLM_MOCK_API_BASE")
        if mock_base:
            self._mock_mode = True
            self.api_base = mock_base
            self.api_key = self._cfg.mock_api_key or os.getenv("LITELLM_MOCK_API_KEY") or self.api_key
            self.model_name = self._cfg.mock_model or self.model_name
            # litellm 需要 provider，默认保持/回落到 openai
            self.provider = self._cfg.provider or self.provider or "openai"
            self._is_kimi_target = False
            logger.warning("LiteLLM 启用本地模拟模式: base={} model={}", self.api_base, self.model_name)

        self._litellm_available = False
        try:  # pragma: no cover - optional dependency
            import litellm  # type: ignore
        except ImportError as exc:  # pragma: no cover
            if not self._is_kimi_target:
                raise RuntimeError("liteLLM is not installed") from exc
            logger.warning("liteLLM 未安装，Kimi 将走直连 HTTP 路径")
            return None

        self._litellm_available = True
        self._litellm = litellm
        self._supports_reasoning_fn = getattr(litellm, "supports_reasoning", None)
        litellm.drop_params = True
        litellm.verbose = bool(self._cfg.verbose)
        if self.api_key:
            litellm.api_key = self.api_key
        if self.api_base:
            litellm.api_base = self.api_base
        if self.headers:
            litellm.headers = self.headers
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

        sampling_params = dict(self._base_sampling)
        sampling_params.update(sample.get("sampling_params") or {})
        sampling_params.update(payload.get("sampling_params") or {})

        return {
            "model": payload.get("model") or self.model_name,
            "messages": messages,
            "sampling_params": sampling_params,
            "stream": bool(payload.get("stream", self._cfg.streaming)),
            "num_samples": payload.get("num_samples") or sampling_params.get("n"),
        }

    def generate(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        if self._mock_mode:
            return self._generate_mock_http(inputs)
        if self._is_kimi_target and (self._force_kimi_direct or not self._litellm_available):
            return self._generate_kimi_http(inputs)

        try:
            return self._generate_litellm(inputs)
        except Exception as exc:
            if self._is_kimi_target and not self._prefer_litellm_kimi:
                logger.warning("LiteLLM 路径失败，尝试 Kimi 直连兜底: {}", exc)
                return self._generate_kimi_http(inputs)
            raise

    # ------------------------------------------------------------------ #
    # LiteLLM 路径                                                      #
    # ------------------------------------------------------------------ #
    def _generate_litellm(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        kwargs, _ = self._build_litellm_kwargs(inputs)
        stream = kwargs.pop("stream", False)
        start = time.time()

        def _call():
            return self._litellm.completion(stream=stream, **kwargs)

        completion = self._call_with_retries(_call)
        if stream:
            answer, raw_response = self._collect_stream(completion)
        else:
            answer = self._extract_answer(completion)
            raw_response = completion
            # LiteLLM 返回的 ModelResponse 不是原生 JSON，可转成 dict 以便后续缓存/序列化。
            if hasattr(completion, "model_dump"):
                raw_response = completion.model_dump()
            elif hasattr(completion, "dict"):
                try:
                    raw_response = completion.dict()
                except Exception:
                    pass
        result = {"answer": answer, "raw_response": raw_response}
        if not stream and hasattr(completion, "usage"):
            usage = getattr(completion, "usage")
            if hasattr(usage, "model_dump"):
                result["usage"] = usage.model_dump()
        result.setdefault("latency_ms", (time.time() - start) * 1000)
        return result

    def _build_litellm_kwargs(self, inputs: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        sampling_params = self._normalize_sampling_params(inputs.get("sampling_params") or {}, inputs.get("num_samples"))
        stop_sequences = self._prepare_stop_sequences(sampling_params.get("stop"))
        provider = self.provider
        if provider == "google" and self.api_base and "127.0.0.1:18082" in self.api_base:
            provider = "openai"  # 本地 google mock 走 openai 兼容路径

        kwargs: Dict[str, Any] = {
            "model": inputs.get("model") or self.model_name,
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
        if provider:
            kwargs["provider"] = provider
            kwargs.setdefault("custom_llm_provider", provider)
        if self.headers:
            kwargs["headers"] = self.headers
        kwargs.update({k: v for k, v in sampling_params.items() if v is not None})
        if stop_sequences:
            kwargs["stop"] = stop_sequences
        kwargs.setdefault("max_completion_tokens", kwargs.get("max_tokens"))
        kwargs.setdefault("n", inputs.get("num_samples"))
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
    # Kimi 直连路径                                                      #
    # ------------------------------------------------------------------ #
    def _generate_kimi_http(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        url = (self.api_base or "https://api.moonshot.cn/v1").rstrip("/") + "/chat/completions"
        sampling_params = self._normalize_sampling_params(inputs.get("sampling_params") or {}, inputs.get("num_samples"))
        stop_sequences = self._prepare_stop_sequences(sampling_params.get("stop"))

        payload: Dict[str, Any] = {
            "model": inputs.get("model") or self.model_name,
            "messages": inputs.get("messages") or [],
        }
        for key in ("temperature", "top_p", "presence_penalty", "frequency_penalty", "repetition_penalty"):
            if sampling_params.get(key) is not None:
                payload[key] = sampling_params[key]
        if stop_sequences:
            payload["stop"] = stop_sequences
        if sampling_params.get("max_tokens") is not None:
            payload["max_tokens"] = sampling_params["max_tokens"]
        if sampling_params.get("n") is not None:
            payload["n"] = sampling_params["n"]

        headers = {"Authorization": f"Bearer {self.api_key or ''}", "Content-Type": "application/json"}
        headers.update(self.headers)

        start = time.time()

        def _call():
            response = requests.post(url, json=payload, headers=headers, timeout=self._timeout or 60)
            response.raise_for_status()
            return response.json()

        data = self._call_with_retries(_call)
        answer = self._extract_answer(data)
        return {
            "answer": answer,
            "raw_response": data,
            "latency_ms": (time.time() - start) * 1000,
        }

    def _generate_mock_http(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """本地/自建 OpenAI 兼容服务的通用调用（用于无额度时的协议模拟）。"""

        base = (self.api_base or "").rstrip("/") or "http://127.0.0.1:1234"
        url = base + "/chat/completions"
        sampling_params = self._normalize_sampling_params(inputs.get("sampling_params") or {}, inputs.get("num_samples"))
        stop_sequences = self._prepare_stop_sequences(sampling_params.get("stop"))

        payload: Dict[str, Any] = {
            "model": inputs.get("model") or self.model_name,
            "messages": inputs.get("messages") or [],
        }
        for key in ("temperature", "top_p", "presence_penalty", "frequency_penalty", "repetition_penalty"):
            if sampling_params.get(key) is not None:
                payload[key] = sampling_params[key]
        if stop_sequences:
            payload["stop"] = stop_sequences
        if sampling_params.get("max_tokens") is not None:
            payload["max_tokens"] = sampling_params["max_tokens"]
        if sampling_params.get("n") is not None:
            payload["n"] = sampling_params["n"]

        headers = {"Authorization": f"Bearer {self.api_key or ''}", "Content-Type": "application/json"}
        headers.update(self.headers)

        start = time.time()

        def _call():
            response = requests.post(url, json=payload, headers=headers, timeout=self._timeout or 60)
            response.raise_for_status()
            return response.json()

        data = self._call_with_retries(_call)
        answer = self._extract_answer(data)
        return {
            "answer": answer,
            "raw_response": data,
            "latency_ms": (time.time() - start) * 1000,
        }

    # ------------------------------------------------------------------ #
    # Helpers                                                            #
    # ------------------------------------------------------------------ #
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
            except Exception as exc:  # pragma: no cover - 网络/第三方异常
                last_exc = exc
                if attempt == self._max_retries - 1:
                    break
                wait = min(64, self._retry_sleep * (self._retry_multiplier**attempt))
                logger.warning("LiteLLM/Kimi 调用失败，重试 {}/{}，等待 {:.1f}s: {}", attempt + 1, self._max_retries, wait, exc)
                time.sleep(wait)
        assert last_exc is not None
        raise last_exc

    @staticmethod
    def _infer_provider(model_name: str | None) -> Optional[str]:
        if not model_name or "/" not in model_name:
            return None
        return model_name.split("/")[0]

    @staticmethod
    def _looks_like_kimi(provider: Optional[str], model: str, api_base: Optional[str] = None) -> bool:
        target = (provider or "").lower()
        model_lower = (model or "").lower()
        base = (api_base or "").lower()
        return target in {"kimi", "moonshot"} or model_lower.startswith("moonshot") or "moonshot" in base or "kimi" in base

    @staticmethod
    def _content_piece(part: Any) -> str:
        if isinstance(part, dict) and "text" in part:
            return str(part["text"])
        return str(part)
