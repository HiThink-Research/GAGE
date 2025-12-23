"""OpenAI-compatible HTTP backend built on top of the official SDK."""

from __future__ import annotations

import asyncio
import os
import time
from typing import Any, Dict, Iterable, List, Optional

from loguru import logger

try:  # optional dependency
    from openai import (
        APIStatusError,
        AsyncOpenAI,
        BadRequestError,
        OpenAI,
        PermissionDeniedError,
        UnprocessableEntityError,
    )
    from openai.types.chat import ChatCompletion
except ImportError:  # pragma: no cover - optional dependency
    APIStatusError = BadRequestError = PermissionDeniedError = UnprocessableEntityError = None
    OpenAI = None
    AsyncOpenAI = None
    ChatCompletion = Any  # type: ignore

from gage_eval.role.model.backends.base_backend import EngineBackend
from gage_eval.registry import registry


@registry.asset(
    "backends",
    "openai_http",
    desc="OpenAI Chat Completion 兼容后端",
    tags=("llm", "remote", "api"),
    modalities=("text",),
)
class OpenAICompatibleHTTPBackend(EngineBackend):
    """Backend that talks to any OpenAI Chat Completion compatible endpoint."""

    def load_model(self, config: Dict[str, Any]) -> str:
        # 远端 HTTP 执行模式
        self.execution_mode = "http"
        if OpenAI is None:
            raise RuntimeError("OpenAICompatibleHTTPBackend requires the 'openai' package")
        self.model_name = config.get("model") or config.get("model_name")
        if not self.model_name:
            raise ValueError("OpenAICompatibleHTTPBackend requires 'model' or 'model_name'")

        self.base_url = config.get("base_url") or os.environ.get("OPENAI_BASE_URL") or os.environ.get(
            "EVALSCOPE_BASE_URL"
        )
        if not self.base_url:
            raise ValueError("OpenAICompatibleHTTPBackend requires 'base_url' in config or OPENAI_BASE_URL env")
        self.base_url = self.base_url.rstrip("/").removesuffix("/chat/completions")

        self.require_api_key = bool(config.get("require_api_key", False))
        self.api_key = config.get("api_key") or os.environ.get("OPENAI_API_KEY") or os.environ.get(
            "EVALSCOPE_API_KEY"
        )
        if not self.api_key:
            if self.require_api_key:
                raise ValueError("OpenAICompatibleHTTPBackend requires 'api_key' or OPENAI_API_KEY env")
            self.api_key = config.get("fallback_api_key", "no-key")
            logger.warning(
                "OpenAICompatibleHTTPBackend is running without API key; "
                "set config.require_api_key=true to enforce credential checks."
            )

        client_kwargs = dict(config.get("client_args") or {})
        timeout = config.get("timeout")
        if timeout is not None:
            client_kwargs["timeout"] = timeout
        self.client = OpenAI(base_url=self.base_url, api_key=self.api_key, **client_kwargs)
        self._async_client = None
        env_enable_async = os.environ.get("GAGE_EVAL_ENABLE_ASYNC_HTTP")
        enable_async_cfg = config.get("enable_async")
        enable_async = (
            bool(enable_async_cfg)
            if enable_async_cfg is not None
            else (env_enable_async.lower() in {"1", "true", "yes", "on"} if env_enable_async else False)
        )
        # 默认走同步客户端，避免在 run_sync + 线程模型下反复创建事件循环带来的额外开销；需要时可在配置或环境中显式开启异步。
        if enable_async and AsyncOpenAI is not None:
            try:
                self._async_client = AsyncOpenAI(base_url=self.base_url, api_key=self.api_key, **client_kwargs)
            except Exception as exc:  # pragma: no cover - optional dependency path
                logger.warning("Failed to initialize AsyncOpenAI client: {}", exc)
                self._async_client = None
        if enable_async and AsyncOpenAI is None:
            logger.warning("openai.AsyncOpenAI unavailable; falling back to synchronous client.")
        max_async = int(config.get("async_max_concurrency", 0))
        self._async_semaphore = asyncio.Semaphore(max_async) if max_async > 0 else None

        self.default_params = config.get("default_params", {})
        self.tool_choice_default = config.get("tool_choice")
        self.stream = config.get("stream", False)
        return self.model_name

    # ------------------------------------------------------------------ #
    # Engine interface                                                   #
    # ------------------------------------------------------------------ #
    def prepare_inputs(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        sample = payload.get("sample", {})
        messages = self._resolve_messages(payload, sample)
        sampling_params = self._collect_sampling_params(payload, sample)
        request: Dict[str, Any] = {
            "model": payload.get("model") or self.model_name,
            "messages": messages,
            **sampling_params,
        }

        tool_defs = payload.get("tools") or sample.get("tools")
        if tool_defs:
            request["tools"] = self._format_tools(tool_defs)
            tool_choice = payload.get("tool_choice") or sample.get("tool_choice") or self.tool_choice_default
            if tool_choice:
                request["tool_choice"] = tool_choice
        request["stream"] = payload.get("stream", self.stream)
        return request

    def generate(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        try:
            completion = self.client.chat.completions.create(**inputs)
            if not isinstance(completion, ChatCompletion):
                completion = self._collect_stream(completion)
            answer = self._extract_answer(completion)
            return {
                "answer": answer,
                "raw_response": completion.model_dump(),
                "usage": completion.usage.model_dump() if completion.usage else None,
            }
        except (BadRequestError, UnprocessableEntityError, PermissionDeniedError) as exc:
            return {"error": str(exc), "status": getattr(exc, "status_code", None)}
        except APIStatusError as exc:
            return {"error": str(exc), "status": exc.status_code}

    async def ainvoke(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        if self._async_client is None:
            return await super().ainvoke(payload)
        inputs = self.prepare_inputs(payload)
        start = time.time()
        try:
            if self._async_semaphore is None:
                completion = await self._async_client.chat.completions.create(**inputs)
            else:
                async with self._async_semaphore:
                    completion = await self._async_client.chat.completions.create(**inputs)
            answer = self._extract_answer(completion)
            result = {
                "answer": answer,
                "raw_response": completion.model_dump(),
                "usage": completion.usage.model_dump() if completion.usage else None,
            }
            result.setdefault("latency_ms", (time.time() - start) * 1000)
            return result
        except (BadRequestError, UnprocessableEntityError, PermissionDeniedError) as exc:
            return {"error": str(exc), "status": getattr(exc, "status_code", None)}
        except APIStatusError as exc:
            return {"error": str(exc), "status": exc.status_code}

    # ------------------------------------------------------------------ #
    # Helpers                                                            #
    # ------------------------------------------------------------------ #
    def _resolve_messages(self, payload: Dict[str, Any], sample: Dict[str, Any]) -> List[Dict[str, Any]]:
        messages = payload.get("messages") or sample.get("messages")
        if messages:
            return messages
        system_prompt = payload.get("system_prompt") or sample.get("system_prompt")
        prompt = payload.get("prompt") or sample.get("prompt") or sample.get("text") or ""
        resolved: List[Dict[str, Any]] = []
        if system_prompt:
            resolved.append({"role": "system", "content": system_prompt})
        resolved.append({"role": "user", "content": prompt})
        return resolved

    def _collect_sampling_params(self, payload: Dict[str, Any], sample: Dict[str, Any]) -> Dict[str, Any]:
        params = dict(self.default_params)
        params.update(sample.get("sampling_params") or {})
        params.update(payload.get("sampling_params") or {})
        mapped = {
            "temperature": params.get("temperature"),
            "top_p": params.get("top_p"),
            "max_tokens": params.get("max_new_tokens") or params.get("max_tokens"),
            "presence_penalty": params.get("presence_penalty"),
            "frequency_penalty": params.get("frequency_penalty"),
            "stop": params.get("stop"),
            "n": params.get("n"),
        }
        # remove None entries and unsupported keys
        return {k: v for k, v in mapped.items() if v is not None}

    def _format_tools(self, tools: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
        formatted = []
        for tool in tools:
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
            else:
                formatted.append(tool)
        return formatted

    def _collect_stream(self, response):
        chunks = []
        for event in response:
            chunks.append(event)
        if not chunks:
            raise RuntimeError("Empty streaming response from OpenAI-compatible backend")
        # The SDK already merges streamed chunks when iterated, but to stay safe we reuse last chunk
        last_chunk = chunks[-1]
        if isinstance(last_chunk, ChatCompletion):
            return last_chunk
        raise RuntimeError("Unable to consolidate streaming response into ChatCompletion")

    def _extract_answer(self, completion: ChatCompletion) -> str:
        if not completion.choices:
            return ""
        message = completion.choices[0].message
        if isinstance(message.content, list):
            return "".join(part.get("text", "") for part in message.content if isinstance(part, dict))
        return message.content or ""
