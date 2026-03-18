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

_OPENAI_REQUEST_ERRORS = tuple(
    error for error in (BadRequestError, UnprocessableEntityError, PermissionDeniedError) if error is not None
)
_OPENAI_STATUS_ERRORS = (APIStatusError,) if APIStatusError is not None else ()


@registry.asset(
    "backends",
    "openai_http",
    desc="OpenAI Chat Completions compatible backend",
    tags=("llm", "remote", "api"),
    modalities=("text",),
)
class OpenAICompatibleHTTPBackend(EngineBackend):
    """Backend that talks to any OpenAI Chat Completion compatible endpoint."""

    def load_model(self, config: Dict[str, Any]) -> str:
        # NOTE: Remote HTTP execution mode.
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
        # NOTE: Default to the synchronous client to avoid repeated event-loop creation
        # overhead under `run_sync` + thread execution. Enable async explicitly via
        # config or env vars when needed.
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
            return self._build_result_from_response(completion)
        except _OPENAI_REQUEST_ERRORS as exc:
            return {"error": str(exc), "status": getattr(exc, "status_code", None)}
        except _OPENAI_STATUS_ERRORS as exc:
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
            result = await self._abuild_result_from_response(completion)
            result.setdefault("latency_ms", (time.time() - start) * 1000)
            # NOTE: reasoning_content extraction is handled centrally by EngineBackend
            self._enrich_result_with_reasoning(result)
            return result
        except _OPENAI_REQUEST_ERRORS as exc:
            return {"error": str(exc), "status": getattr(exc, "status_code", None)}
        except _OPENAI_STATUS_ERRORS as exc:
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
        max_completion_tokens = params.get("max_completion_tokens")
        max_tokens = params.get("max_new_tokens") or params.get("max_tokens")
        mapped = {
            "temperature": params.get("temperature"),
            "top_p": params.get("top_p"),
            "max_tokens": None if max_completion_tokens is not None else max_tokens,
            "max_completion_tokens": max_completion_tokens,
            "presence_penalty": params.get("presence_penalty"),
            "frequency_penalty": params.get("frequency_penalty"),
            "stop": params.get("stop"),
            "n": params.get("n"),
        }
        # Reasoning effort for thinking models (e.g. GPT-OSS)
        reasoning_effort = params.get("reasoning_effort") or self._reasoning_effort
        if reasoning_effort:
            mapped["reasoning_effort"] = reasoning_effort
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

    def _build_result_from_response(self, response: Any) -> Dict[str, Any]:
        raw_response = self._normalize_response(response)
        return {
            "answer": self._extract_answer(raw_response),
            "raw_response": raw_response,
            "usage": raw_response.get("usage"),
        }

    async def _abuild_result_from_response(self, response: Any) -> Dict[str, Any]:
        raw_response = await self._anormalize_response(response)
        return {
            "answer": self._extract_answer(raw_response),
            "raw_response": raw_response,
            "usage": raw_response.get("usage"),
        }

    def _normalize_response(self, response: Any) -> Dict[str, Any]:
        if self._looks_like_completion(response):
            raw_response = self._to_dict(response)
            if isinstance(raw_response, dict):
                return raw_response
        if self._is_sync_stream(response):
            return self._collect_stream(response)
        raw_response = self._to_dict(response)
        if isinstance(raw_response, dict):
            return raw_response
        return {"response": raw_response}

    async def _anormalize_response(self, response: Any) -> Dict[str, Any]:
        if self._looks_like_completion(response):
            raw_response = self._to_dict(response)
            if isinstance(raw_response, dict):
                return raw_response
        if self._is_async_stream(response):
            return await self._acollect_stream(response)
        if self._is_sync_stream(response):
            return self._collect_stream(response)
        raw_response = self._to_dict(response)
        if isinstance(raw_response, dict):
            return raw_response
        return {"response": raw_response}

    def _collect_stream(self, response: Any) -> Dict[str, Any]:
        chunks: List[Dict[str, Any]] = []
        for event in response:
            chunk = self._to_dict(event)
            if isinstance(chunk, dict):
                chunks.append(chunk)
        if not chunks:
            raise RuntimeError("Empty streaming response from OpenAI-compatible backend")
        return self._merge_stream_chunks(chunks)

    async def _acollect_stream(self, response: Any) -> Dict[str, Any]:
        chunks: List[Dict[str, Any]] = []
        async for event in response:
            chunk = self._to_dict(event)
            if isinstance(chunk, dict):
                chunks.append(chunk)
        if not chunks:
            raise RuntimeError("Empty streaming response from OpenAI-compatible backend")
        return self._merge_stream_chunks(chunks)

    def _merge_stream_chunks(self, chunks: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Consolidates streaming chunks into a completion-shaped response dict."""

        merged_choices: Dict[int, Dict[str, Any]] = {}
        usage: Optional[Dict[str, Any]] = None
        metadata: Dict[str, Any] = {}

        # STEP 1: Collect response-level metadata and usage from the stream.
        for chunk in chunks:
            for key in ("id", "created", "model", "system_fingerprint", "service_tier"):
                value = chunk.get(key)
                if value is not None:
                    metadata[key] = value
            raw_usage = chunk.get("usage")
            if isinstance(raw_usage, dict):
                usage = raw_usage

            # STEP 2: Merge chunk choices into a final assistant message.
            for position, raw_choice in enumerate(chunk.get("choices") or []):
                choice = raw_choice if isinstance(raw_choice, dict) else self._to_dict(raw_choice)
                if not isinstance(choice, dict):
                    continue
                choice_index = self._coerce_index(choice.get("index"), default=position)
                state = merged_choices.setdefault(
                    choice_index,
                    {
                        "index": choice_index,
                        "finish_reason": None,
                        "message": {"role": "assistant", "content": ""},
                    },
                )
                if choice.get("finish_reason") is not None:
                    state["finish_reason"] = choice.get("finish_reason")
                payload = choice.get("message") or choice.get("delta") or {}
                message_delta = payload if isinstance(payload, dict) else self._to_dict(payload)
                if isinstance(message_delta, dict):
                    self._merge_message_state(state["message"], message_delta)

        if not merged_choices:
            raise RuntimeError("Unable to consolidate streaming response into completion payload")

        metadata["object"] = "chat.completion"
        metadata["choices"] = [merged_choices[idx] for idx in sorted(merged_choices)]
        metadata["stream_chunks"] = chunks
        if usage is not None:
            metadata["usage"] = usage
        return metadata

    def _merge_message_state(self, message_state: Dict[str, Any], delta: Dict[str, Any]) -> None:
        role = delta.get("role")
        if role:
            message_state["role"] = role

        content = self._content_to_text(delta.get("content"))
        if content:
            message_state["content"] = f"{message_state.get('content', '')}{content}"

        reasoning = delta.get("reasoning_content")
        if reasoning is not None:
            message_state["reasoning_content"] = f"{message_state.get('reasoning_content', '')}{reasoning}"

        raw_function_call = delta.get("function_call")
        function_call = raw_function_call if isinstance(raw_function_call, dict) else self._to_dict(raw_function_call)
        if isinstance(function_call, dict):
            merged = message_state.setdefault("function_call", {})
            if function_call.get("name"):
                merged["name"] = function_call["name"]
            arguments = function_call.get("arguments")
            if arguments is not None:
                merged["arguments"] = f"{merged.get('arguments', '')}{arguments}"

        tool_calls = delta.get("tool_calls")
        if isinstance(tool_calls, list):
            merged_tool_calls = message_state.setdefault("tool_calls", [])
            self._merge_tool_calls(merged_tool_calls, tool_calls)

    def _merge_tool_calls(self, merged_tool_calls: List[Dict[str, Any]], tool_calls: List[Any]) -> None:
        for position, raw_tool_call in enumerate(tool_calls):
            tool_call = raw_tool_call if isinstance(raw_tool_call, dict) else self._to_dict(raw_tool_call)
            if not isinstance(tool_call, dict):
                continue
            tool_index = self._coerce_index(tool_call.get("index"), default=position)
            while len(merged_tool_calls) <= tool_index:
                merged_tool_calls.append({})
            merged = merged_tool_calls[tool_index]
            if tool_call.get("id"):
                merged["id"] = tool_call["id"]
            if tool_call.get("type"):
                merged["type"] = tool_call["type"]

            raw_function = tool_call.get("function")
            function = raw_function if isinstance(raw_function, dict) else self._to_dict(raw_function)
            if not isinstance(function, dict):
                continue
            merged_function = merged.setdefault("function", {})
            if function.get("name"):
                merged_function["name"] = function["name"]
            arguments = function.get("arguments")
            if arguments is not None:
                merged_function["arguments"] = f"{merged_function.get('arguments', '')}{arguments}"

    def _looks_like_completion(self, response: Any) -> bool:
        if self._is_chat_completion_instance(response):
            return True
        if isinstance(response, dict):
            if response.get("object") == "chat.completion":
                return True
            return self._choice_has_message(response.get("choices"))
        if getattr(response, "object", None) == "chat.completion":
            return True
        return self._choice_has_message(getattr(response, "choices", None))

    def _choice_has_message(self, choices: Any) -> bool:
        if not choices:
            return False
        first_choice = choices[0]
        if isinstance(first_choice, dict):
            return first_choice.get("message") is not None
        return getattr(first_choice, "message", None) is not None

    def _is_chat_completion_instance(self, response: Any) -> bool:
        return isinstance(ChatCompletion, type) and isinstance(response, ChatCompletion)

    def _is_sync_stream(self, response: Any) -> bool:
        if isinstance(response, (str, bytes, dict, list, tuple)):
            return False
        return callable(getattr(response, "__iter__", None))

    def _is_async_stream(self, response: Any) -> bool:
        return callable(getattr(response, "__aiter__", None))

    def _to_dict(self, value: Any) -> Any:
        if value is None or isinstance(value, (str, int, float, bool)):
            return value
        if isinstance(value, dict):
            return {key: self._to_dict(item) for key, item in value.items()}
        if isinstance(value, list):
            return [self._to_dict(item) for item in value]
        model_dump = getattr(value, "model_dump", None)
        if callable(model_dump):
            dumped = model_dump()
            return self._to_dict(dumped)
        if hasattr(value, "__dict__"):
            return {key: self._to_dict(item) for key, item in vars(value).items() if not key.startswith("_")}
        return value

    def _content_to_text(self, content: Any) -> str:
        if content is None:
            return ""
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            return "".join(self._content_to_text(part) for part in content)
        if isinstance(content, dict):
            text = content.get("text")
            if text is not None:
                return str(text)
            if content.get("type") == "text":
                return str(content.get("content", ""))
        text_attr = getattr(content, "text", None)
        if text_attr is not None:
            return str(text_attr)
        return str(content)

    def _coerce_index(self, value: Any, *, default: int) -> int:
        try:
            return int(value)
        except (TypeError, ValueError):
            return default

    def _extract_answer(self, completion: Any) -> str:
        choices = completion.get("choices") if isinstance(completion, dict) else getattr(completion, "choices", None)
        if not choices:
            return ""
        first_choice = choices[0]
        if isinstance(first_choice, dict):
            message = first_choice.get("message") or first_choice.get("delta") or {}
            return self._content_to_text(message.get("content"))
        message = getattr(first_choice, "message", None) or getattr(first_choice, "delta", None)
        content = getattr(message, "content", None) if message is not None else None
        return self._content_to_text(content)
