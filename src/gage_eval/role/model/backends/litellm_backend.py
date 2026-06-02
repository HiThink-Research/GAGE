"""LiteLLM backend adaptor."""

from __future__ import annotations

import asyncio
import inspect
import json
import os
import time
from collections.abc import Mapping
from typing import Any, Dict, List, Tuple

from loguru import logger

from gage_eval.role.model.backends.base_backend import EngineBackend
from gage_eval.role.model.backends.litellm.credential_resolver import ProviderCredentialResolver
from gage_eval.role.model.backends.litellm.errors import (
    DEPENDENCY_UNAVAILABLE,
    LiteLLMBackendError,
    status_code_from_error,
)
from gage_eval.role.model.backends.litellm.message_normalizer import MultimodalMessageNormalizer
from gage_eval.role.model.backends.litellm.model_server_lifecycle import ModelServerLifecycle
from gage_eval.role.model.backends.litellm.policies import ThinkingControlPolicy, ToolCallPolicy
from gage_eval.role.model.backends.litellm.provider_detection import (
    infer_provider_from_model,
    is_default_local_litellm_gateway,
    is_explicit_litellm_gateway,
    looks_like_azure,
    looks_like_deepseek,
    looks_like_grok,
    looks_like_kimi,
    normalize_custom_provider,
)
from gage_eval.role.model.backends.litellm.request_builder import LiteLLMRequestBuilder
from gage_eval.role.model.backends.litellm.response_normalizer import LiteLLMResponseNormalizer
from gage_eval.role.model.backends.litellm.router_factory import LiteLLMRouterFactory
from gage_eval.role.model.backends.litellm.service_profile import VLLMServiceProfile
from gage_eval.role.model.config.litellm import LiteLLMBackendConfig, assert_litellm_capabilities
from gage_eval.registry import registry
from gage_eval.assets.datasets.utils.multimodal import embed_remote_image_as_data_url
from gage_eval.utils.messages import normalize_messages_for_template


ROUTER_COMPLETION_DIRECT_TRANSPORT_FIELDS = frozenset(
    {
        "api_base",
        "base_url",
        "api_key",
        "custom_llm_provider",
        "api_type",
        "api_version",
        "headers",
    }
)


@registry.asset(
    "backends",
    "litellm",
    desc="LiteLLM backend for unified provider access (Grok/Kimi base URLs + param normalization)",
    tags=("llm", "remote", "api"),
    modalities=("text", "vision", "audio"),
)
class LiteLLMBackend(EngineBackend):
    """LiteLLM backend with provider inference and sampling normalization."""

    def __init__(self, config: Dict[str, Any]) -> None:
        self.http_retry_mode = "native"
        self.transport = "http"
        self._litellm: Any | None = None
        self._router: Any | None = None
        self._service_profile: VLLMServiceProfile | None = None
        self._thinking_policy: ThinkingControlPolicy | None = None
        self._tool_call_policy: ToolCallPolicy | None = None
        self._response_normalizer: LiteLLMResponseNormalizer | None = None
        self._message_normalizer: MultimodalMessageNormalizer | None = None
        self._supports_reasoning_fn: Any | None = None
        self._supports_function_calling_fn: Any | None = None
        self._custom_llm_provider: str | None = None
        self._model_server_lifecycle = config.get("_model_server_lifecycle") or ModelServerLifecycle()
        super().__init__(config)

    # ------------------------------------------------------------------ #
    # Engine interface                                                   #
    # ------------------------------------------------------------------ #
    def load_model(self, config_dict: Dict[str, Any]):
        # STEP 1: Resolve provider-specific routing, credentials, and sampling defaults.
        self._cfg = LiteLLMBackendConfig(**config_dict)
        self._apply_deprecated_mock_config()
        if self._cfg.model_server.enabled:
            self._model_server_lifecycle.ensure_ready(self._cfg.model_server)
        self._tool_choice_default = config_dict.get("tool_choice")
        self.model_name = self._cfg.model
        self.provider = self._cfg.provider or infer_provider_from_model(self.model_name)
        self.api_base = self._cfg.api_base
        self._is_deepseek_target = looks_like_deepseek(self.provider, self.model_name, self.api_base)
        self.api_key: str | None = None
        self.headers = dict(self._cfg.extra_headers or {})
        self._timeout = self._cfg.timeout
        self._max_retries = self._resolve_backend_retry_budget()
        self._retry_sleep = float(self._cfg.retry_sleep)
        self._retry_multiplier = max(1.0, float(self._cfg.retry_multiplier))
        self._max_context_length = self._cfg.max_model_length
        self._base_sampling = self._cfg.generation_parameters.to_dict()
        self._embed_remote_images = bool(self._cfg.embed_remote_images)
        self._remote_image_timeout_s = float(self._cfg.remote_image_timeout_s)
        if self._embed_remote_images:
            logger.info("LiteLLM remote image embedding enabled (timeout_s={})", self._remote_image_timeout_s)
        self._is_kimi_target = looks_like_kimi(self.provider, self.model_name, self.api_base)
        self._is_grok_target = looks_like_grok(self.provider, self.model_name, self.api_base)
        self._is_azure_target = looks_like_azure(self.provider, self.model_name, self.api_base)
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

        # NOTE: Normalize `custom_llm_provider`: prefer an explicit user value, then
        # fall back to provider inference.
        self._custom_llm_provider = normalize_custom_provider(
            getattr(self._cfg, "custom_llm_provider", None) or self.provider
        )
        self.api_key = ProviderCredentialResolver(
            self._cfg,
            provider=self.provider,
            custom_llm_provider=self._custom_llm_provider,
            topology_kind=self._vllm_topology_kind(),
        ).resolve()
        self._azure_api_version = self._cfg.azure_api_version or os.getenv("AZURE_OPENAI_API_VERSION")
        if self._is_azure_target and not self._azure_api_version:
            self._azure_api_version = "2024-02-15-preview"

        if self._cfg.force_kimi_direct or self._cfg.prefer_litellm_kimi:
            logger.warning("force_kimi_direct/prefer_litellm_kimi 已废弃，LiteLLM 现统一走 litellm 调用路径")

        try:  # pragma: no cover - optional dependency
            import litellm  # type: ignore
        except ImportError as exc:  # pragma: no cover
            raise LiteLLMBackendError("LiteLLM is not installed", DEPENDENCY_UNAVAILABLE) from exc
        assert_litellm_capabilities(litellm_module=litellm)

        # STEP 2: Capture the imported module. Request-specific options are passed
        # through completion kwargs to avoid mutating LiteLLM module globals.
        self._litellm = litellm
        self._supports_reasoning_fn = getattr(litellm, "supports_reasoning", None)
        self._supports_function_calling_fn = getattr(litellm, "supports_function_calling", None)
        service_profile = VLLMServiceProfile.from_config(self._cfg)
        self._service_profile = service_profile
        for warning in service_profile.validate_deployment_consistency():
            logger.warning("LiteLLM service profile: {}", warning)
        self._message_normalizer = MultimodalMessageNormalizer(
            self._cfg.multimodal,
            service_profile=service_profile,
        )
        self._thinking_policy = ThinkingControlPolicy.from_config(
            self._cfg.thinking_policy,
            service_profile=service_profile,
        )
        self._tool_call_policy = ToolCallPolicy.from_config(self._cfg.tool_calling)
        self._response_normalizer = LiteLLMResponseNormalizer()
        self._router = LiteLLMRouterFactory(litellm).build(self._cfg)
        return None

    def _apply_deprecated_mock_config(self) -> None:
        deprecated_fields = []
        if self._cfg.mock_api_base:
            deprecated_fields.append("mock_api_base")
            if not self._cfg.api_base:
                self._cfg.api_base = self._cfg.mock_api_base
        if self._cfg.mock_api_key:
            deprecated_fields.append("mock_api_key")
            if not self._cfg.api_key:
                self._cfg.api_key = self._cfg.mock_api_key
        if self._cfg.mock_model:
            deprecated_fields.append("mock_model")
            if not self._cfg._field_was_explicitly_set("model"):
                self._cfg.model = self._cfg.mock_model
        if deprecated_fields:
            logger.warning(
                "LiteLLM config fields "
                + ", ".join(deprecated_fields)
                + " are deprecated; use api_base/api_key/model instead."
            )

    def _resolve_backend_retry_budget(self) -> int:
        if self._cfg.resolved_route_mode() == "router":
            return 1
        if self._looks_like_litellm_gateway_endpoint():
            return 1
        return max(1, int(self._cfg.max_retries))

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

        sample_sampling_params = sample.get("sampling_params") or {}
        payload_sampling_params = payload.get("sampling_params") or {}
        runtime_sampling_params = dict(sample_sampling_params)
        runtime_sampling_params.update(payload_sampling_params)

        sampling_params = dict(self._base_sampling)
        sampling_params.update(sample_sampling_params)
        sampling_params.update(payload_sampling_params)
        tool_defs = payload.get("tools") or sample.get("tools")
        tool_choice = payload.get("tool_choice") or sample.get("tool_choice") or self._tool_choice_default
        parallel_tool_calls = payload.get("parallel_tool_calls")
        if parallel_tool_calls is None:
            parallel_tool_calls = sample.get("parallel_tool_calls")

        return {
            "model": payload.get("model") or self.model_name,
            "messages": messages,
            "sampling_params": sampling_params,
            "runtime_sampling_params": runtime_sampling_params,
            "tools": tool_defs,
            "tool_choice": tool_choice,
            "parallel_tool_calls": parallel_tool_calls,
            "stream": bool(payload.get("stream", self._cfg.streaming)),
            "num_samples": payload.get("num_samples") or sampling_params.get("n"),
        }

    def generate(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        return self._generate_litellm(inputs)

    def invoke(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        inputs = self.prepare_inputs(payload)
        start = time.time()
        result = self.generate(inputs)
        result.setdefault("latency_ms", (time.time() - start) * 1000)
        self._enrich_result_with_reasoning(result)
        logger.debug(
            "Backend {} finished request latency={:.2f}ms",
            self.__class__.__name__,
            result["latency_ms"],
        )
        return result

    async def ainvoke(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        inputs = self.prepare_inputs(payload)
        start = time.time()
        result = await self._agenerate_litellm(inputs)
        result.setdefault("latency_ms", (time.time() - start) * 1000)
        self._enrich_result_with_reasoning(result)
        logger.debug(
            "Backend {} finished request latency={:.2f}ms",
            self.__class__.__name__,
            result["latency_ms"],
        )
        return result

    # ------------------------------------------------------------------ #
    # LiteLLM call path                                                 #
    # ------------------------------------------------------------------ #
    def _generate_litellm(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        # STEP 1: Build isolated request kwargs and execute the LiteLLM call path.
        kwargs, request_kwargs, request_context, stream = self._prepare_completion_call(inputs)
        start = time.time()

        def _call():
            completion_target = self._router or self._litellm
            return completion_target.completion(stream=stream, **kwargs)

        completion, outer_retry_attempts = self._call_with_retries(_call)
        request_context["outer_retry_attempts"] = outer_retry_attempts

        # STEP 2: Normalize the response for downstream consumers and safe diagnostics.
        latency_ms = (time.time() - start) * 1000
        return self._normalize_litellm_completion(
            completion,
            stream=stream,
            latency_ms=latency_ms,
            request_kwargs=request_kwargs,
            request_context=request_context,
        )

    async def _agenerate_litellm(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        kwargs, request_kwargs, request_context, stream = self._prepare_completion_call(inputs)
        start = time.time()
        if self._router is not None:
            acompletion = getattr(self._router, "acompletion", None)
            if not callable(acompletion):
                raise RuntimeError(
                    "LiteLLM Router async path requires router.acompletion; "
                    "upgrade LiteLLM or use a Router implementation with async completion support."
                )
        else:
            acompletion = getattr(self._litellm, "acompletion", None)
            if not callable(acompletion):
                raise RuntimeError(
                    "LiteLLM async path requires litellm.acompletion; "
                    "upgrade LiteLLM or use the synchronous generate path."
                )

        async def _call():
            return await acompletion(stream=stream, **kwargs)

        completion, outer_retry_attempts = await self._acall_with_retries(_call)
        request_context["outer_retry_attempts"] = outer_retry_attempts
        if stream:
            completion = await self._collect_async_stream(completion)
        latency_ms = (time.time() - start) * 1000
        return self._normalize_litellm_completion(
            completion,
            stream=stream,
            latency_ms=latency_ms,
            request_kwargs=request_kwargs,
            request_context=request_context,
        )

    def _prepare_completion_call(
        self,
        inputs: Dict[str, Any],
    ) -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any], bool]:
        if "messages" in inputs:
            inputs = dict(inputs)
            inputs["messages"] = self._normalize_messages_for_provider(inputs.get("messages") or [])
        kwargs, _ = self._build_litellm_kwargs(inputs)
        request_kwargs = dict(kwargs)
        request_context = self._build_request_context(request_kwargs)
        if self._router is not None:
            kwargs = self._router_completion_kwargs(kwargs)
        stream = kwargs.pop("stream", False)
        request_context["stream"] = stream
        return kwargs, request_kwargs, request_context, bool(stream)

    def _normalize_litellm_completion(
        self,
        completion: Any,
        *,
        stream: bool,
        latency_ms: float,
        request_kwargs: Dict[str, Any],
        request_context: Dict[str, Any],
    ) -> Dict[str, Any]:
        request_context["latency_ms"] = latency_ms
        normalizer = self._response_normalizer or LiteLLMResponseNormalizer()
        if stream:
            result = normalizer.normalize_stream(completion, request_context=request_context)
        else:
            result = normalizer.normalize(completion, request_context=request_context)
        result.setdefault("latency_ms", latency_ms)
        observation_summary = normalizer.build_observation_summary(
            request_kwargs,
            result,
            request_context=request_context,
        )
        result.setdefault("metadata", {})
        result["metadata"]["observation_summary"] = observation_summary
        logger.info(
            "LiteLLM response summary: {}",
            self._format_response_json(observation_summary),
        )
        return result

    @staticmethod
    async def _collect_async_stream(stream_response: Any) -> List[Any]:
        if not hasattr(stream_response, "__aiter__"):
            if inspect.isawaitable(stream_response):
                stream_response = await stream_response
            if not hasattr(stream_response, "__aiter__"):
                if isinstance(stream_response, Mapping):
                    return [stream_response]
                if isinstance(stream_response, (str, bytes, bytearray)):
                    raise RuntimeError(
                        "LiteLLM async streaming expected an async iterator or completion-like mapping, "
                        f"got {type(stream_response).__name__}."
                    )
                try:
                    return list(stream_response)
                except TypeError:
                    return [stream_response]
        chunks: List[Any] = []
        async for chunk in stream_response:
            chunks.append(chunk)
        return chunks

    @staticmethod
    def _router_completion_kwargs(kwargs: Dict[str, Any]) -> Dict[str, Any]:
        return {
            key: value
            for key, value in kwargs.items()
            if key not in ROUTER_COMPLETION_DIRECT_TRANSPORT_FIELDS
        }

    def _build_litellm_kwargs(self, inputs: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        builder = LiteLLMRequestBuilder(
            model_name=self.model_name,
            provider=self.provider,
            api_base=self.api_base,
            api_key=self.api_key,
            timeout=self._timeout,
            headers=self.headers,
            custom_llm_provider=self._custom_llm_provider,
            base_sampling=self._base_sampling,
            litellm_request=self._cfg.litellm_request,
            drop_params=self._cfg.drop_params,
            is_azure_target=self._is_azure_target,
            azure_api_version=self._azure_api_version,
            max_context_length=self._max_context_length,
            supports_reasoning=self._supports_reasoning_model(),
            supports_function_calling=self._supports_function_calling_model(inputs.get("model") or self.model_name),
            thinking_policy=self._thinking_policy,
            tool_call_policy=self._tool_call_policy,
        )
        return builder.build(inputs, thinking_config=self.get_thinking_config())

    def _build_request_context(self, request_kwargs: Dict[str, Any]) -> Dict[str, Any]:
        """Builds non-sensitive request metadata for response normalization."""

        route_mode = self._cfg.resolved_route_mode()
        resolved_thinking_mode = self._resolved_thinking_mode_from_kwargs(request_kwargs)
        retry_owner = self._retry_owner(route_mode)
        context = {
            "provider": self.provider or self._custom_llm_provider,
            "model": request_kwargs.get("model") or self.model_name,
            "api_base": request_kwargs.get("api_base") or request_kwargs.get("base_url") or self.api_base,
            "route_mode": route_mode,
            "topology": dict(self._service_profile.topology) if self._service_profile else {},
            "thinking_mode": resolved_thinking_mode,
            "resolved_thinking_mode": resolved_thinking_mode,
            "thinking_policy": self._cfg.thinking_policy,
            "retry_owner": retry_owner,
        }
        if route_mode == "router":
            context["router_num_retries"] = self._cfg.router_settings.num_retries
        if self._thinking_inherited_from_server_default(resolved_thinking_mode):
            context["thinking_inherited_from_server_default"] = True
        return context

    def _resolved_thinking_mode_from_kwargs(self, request_kwargs: Dict[str, Any]) -> str:
        extra_body = request_kwargs.get("extra_body")
        if isinstance(extra_body, dict):
            chat_template_kwargs = extra_body.get("chat_template_kwargs")
            if isinstance(chat_template_kwargs, dict):
                enable_thinking = chat_template_kwargs.get("enable_thinking")
                if enable_thinking is True:
                    return "enabled"
                if enable_thinking is False:
                    return "disabled"
        if self._thinking_mode:
            return str(self._thinking_mode)
        mode = self._base_sampling.get("thinking_mode")
        return str(mode) if mode else "auto"

    def _supports_reasoning_model(self) -> bool:
        if not self._supports_reasoning_fn:
            return False
        try:
            return bool(self._supports_reasoning_fn(self.model_name))
        except Exception:  # pragma: no cover - defensive around third-party helpers
            return False

    def _supports_function_calling_model(self, model_name: str | None = None) -> bool | None:
        if not callable(self._supports_function_calling_fn):
            return None
        try:
            result = self._supports_function_calling_fn(model_name or self.model_name)
        except Exception:  # pragma: no cover - defensive around third-party helpers
            return None
        if result is None:
            return None
        return bool(result)

    # ------------------------------------------------------------------ #
    # Helpers                                                            #
    # ------------------------------------------------------------------ #
    def _normalize_messages_for_provider(self, messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Normalize message content according to the target provider's input contract."""

        if self._should_flatten_multimodal_messages():
            return normalize_messages_for_template(messages, image_placeholder="<image>")
        normalizer = self._message_normalizer or MultimodalMessageNormalizer(self._cfg.multimodal)
        normalized = normalizer.normalize(messages, service_profile=self._service_profile)
        if self._embed_remote_images:
            return self._embed_remote_image_urls_in_messages(normalized)
        return normalized

    def _embed_remote_image_urls_in_messages(self, messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        embedded_messages: List[Dict[str, Any]] = []
        for message in messages or []:
            new_message = dict(message)
            content = message.get("content")
            if isinstance(content, list):
                new_message["content"] = [self._embed_remote_image_url_in_block(block) for block in content]
            embedded_messages.append(new_message)
        return embedded_messages

    def _embed_remote_image_url_in_block(self, block: Any) -> Any:
        if not isinstance(block, Mapping):
            return block
        if block.get("type") != "image_url":
            return dict(block)

        new_block = dict(block)
        image_url = block.get("image_url")
        if isinstance(image_url, Mapping):
            payload = dict(image_url)
            url = payload.get("url")
            if isinstance(url, str):
                payload["url"] = self._maybe_embed_remote_image_url(url)
            new_block["image_url"] = payload
            return new_block
        if isinstance(image_url, str):
            new_block["image_url"] = {"url": self._maybe_embed_remote_image_url(image_url)}
            return new_block
        return new_block

    def _maybe_embed_remote_image_url(self, url: str) -> str:
        if not self._embed_remote_images:
            return url
        if not url.startswith(("http://", "https://")):
            return url
        embedded = embed_remote_image_as_data_url(
            url,
            strict=False,
            timeout_s=self._remote_image_timeout_s,
        )
        if embedded is None:
            logger.debug("Remote image embedding failed or was skipped for an http(s) image URL")
        return embedded or url

    def _should_flatten_multimodal_messages(self) -> bool:
        provider = (self._custom_llm_provider or self.provider or "").lower()
        return provider == "deepseek" or looks_like_deepseek(self.provider, self.model_name, self.api_base)

    def _call_with_retries(self, func):
        last_exc: Exception | None = None
        for attempt in range(self._max_retries):
            try:
                return func(), attempt + 1
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
        if last_exc is None:
            raise LiteLLMBackendError("LiteLLM retry loop exited without capturing an exception", DEPENDENCY_UNAVAILABLE)
        self._raise_classified_retry_failure(last_exc)
        raise last_exc

    async def _acall_with_retries(self, func):
        last_exc: Exception | None = None
        for attempt in range(self._max_retries):
            try:
                return await func(), attempt + 1
            except Exception as exc:  # pragma: no cover - network/third-party errors
                last_exc = exc
                if self._is_non_retryable_error(exc):
                    logger.debug("LiteLLM async call aborted without retry due to non-retryable error: {}", exc)
                    break
                if attempt == self._max_retries - 1:
                    break
                wait = min(64, self._retry_sleep * (self._retry_multiplier**attempt))
                logger.warning(
                    "LiteLLM async 调用失败，重试 {}/{}，等待 {:.1f}s: {}",
                    attempt + 1,
                    self._max_retries,
                    wait,
                    exc,
                )
                await asyncio.sleep(wait)
        if last_exc is None:
            raise LiteLLMBackendError("LiteLLM retry loop exited without capturing an exception", DEPENDENCY_UNAVAILABLE)
        self._raise_classified_retry_failure(last_exc)
        raise last_exc

    @staticmethod
    def _is_non_retryable_error(exc: Exception) -> bool:
        """Return True when an error should bypass retry loops."""

        status_code = status_code_from_error(exc)
        if status_code in {400, 401, 403, 404, 422}:
            return True
        message = str(exc).strip().lower()
        if not message:
            return False
        non_retryable_markers = (
            "cannot schedule new futures after shutdown",
            "rolepool",
            "is shut down",
        )
        if any(marker in message for marker in non_retryable_markers):
            return True
        return False

    def _retry_owner(self, route_mode: str) -> str:
        if route_mode == "router":
            return "litellm_router"
        if self._looks_like_litellm_gateway_endpoint():
            return "litellm_gateway"
        return "gage"

    def _thinking_inherited_from_server_default(self, resolved_thinking_mode: str) -> bool:
        if resolved_thinking_mode != "auto":
            return False
        if not self._service_profile or not self._service_profile.reasoning_parser:
            return False
        model = (self.model_name or "").lower()
        provider = (self.provider or "").lower().replace("-", "_")
        custom_provider = (self._custom_llm_provider or "").lower().replace("-", "_")
        return model.startswith("hosted_vllm/") or bool(
            {provider, custom_provider} & {"hosted_vllm", "vllm", "openai_compatible"}
        )

    def close(self) -> None:
        close = getattr(self._model_server_lifecycle, "close", None)
        if callable(close):
            close()

    def shutdown(self) -> None:
        self.close()

    def _looks_like_litellm_gateway_endpoint(self) -> bool:
        if is_default_local_litellm_gateway(self.api_base or getattr(self._cfg, "api_base", None)):
            return True
        return is_explicit_litellm_gateway(
            self.provider or getattr(self._cfg, "provider", None),
            custom_llm_provider=self._custom_llm_provider or getattr(self._cfg, "custom_llm_provider", None),
            topology_kind=self._vllm_topology_kind(),
        )

    def _vllm_topology_kind(self) -> str | None:
        topology = getattr(getattr(self._cfg, "vllm", None), "topology", None)
        if topology is None:
            return None
        return getattr(topology, "kind", None)

    @classmethod
    def _raise_classified_retry_failure(cls, exc: Exception) -> None:
        if cls._is_dependency_unavailable_error(exc):
            raise LiteLLMBackendError(str(exc), DEPENDENCY_UNAVAILABLE) from exc

    @staticmethod
    def _is_dependency_unavailable_error(exc: Exception) -> bool:
        status_code = status_code_from_error(exc)
        if status_code in {408, 429, 500, 502, 503, 504}:
            return True
        message = str(exc).strip().lower()
        if not message:
            return False
        dependency_markers = (
            "connection refused",
            "connection error",
            "failed to connect",
            "failed to establish",
            "max retries exceeded",
            "name resolution",
            "temporary failure in name resolution",
            "timed out",
            "timeout",
            "connect timeout",
            "read timeout",
            "endpoint unavailable",
            "service unavailable",
            "bad gateway",
            "gateway timeout",
            "network is unreachable",
        )
        return any(marker in message for marker in dependency_markers)

    @staticmethod
    def _format_response_json(raw_response: Any) -> str:
        try:
            return json.dumps(raw_response, ensure_ascii=True)
        except TypeError:
            return str(raw_response)
