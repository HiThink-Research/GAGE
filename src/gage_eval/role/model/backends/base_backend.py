"""Generic engine adaptor base extracted from llm-eval."""

from __future__ import annotations

import time
from typing import Any, Dict, Optional

from loguru import logger
from gage_eval.registry.utils import ensure_async, run_sync
from gage_eval.role.model.reasoning import (
    extract_reasoning_content,
    resolve_thinking_kwargs,
    strip_thinking_tags,
)


class Backend:
    """Async-friendly backend interface consumed by role adapters."""

    kind = "backends"

    def __init__(self, config: Dict[str, Any]) -> None:
        self.config = dict(config)
        # NOTE: Execution mode: `native` means local engine, `http` means remote API.
        self.execution_mode: str = self.config.get("execution_mode", "native")

    async def ainvoke(self, payload: Dict[str, Any]) -> Dict[str, Any]:  # pragma: no cover - abstract
        raise NotImplementedError

    def invoke(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Synchronous entry point retained for backwards compatibility."""

        return run_sync(self.ainvoke(payload))

    def __call__(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        return self.invoke(payload)


def build_backend_error_result(
    exc: Exception,
    *,
    backend_name: str,
    status: Optional[int] = None,
) -> Dict[str, Any]:
    """Builds a normalized backend error payload from an exception.

    Args:
        exc: Original exception raised by the backend.
        backend_name: Human-readable backend name for diagnostics.
        status: Optional explicit status code override.

    Returns:
        A normalized backend error payload consumed by role adapters and steps.
    """

    resolved_status = status if status is not None else getattr(exc, "status_code", None)
    if resolved_status is None:
        response = getattr(exc, "response", None)
        resolved_status = getattr(response, "status_code", None)
    return {
        "error": str(exc),
        "status": resolved_status,
        "error_type": type(exc).__name__,
        "backend": backend_name,
    }


class EngineBackend(Backend):
    """Base class mirroring the llm-eval EngineBackend contract.

    Provides centralized thinking/reasoning support so that all backends
    inherit it automatically.
    """

    def __init__(self, config: Dict[str, Any]) -> None:
        super().__init__(config)
        logger.info("Initializing backend {}", self.__class__.__name__)

        # STEP 1: Parse thinking/reasoning config from the raw config dict.
        # These fields can come from top-level config or nested generation_parameters.
        gen_params = config.get("generation_parameters") or {}
        if isinstance(gen_params, dict):
            gen_thinking = gen_params.get("thinking_mode")
            gen_effort = gen_params.get("reasoning_effort")
            gen_chat_kwargs = gen_params.get("chat_template_kwargs")
        else:
            gen_thinking = getattr(gen_params, "thinking_mode", None)
            gen_effort = getattr(gen_params, "reasoning_effort", None)
            gen_chat_kwargs = getattr(gen_params, "chat_template_kwargs", None)

        self._thinking_mode: Optional[str] = config.get("thinking_mode") or gen_thinking
        self._reasoning_effort: Optional[str] = config.get("reasoning_effort") or gen_effort
        self._thinking_chat_template_kwargs: Optional[Dict[str, Any]] = (
            config.get("chat_template_kwargs") or gen_chat_kwargs
        )
        # Pre-resolve the merged thinking kwargs for subclasses to use
        self._resolved_thinking_kwargs: Dict[str, Any] = resolve_thinking_kwargs(
            self._thinking_mode, self._thinking_chat_template_kwargs
        )

        self.model = self.load_model(config)

    def load_model(self, config: Dict[str, Any]):  # pragma: no cover
        raise NotImplementedError

    def prepare_inputs(self, payload: Dict[str, Any]) -> Dict[str, Any]:  # pragma: no cover
        return payload

    def generate(self, inputs: Dict[str, Any]) -> Dict[str, Any]:  # pragma: no cover
        raise NotImplementedError

    async def ainvoke(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        inputs = self.prepare_inputs(payload)
        start = time.time()
        generate = ensure_async(self.generate)
        result = await generate(inputs)
        result.setdefault("latency_ms", (time.time() - start) * 1000)

        # STEP 2: Enrich result with reasoning content (centralized for all backends)
        self._enrich_result_with_reasoning(result)

        logger.debug(
            "Backend {} finished request latency={:.2f}ms",
            self.__class__.__name__,
            result["latency_ms"],
        )
        return result

    # ------------------------------------------------------------------ #
    # Thinking / Reasoning helpers (shared by all backends)               #
    # ------------------------------------------------------------------ #
    def _enrich_result_with_reasoning(self, result: Dict[str, Any]) -> None:
        """Post-processes a generate() result to extract reasoning content.

        Performs two complementary extractions:
        1. Structured: Extracts ``reasoning_content`` from the raw_response
           message object (e.g. DeepSeek Reasoner API returns it as a field).
        2. Tag-based: Strips ``<think>...</think>`` tags from the answer text
           (e.g. Qwen3 in thinking mode wraps chain-of-thought in tags).

        Args:
            result: The mutable result dict from ``generate()``.
        """
        # Skip if result already contains reasoning_content (backend set it explicitly)
        if result.get("reasoning_content") is not None:
            return

        # STEP 1: Try structured extraction from raw_response
        raw = result.get("raw_response")
        reasoning = self._extract_reasoning_from_raw(raw)

        # STEP 2: Try tag-based extraction from answer text
        answer = result.get("answer") or ""
        if answer:
            clean_answer, tag_thinking = strip_thinking_tags(answer)
            if tag_thinking:
                result["answer"] = clean_answer
                # Combine structured and tag-based if both present
                if reasoning:
                    reasoning = reasoning + "\n" + tag_thinking
                else:
                    reasoning = tag_thinking

        if reasoning:
            result["reasoning_content"] = reasoning

    def _extract_reasoning_from_raw(self, raw_response: Any) -> Optional[str]:
        """Extracts reasoning_content from a raw response (dict or object).

        Handles both OpenAI SDK objects and plain dicts.
        """
        if raw_response is None:
            return None

        # Dict-based raw response (most common after model_dump())
        if isinstance(raw_response, dict):
            choices = raw_response.get("choices") or []
            if choices:
                message = choices[0].get("message") or choices[0].get("delta") or {}
                return extract_reasoning_content(message)
            return None

        # Object-based raw response (before serialization)
        choices = getattr(raw_response, "choices", None)
        if choices:
            message = getattr(choices[0], "message", None) or getattr(choices[0], "delta", None)
            return extract_reasoning_content(message)

        return None

    def get_thinking_config(self) -> Dict[str, Any]:
        """Returns the resolved thinking configuration for subclasses.

        Subclasses can use this to inject thinking-related parameters
        into their API calls (e.g. ``enable_thinking``, ``reasoning_effort``).

        Returns:
            A dict with resolved thinking kwargs. May include
            ``enable_thinking`` (bool) and any additional template kwargs.
        """
        config = dict(self._resolved_thinking_kwargs)
        if self._reasoning_effort:
            config.setdefault("reasoning_effort", self._reasoning_effort)
        return config
