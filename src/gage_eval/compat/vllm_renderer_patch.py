"""Compatibility patches for vLLM renderer multimodal initialization gaps."""

from __future__ import annotations

import contextlib
import copy
import importlib
import logging
from typing import Any, Iterator, Optional

LOGGER = logging.getLogger(__name__)


class _NullMMTimingRegistry:
    """Compatibility shim for vLLM builds missing timing registry initialization."""

    def get(self, _request_id: str):
        return None


class _FallbackMMCounter:
    """Minimal AtomicCounter replacement for vLLM layouts without the utility export."""

    def __init__(self) -> None:
        self._value = 0

    def inc(self, step: int) -> int:
        self._value += step
        return self._value


def _iter_renderer_classes() -> Iterator[type[Any]]:
    """Yield known vLLM renderer classes that need the compatibility patch."""

    seen: set[int] = set()
    for module_name, class_name in (
        ("vllm.renderers.base", "BaseRenderer"),
        ("vllm.renderers.hf", "HfRenderer"),
        ("vllm.renderers.hf_renderer", "HfRenderer"),
    ):
        try:
            module = importlib.import_module(module_name)
        except Exception:
            continue
        renderer_cls = getattr(module, class_name, None)
        if renderer_cls is None:
            continue
        marker = id(renderer_cls)
        if marker in seen:
            continue
        seen.add(marker)
        yield renderer_cls


def _resolve_api_process_rank(renderer: Any) -> int:
    """Best-effort resolution for renderer API process rank."""

    config = getattr(renderer, "config", None)
    parallel_config = getattr(config, "parallel_config", None)
    for source in (renderer, config, parallel_config):
        if source is None:
            continue
        for attr in ("api_process_rank", "_api_process_rank", "rank"):
            value = getattr(source, attr, None)
            if value is None:
                continue
            try:
                return int(value)
            except (TypeError, ValueError):
                continue
    return 0


def _ensure_renderer_counter(renderer: Any) -> None:
    """Ensure the renderer has a request counter compatible with vLLM MM paths."""

    if getattr(renderer, "_mm_req_counter", None) is not None:
        return
    try:
        from vllm.utils.counter import AtomicCounter  # type: ignore
    except Exception:
        try:
            from vllm.utils import AtomicCounter  # type: ignore
        except Exception:
            AtomicCounter = _FallbackMMCounter  # type: ignore
    renderer._mm_req_counter = AtomicCounter()


def _ensure_renderer_timing_registry(renderer: Any) -> None:
    """Ensure the renderer exposes a timing registry expected by MM code paths."""

    if getattr(renderer, "_mm_timing_registry", None) is not None:
        return
    observability_config = getattr(getattr(renderer, "config", None), "observability_config", None)
    try:
        from vllm.multimodal.registry import MultiModalTimingRegistry  # type: ignore
    except Exception:
        renderer._mm_timing_registry = _NullMMTimingRegistry()
        return
    renderer._mm_timing_registry = MultiModalTimingRegistry(observability_config)


def _ensure_renderer_mm_processor(renderer: Any) -> None:
    """Lazily construct a multimodal processor when vLLM skipped renderer MM init."""

    if getattr(renderer, "mm_processor", None) is not None:
        return

    config = getattr(renderer, "config", None)
    model_config = getattr(renderer, "model_config", None) or getattr(config, "model_config", None)
    if config is None or model_config is None:
        return

    try:
        from vllm.multimodal import MULTIMODAL_REGISTRY  # type: ignore
    except Exception as exc:
        LOGGER.debug("vllm_renderer_patch missing MULTIMODAL_REGISTRY for %s: %s", type(renderer).__name__, exc)
        return

    try:
        mm_processor_cache = MULTIMODAL_REGISTRY.processor_cache_from_config(config)
    except Exception:
        mm_processor_cache = None

    tokenizer = getattr(renderer, "tokenizer", None)
    try:
        mm_tokenizer = copy.deepcopy(tokenizer) if tokenizer is not None else None
    except Exception:
        mm_tokenizer = tokenizer

    try:
        from vllm.utils.torch_utils import set_default_torch_num_threads  # type: ignore
    except Exception:
        thread_context = contextlib.nullcontext()
    else:
        thread_context = set_default_torch_num_threads()

    try:
        with thread_context:
            renderer.mm_processor = MULTIMODAL_REGISTRY.create_processor(
                model_config,
                tokenizer=mm_tokenizer,
                cache=mm_processor_cache,
            )
    except Exception as exc:
        LOGGER.debug(
            "vllm_renderer_patch failed to initialize mm_processor for %s: %s",
            type(renderer).__name__,
            exc,
        )
        return

    if getattr(renderer, "_mm_cache_stats", None) is None:
        try:
            from vllm.v1.metrics.stats import MultiModalCacheStats  # type: ignore
        except Exception:
            renderer._mm_cache_stats = None
        else:
            renderer._mm_cache_stats = MultiModalCacheStats()


def ensure_vllm_renderer_mm_state(renderer: Any, *, require_mm_processor: bool = False) -> None:
    """Backfill MM renderer state for vLLM builds that skip initialization.

    Args:
        renderer: The vLLM renderer instance.
        require_mm_processor: Whether to also lazily construct `mm_processor`.
    """

    if renderer is None:
        return

    if getattr(renderer, "api_process_rank", None) is None:
        renderer.api_process_rank = _resolve_api_process_rank(renderer)

    if getattr(renderer, "model_config", None) is None:
        config = getattr(renderer, "config", None)
        model_config = getattr(config, "model_config", None)
        if model_config is not None:
            renderer.model_config = model_config

    _ensure_renderer_counter(renderer)
    _ensure_renderer_timing_registry(renderer)
    if require_mm_processor:
        _ensure_renderer_mm_processor(renderer)


def _patch_renderer_method(
    renderer_cls: type[Any],
    method_name: str,
    *,
    marker_attr: str,
    require_mm_processor: bool,
    ensure_after: bool = False,
) -> None:
    """Wrap a renderer method so missing MM state is backfilled before execution."""

    original = getattr(renderer_cls, method_name, None)
    if not callable(original) or getattr(original, marker_attr, False):
        return

    def _wrapped(self, *args, **kwargs):
        if ensure_after:
            result = original(self, *args, **kwargs)
            ensure_vllm_renderer_mm_state(self, require_mm_processor=require_mm_processor)
            return result
        ensure_vllm_renderer_mm_state(self, require_mm_processor=require_mm_processor)
        return original(self, *args, **kwargs)

    setattr(_wrapped, marker_attr, True)
    setattr(renderer_cls, method_name, _wrapped)


def install_vllm_renderer_compat_patches() -> None:
    """Install vLLM renderer patches for missing multimodal state."""

    for renderer_cls in _iter_renderer_classes():
        _patch_renderer_method(
            renderer_cls,
            "__init__",
            marker_attr="_gage_mm_state_init_patch",
            require_mm_processor=False,
            ensure_after=True,
        )
        _patch_renderer_method(
            renderer_cls,
            "_process_multimodal",
            marker_attr="_gage_mm_state_process_patch",
            require_mm_processor=True,
        )


def _iter_renderer_instances(engine: Any) -> Iterator[Any]:
    """Yield renderer-like objects reachable from a vLLM engine wrapper."""

    pending = [engine]
    visited: set[int] = set()
    while pending:
        current = pending.pop()
        if current is None:
            continue
        marker = id(current)
        if marker in visited:
            continue
        visited.add(marker)

        if callable(getattr(current, "_process_multimodal", None)):
            yield current

        for attr in (
            "renderer",
            "input_preprocessor",
            "preprocessor",
            "input_processor",
            "engine",
            "llm_engine",
        ):
            child = getattr(current, attr, None)
            if child is not None:
                pending.append(child)


def detect_vllm_engine_multimodal_support(engine: Any) -> Optional[bool]:
    """Probe whether a live vLLM engine can accept multimodal requests.

    Returns:
        `True` when a renderer exposes a multimodal processor, `False` when the
        renderer explicitly reports a text-only model, and `None` when the
        engine structure cannot be inspected reliably.
    """

    saw_renderer = False
    saw_text_only = False
    for renderer in _iter_renderer_instances(engine):
        saw_renderer = True
        ensure_vllm_renderer_mm_state(renderer, require_mm_processor=False)

        if getattr(renderer, "mm_processor", None) is not None:
            return True

        getter = getattr(renderer, "get_mm_processor", None)
        if not callable(getter):
            continue

        try:
            mm_processor = getter()
        except ValueError as exc:
            if "text-only model" in str(exc).lower():
                saw_text_only = True
            else:
                LOGGER.debug(
                    "vllm_renderer_patch get_mm_processor probe failed for %s: %s",
                    type(renderer).__name__,
                    exc,
                )
            continue
        except Exception as exc:
            LOGGER.debug(
                "vllm_renderer_patch get_mm_processor probe failed for %s: %s",
                type(renderer).__name__,
                exc,
            )
            continue

        if mm_processor is not None:
            return True

    if saw_text_only:
        return False
    if saw_renderer:
        return None
    return None


def prime_vllm_engine_renderer_state(engine: Any, *, require_mm_processor: bool = False) -> None:
    """Prime renderer instances already attached to a live vLLM engine.

    Args:
        engine: The vLLM engine or a nested engine wrapper.
        require_mm_processor: Whether renderer MM processors should be constructed eagerly.
    """

    for renderer in _iter_renderer_instances(engine):
        ensure_vllm_renderer_mm_state(renderer, require_mm_processor=require_mm_processor)
