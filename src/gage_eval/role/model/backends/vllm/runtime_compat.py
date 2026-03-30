"""Compatibility helpers for vLLM engine/runtime API differences."""

from __future__ import annotations

import inspect
from dataclasses import dataclass
from functools import lru_cache
from typing import Any


@dataclass(frozen=True)
class VLLMEngineRuntime:
    """Describe the installed vLLM engine API surface."""

    async_engine_args_cls: type[Any]
    async_llm_engine_cls: type[Any]
    engine_arg_names: frozenset[str]

    @property
    def engine_variant(self) -> str:
        """Return a coarse engine family identifier for diagnostics."""

        module_name = getattr(self.async_llm_engine_cls, "__module__", "")
        return "v1" if module_name.startswith("vllm.v1.") else "legacy"


def _resolve_parameter_names(target: Any) -> frozenset[str]:
    """Collect explicit parameter names from a callable signature."""

    try:
        signature = inspect.signature(target)
    except (TypeError, ValueError):
        return frozenset()

    names: set[str] = set()
    for parameter in signature.parameters.values():
        if parameter.name == "self":
            continue
        if parameter.kind in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD):
            continue
        names.add(parameter.name)
    return frozenset(names)


@lru_cache(maxsize=1)
def load_vllm_engine_runtime() -> VLLMEngineRuntime:
    """Import the installed vLLM engine classes and inspect their signatures."""

    from vllm.engine.arg_utils import AsyncEngineArgs  # type: ignore
    from vllm.engine.async_llm_engine import AsyncLLMEngine  # type: ignore

    return VLLMEngineRuntime(
        async_engine_args_cls=AsyncEngineArgs,
        async_llm_engine_cls=AsyncLLMEngine,
        engine_arg_names=_resolve_parameter_names(AsyncEngineArgs),
    )


def prepare_async_engine_kwargs(
    engine_kwargs: dict[str, Any],
    runtime: VLLMEngineRuntime,
) -> tuple[dict[str, Any], tuple[str, ...]]:
    """Filter engine kwargs to the installed AsyncEngineArgs signature.

    Args:
        engine_kwargs: Candidate kwargs assembled by the backend.
        runtime: Introspected vLLM runtime information.

    Returns:
        A tuple of filtered kwargs and unsupported keys that were dropped.
    """

    normalized = dict(engine_kwargs)
    supported_names = runtime.engine_arg_names

    if "enable_log_requests" in supported_names and "enable_log_requests" not in normalized:
        normalized["enable_log_requests"] = False
    elif "disable_log_requests" in supported_names and "disable_log_requests" not in normalized:
        normalized["disable_log_requests"] = True

    if "disable_log_stats" in supported_names and "disable_log_stats" not in normalized:
        normalized["disable_log_stats"] = True

    dropped = tuple(sorted(name for name in normalized if name not in supported_names))
    filtered = {name: value for name, value in normalized.items() if name in supported_names}
    return filtered, dropped
