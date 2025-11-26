"""Registry utilities for preprocessing functions."""

from __future__ import annotations

import importlib
import inspect
from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Callable, Dict, Tuple


@dataclass(frozen=True)
class PreprocessHandle:
    """Wrapped preprocessing callable with filtered kwargs support."""

    func: Callable[..., Any]
    accepted_kwargs: Tuple[str, ...]
    allow_extra_kwargs: bool = False

    def apply(self, sample: Dict[str, Any], **kwargs: Any) -> Any:
        if self.allow_extra_kwargs:
            return self.func(sample, **kwargs)
        filtered = {k: v for k, v in kwargs.items() if k in self.accepted_kwargs}
        return self.func(sample, **filtered)


@lru_cache(maxsize=64)
def resolve_preprocess(module_path: str) -> PreprocessHandle:
    """Resolve ``convert_sample_to_*`` from the given module path."""

    module = importlib.import_module(module_path)
    func = _pick_preprocess_func(module)
    sig = inspect.signature(func)
    params = list(sig.parameters.values())
    if not params:
        raise ValueError(f"Preprocess function '{func.__module__}.{func.__name__}' is missing parameters")
    accepted: list[str] = []
    allow_extra = False
    for param in params[1:]:  # skip the sample argument
        if param.kind == inspect.Parameter.VAR_KEYWORD:
            allow_extra = True
            break
        if param.kind in (inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY):
            accepted.append(param.name)
    return PreprocessHandle(func=func, accepted_kwargs=tuple(accepted), allow_extra_kwargs=allow_extra)


def _pick_preprocess_func(module) -> Callable[..., Any]:
    for name in ("convert_sample_to_inputs", "convert_sample_to_input_ids"):
        candidate = getattr(module, name, None)
        if callable(candidate):
            return candidate
    available = ", ".join(sorted(dir(module)))
    raise AttributeError(f"Module '{module.__name__}' does not define convert_sample_to_* (candidates: {available})")
