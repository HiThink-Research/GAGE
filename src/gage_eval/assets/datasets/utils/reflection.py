"""Reflection helpers for dynamic callable/kwargs resolution."""

from __future__ import annotations

import ast
import json
from typing import Any, Dict, Optional


def resolve_callable(ref: Optional[Any]):
    """Resolve dotted-path references used by doc_to_* hooks."""

    if ref is None:
        return None
    if callable(ref):
        return ref
    if isinstance(ref, str):
        module_path, attr = _split_ref(ref)
        module = __import__(module_path, fromlist=[attr])
        return getattr(module, attr)
    raise TypeError(f"Unsupported callable reference: {ref!r}")


def coerce_kwargs(raw: Any) -> Dict[str, Any]:
    """Parse kwargs from dict/JSON/literal string inputs."""

    if raw is None:
        return {}
    if isinstance(raw, dict):
        return dict(raw)
    if isinstance(raw, str):
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            try:
                value = ast.literal_eval(raw)
            except (ValueError, SyntaxError) as exc:  # pragma: no cover - defensive
                raise ValueError(f"Unable to parse preprocess_kwargs: {raw}") from exc
            else:
                if isinstance(value, dict):
                    return dict(value)
                raise ValueError(f"preprocess_kwargs string must evaluate to dict: {raw}")
    raise TypeError(f"Unsupported preprocess_kwargs type: {type(raw)!r}")


def _split_ref(ref: str) -> tuple[str, str]:
    if ":" in ref:
        module_path, attr = ref.rsplit(":", 1)
    elif "." in ref:
        module_path, attr = ref.rsplit(".", 1)
    else:
        raise ValueError(f"Callable reference must include module path: {ref}")
    return module_path, attr


__all__ = ["resolve_callable", "coerce_kwargs"]
