"""In-memory store for resolved model handles."""

from __future__ import annotations

from typing import Dict

from .base import ModelHandle

_MODEL_STORE: Dict[str, ModelHandle] = {}


def clear_model_store() -> None:
    _MODEL_STORE.clear()


def register_model_handle(handle: ModelHandle) -> None:
    _MODEL_STORE[handle.model_id] = handle


def resolve_model_handle(model_id: str) -> ModelHandle:
    try:
        return _MODEL_STORE[model_id]
    except KeyError as exc:  # pragma: no cover - defensive
        raise KeyError(f"Model '{model_id}' has not been materialized") from exc
