"""Materialize ModelSpec entries via model hubs."""

from __future__ import annotations

from typing import Dict

from gage_eval.config.pipeline_config import ModelSpec

from .base import ModelHandle
from .store import register_model_handle
from .hubs import HuggingFaceModelHub, LocalModelHub, ModelScopeModelHub
from .base import ModelHub

_HUBS = {
    "huggingface": HuggingFaceModelHub,
    "hf": HuggingFaceModelHub,
    "local": LocalModelHub,
    "modelscope": ModelScopeModelHub,
}


def resolve_model(spec: ModelSpec) -> ModelHandle:
    hub_name = spec.hub or spec.source or "huggingface"
    hub_cls: type[ModelHub]
    try:
        hub_cls = _HUBS[hub_name]
    except KeyError as exc:  # pragma: no cover - defensive
        raise KeyError(f"Unknown model hub '{hub_name}' for model '{spec.model_id}'") from exc
    hub_args: Dict = spec.hub_params
    hub = hub_cls(spec, hub_args=hub_args)
    handle = hub.resolve()
    register_model_handle(handle)
    return handle
