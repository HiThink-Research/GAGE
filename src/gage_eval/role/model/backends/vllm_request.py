"""Deprecated shim for vLLM request helpers."""

from __future__ import annotations

import warnings

from gage_eval.role.model.backends.vllm import vllm_request as _vllm_request

warnings.warn(
    "gage_eval.role.model.backends.vllm_request is deprecated; "
    "use gage_eval.role.model.backends.vllm.vllm_request",
    DeprecationWarning,
    stacklevel=2,
)

from gage_eval.role.model.backends.vllm.vllm_request import *  # noqa: F403

__all__ = list(getattr(_vllm_request, "__all__", [])) or [
    name for name in dir(_vllm_request) if not name.startswith("_")
]
