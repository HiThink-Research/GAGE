"""Deprecated shim for backend helpers (moved to gage_eval.role.common.backend_utils)."""

from __future__ import annotations

import warnings

from gage_eval.role.common import backend_utils as _backend_utils

warnings.warn(
    "gage_eval.role.model.backends.shared_utils is deprecated; "
    "use gage_eval.role.common.backend_utils",
    DeprecationWarning,
    stacklevel=2,
)

from gage_eval.role.common.backend_utils import *  # noqa: F403
from gage_eval.role.model.backends.vllm.vllm_request import (  # noqa: F401
    build_engine_args,
    build_sampling_params,
    detect_vllm_version,
    ensure_spawn_start_method,
    resolve_sampling_class,
    resolve_vllm_mm_support,
)

__all__ = list(getattr(_backend_utils, "__all__", [])) + [
    "build_engine_args",
    "build_sampling_params",
    "detect_vllm_version",
    "ensure_spawn_start_method",
    "resolve_sampling_class",
    "resolve_vllm_mm_support",
]
