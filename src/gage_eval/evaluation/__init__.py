"""Evaluation helpers exposed to callers."""

from __future__ import annotations

from typing import Any

__all__ = ["build_runtime"]


def build_runtime(*args: Any, **kwargs: Any):
    """Lazily import build_runtime to avoid package import cycles."""

    from gage_eval.evaluation.runtime_builder import build_runtime as _build_runtime

    return _build_runtime(*args, **kwargs)
