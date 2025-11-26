"""Decorators for wrapping runtime stages with observability hooks."""

from __future__ import annotations

import functools
import time
from typing import Any, Callable, Dict, Optional, TypeVar

from gage_eval.observability.config import ObservabilityConfig, get_observability_config
from gage_eval.observability.trace import ObservabilityTrace

F = TypeVar("F", bound=Callable[..., Any])


def observable_stage(
    stage: str,
    *,
    config: Optional[ObservabilityConfig] = None,
    payload_fn: Optional[Callable[..., Dict[str, Any]]] = None,
    sample_id_getter: Optional[Callable[..., Optional[str]]] = None,
    sample_idx_getter: Optional[Callable[..., Optional[int]]] = None,
) -> Callable[[F], F]:
    """Wrap a callable so that trace events are emitted when enabled."""

    def decorator(func: F) -> F:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            cfg = config or get_observability_config()
            trace = _resolve_trace(args, kwargs)
            if not cfg.enabled or trace is None:
                return func(*args, **kwargs)
            sample_id = sample_id_getter(*args, **kwargs) if sample_id_getter else kwargs.get("sample_id")
            sample_idx = sample_idx_getter(*args, **kwargs) if sample_idx_getter else kwargs.get("sample_idx")
            sampled = cfg.should_sample(stage, sample_idx=sample_idx, sample_id=sample_id)
            payload = payload_fn(*args, **kwargs) if payload_fn else {}
            payload = dict(payload or {})
            payload.setdefault("stage", stage)
            if sample_id:
                payload.setdefault("sample_id", sample_id)
            start = time.perf_counter()
            event = f"{stage}_start"
            start_emitted = False
            if sampled:
                trace.emit(event, payload=payload, sample_id=sample_id)
                start_emitted = True
            status = "success"
            try:
                return func(*args, **kwargs)
            except Exception:
                status = "error"
                if sample_id:
                    trace.force_log(sample_id)
                raise
            finally:
                should_emit = sampled or status == "error"
                if should_emit:
                    if not start_emitted:
                        trace.emit(event, payload=payload, sample_id=sample_id)
                        start_emitted = True
                    elapsed = time.perf_counter() - start
                    end_payload = dict(payload)
                    end_payload["elapsed_s"] = elapsed
                    end_payload["status"] = status
                    trace.emit(f"{stage}_end", payload=end_payload, sample_id=sample_id)
                    cfg.record_timing(stage, elapsed)

        return wrapper  # type: ignore[return-value]

    return decorator


def _resolve_trace(args: tuple[Any, ...], kwargs: Dict[str, Any]) -> Optional[ObservabilityTrace]:
    for value in kwargs.values():
        if isinstance(value, ObservabilityTrace):
            return value
    for value in args:
        if isinstance(value, ObservabilityTrace):
            return value
    return None
