"""Structured logger that respects observability sampling settings."""

from __future__ import annotations

from typing import Any, Dict, Optional

from loguru import logger as base_logger

from gage_eval.observability.config import ObservabilityConfig, get_observability_config
from gage_eval.observability.log_sink import is_log_sink_active, register_observable_trace
from gage_eval.observability.plugins import DebugLogBuffer
from gage_eval.observability.trace import ObservabilityTrace


class ObservableLogger:
    """Wraps loguru so stage/sample aware logging can be sampled."""

    def __init__(
        self,
        *,
        config: Optional[ObservabilityConfig] = None,
        debug_buffer: DebugLogBuffer | None = None,
    ) -> None:
        self._config_override = config
        self._debug_buffer = debug_buffer

    @classmethod
    def with_debug_buffer(
        cls,
        *,
        config: Optional[ObservabilityConfig] = None,
    ) -> "ObservableLogger":
        return cls(config=config, debug_buffer=DebugLogBuffer())

    @property
    def debug_buffer(self) -> DebugLogBuffer | None:
        return self._debug_buffer

    def configure(self, config: ObservabilityConfig) -> None:
        self._config_override = config

    def debug(self, stage: str, message: str, *args, **kwargs) -> None:
        self.log("DEBUG", stage, message, *args, **kwargs)

    def info(self, stage: str, message: str, *args, **kwargs) -> None:
        self.log("INFO", stage, message, *args, **kwargs)

    def warning(self, stage: str, message: str, *args, **kwargs) -> None:
        self.log("WARNING", stage, message, *args, **kwargs)

    def error(self, stage: str, message: str, *args, **kwargs) -> None:
        self.log("ERROR", stage, message, *args, **kwargs)

    def log(
        self,
        level: str,
        stage: str,
        message: str,
        *args,
        sample_id: Optional[str] = None,
        sample_idx: Optional[int] = None,
        trace: Optional[ObservabilityTrace] = None,
        extra: Optional[Dict[str, Any]] = None,
        **fmt_kwargs,
    ) -> None:
        cfg = self._current_config()
        emit_to_logger = True
        if cfg.enabled:
            emit_to_logger = cfg.should_sample(stage, sample_idx=sample_idx, sample_id=sample_id)

        prefix = "[{}|sample_id={}|sample_idx={}] ".format(
            stage,
            sample_id or "-",
            sample_idx if sample_idx is not None else "-",
        )
        templated = prefix + message
        formatted = _safe_format(templated, *args, **fmt_kwargs)
        record = {
            "stage": stage,
            "level": level,
            "sample_id": sample_id,
            "sample_idx": sample_idx,
            "message": formatted,
        }
        if extra:
            record["extra"] = extra
        if self._debug_buffer is not None:
            self._debug_buffer.record(stage, record, cfg)

        if emit_to_logger:
            bind_kwargs = {
                "stage": stage,
                "sample_id": sample_id,
                "sample_idx": sample_idx,
            }
            if trace is not None:
                bind_kwargs["trace_run_id"] = trace.run_id
                if is_log_sink_active() and trace.accepts_new_events():
                    register_observable_trace(trace)
            bound = base_logger.bind(**bind_kwargs)
            (bound or base_logger).log(level, templated, *args, **fmt_kwargs)

    def _current_config(self) -> ObservabilityConfig:
        return self._config_override or get_observability_config()


def _safe_format(message: str, *args, **kwargs) -> str:
    try:
        if not args and not kwargs:
            return message
        return message.format(*args, **kwargs)
    except Exception:
        return message
