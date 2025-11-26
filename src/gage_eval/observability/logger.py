"""Structured logger that respects observability sampling settings."""

from __future__ import annotations

from collections import deque
from threading import Lock
from typing import Any, Deque, Dict, List, Optional

from loguru import logger as base_logger

from gage_eval.observability.config import ObservabilityConfig, get_observability_config
from gage_eval.observability.log_sink import is_log_sink_active
from gage_eval.observability.trace import ObservabilityTrace


class ObservableLogger:
    """Wraps loguru so stage/sample aware logging can be sampled."""

    def __init__(self, *, config: Optional[ObservabilityConfig] = None) -> None:
        self._config_override = config
        self._buffers: Dict[str, Deque[Dict[str, Any]]] = {}
        self._lock = Lock()

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
        self._buffer_record(stage, record, cfg)

        if emit_to_logger:
            bound = base_logger.bind(stage=stage, sample_id=sample_id, sample_idx=sample_idx)
            (bound or base_logger).log(level, templated, *args, **fmt_kwargs)
        if trace and cfg.enabled and not is_log_sink_active():
            trace.emit("log", {"stage": stage, "message": formatted, "level": level}, sample_id=sample_id)

    def drain_buffer(self, stage: Optional[str] = None) -> List[Dict[str, Any]]:
        with self._lock:
            if stage:
                buf = self._buffers.pop(stage, None)
                return list(buf) if buf else []
            records: List[Dict[str, Any]] = []
            for buf in self._buffers.values():
                records.extend(list(buf))
            self._buffers.clear()
            return records

    def _buffer_record(self, stage: str, record: Dict[str, Any], cfg: ObservabilityConfig) -> None:
        size = cfg.buffer_size_for(stage)
        if size <= 0:
            return
        with self._lock:
            buffer = self._buffers.get(stage)
            if buffer is None or buffer.maxlen != size:
                buffer = deque(maxlen=size)
                self._buffers[stage] = buffer
            buffer.append(record)

    def _current_config(self) -> ObservabilityConfig:
        return self._config_override or get_observability_config()


def _safe_format(message: str, *args, **kwargs) -> str:
    try:
        if not args and not kwargs:
            return message
        return message.format(*args, **kwargs)
    except Exception:
        return message
