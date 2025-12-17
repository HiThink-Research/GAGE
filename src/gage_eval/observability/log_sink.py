"""Loguru sink that forwards logs into ObservabilityTrace."""

from __future__ import annotations

import atexit
import os
import threading
import time
from collections import deque
from typing import Any, Deque, Dict, Optional, Tuple, TYPE_CHECKING

from loguru import logger

from gage_eval.observability.config import get_observability_config

if TYPE_CHECKING:  # pragma: no cover
    from gage_eval.observability.trace import ObservabilityTrace

_GLOBAL_SINK: Optional["ObservabilityLogSink"] = None
_GLOBAL_SINK_ID: Optional[int] = None
_SINK_LOCK = threading.Lock()


class ObservabilityLogSink:
    """Background sink that batches log entries and emits them via ObservabilityTrace."""

    def __init__(
        self,
        trace: "ObservabilityTrace",
        *,
        max_queue_size: int = 1024,
        flush_interval_s: float = 0.25,
    ) -> None:
        self._trace = trace
        self._queue: Deque[Tuple[Dict[str, Any], Optional[str]]] = deque()
        self._max_queue = max(1, max_queue_size)
        self._flush_interval = max(0.05, flush_interval_s)
        self._cv = threading.Condition()
        self._closed = False
        self._dropped = 0
        self._backpressure_notified = False
        self._worker = threading.Thread(target=self._worker_loop, name="ObservableLogSink", daemon=True)
        self._worker.start()
        atexit.register(self.close)

    def set_trace(self, trace: "ObservabilityTrace") -> None:
        with self._cv:
            self._trace = trace

    def __call__(self, message) -> None:
        cfg = get_observability_config()
        if not cfg.enabled:
            return
        record = message.record
        extra = record.get("extra", {})
        stage = extra.get("stage") or record.get("name") or "log"
        sample_id = extra.get("sample_id")
        sample_idx = extra.get("sample_idx")
        if not cfg.should_sample(stage, sample_idx=sample_idx, sample_id=sample_id):
            return
        level = record.get("level")
        level_name = None
        if isinstance(level, dict):
            level_name = level.get("name")
        elif hasattr(level, "name"):
            level_name = level.name
        payload = {
            "stage": stage,
            "message": record.get("message"),
            "level": level_name,
            "time": record.get("time").isoformat() if record.get("time") else None,
            "file": (record.get("file") or {}).get("name") if isinstance(record.get("file"), dict) else None,
            "function": record.get("function"),
            "line": record.get("line"),
        }
        with self._cv:
            if self._closed:
                return
            if len(self._queue) >= self._max_queue:
                self._dropped += 1
                if not self._backpressure_notified:
                    self._emit_backpressure_locked()
                    self._backpressure_notified = True
                return
            self._queue.append((payload, sample_id))
            self._cv.notify()

    def close(self) -> None:
        with self._cv:
            if self._closed:
                return
            self._closed = True
            self._cv.notify_all()
        self._worker.join(timeout=1.0)

    def _worker_loop(self) -> None:
        while True:
            with self._cv:
                if not self._queue and not self._closed:
                    self._cv.wait(timeout=self._flush_interval)
                if not self._queue and self._closed:
                    return
                if not self._queue:
                    continue
                payload, sample_id = self._queue.popleft()
                if not self._queue:
                    self._backpressure_notified = False
            try:
                trace = self._trace
                trace.emit("log", payload, sample_id=sample_id)
            except Exception:
                logger.exception("ObservableLogSink failed to emit log event")

    def _emit_backpressure_locked(self) -> None:
        trace = self._trace
        dropped = self._dropped
        self._dropped = 0
        try:
            trace.emit(
                "log_backpressure",
                {"dropped": dropped, "queue_size": self._max_queue},
            )
        except Exception:
            logger.exception("ObservableLogSink failed to emit backpressure event")


def configure_observable_log_sink(trace: "ObservabilityTrace") -> None:
    """Install or update the global log sink to forward loguru logs to ObservabilityTrace."""

    if not _log_sink_enabled():
        return
    global _GLOBAL_SINK, _GLOBAL_SINK_ID
    with _SINK_LOCK:
        if _GLOBAL_SINK is None:
            sink = ObservabilityLogSink(trace)
            sink_id = logger.add(sink, level=os.environ.get("GAGE_EVAL_LOG_SINK_LEVEL", "INFO"))
            _GLOBAL_SINK = sink
            _GLOBAL_SINK_ID = sink_id
        else:
            _GLOBAL_SINK.set_trace(trace)


def _log_sink_enabled() -> bool:
    value = os.environ.get("GAGE_EVAL_ENABLE_LOG_SINK")
    if value is None:
        return True
    return value.lower() in {"1", "true", "yes", "on"}


def is_log_sink_active() -> bool:
    with _SINK_LOCK:
        return _GLOBAL_SINK is not None
