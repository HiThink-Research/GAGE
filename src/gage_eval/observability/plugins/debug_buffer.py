from __future__ import annotations

from collections import deque
from threading import Lock
from typing import Any

from gage_eval.observability.config import ObservabilityConfig


class DebugLogBuffer:
    """Optional in-memory log buffer for debug and tests."""

    def __init__(self) -> None:
        self._buffers: dict[str, deque[dict[str, Any]]] = {}
        self._lock = Lock()

    def record(self, stage: str, record: dict[str, Any], cfg: ObservabilityConfig) -> None:
        size = cfg.buffer_size_for(stage)
        if size <= 0:
            return
        with self._lock:
            buffer = self._buffers.get(stage)
            if buffer is None or buffer.maxlen != size:
                buffer = deque(maxlen=size)
                self._buffers[stage] = buffer
            buffer.append(record)

    def drain(self, stage: str | None = None) -> list[dict[str, Any]]:
        with self._lock:
            if stage:
                buf = self._buffers.pop(stage, None)
                return list(buf) if buf else []
            records: list[dict[str, Any]] = []
            for buf in self._buffers.values():
                records.extend(list(buf))
            self._buffers.clear()
            return records
