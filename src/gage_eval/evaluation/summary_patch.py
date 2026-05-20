"""Deferred summary patch buffering for EvalCache."""

from __future__ import annotations

import threading
from typing import Any, Mapping


class SummaryPatchBuffer:
    """Thread-safe buffer for fields that must be merged into summary.json."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._payload: dict[str, Any] = {}

    def add(self, payload: Mapping[str, Any]) -> None:
        with self._lock:
            self._payload.update(dict(payload))

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            return dict(self._payload)

    def drain(self) -> dict[str, Any]:
        with self._lock:
            payload = dict(self._payload)
            self._payload.clear()
            return payload


__all__ = ["SummaryPatchBuffer"]
