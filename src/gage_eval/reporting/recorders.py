"""Recorder implementations inspired by OpenAI Evals."""

from __future__ import annotations

import contextlib
import json
import threading
import time
from contextvars import ContextVar
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

import requests
from loguru import logger
from gage_eval.registry import registry


@dataclass
class TraceEvent:
    """Structured event emitted by ObservabilityTrace."""

    run_id: str
    event_id: int
    event: str
    payload: Dict[str, Any]
    sample_id: Optional[str]
    created_at: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "run_id": self.run_id,
            "event_id": self.event_id,
            "event": self.event,
            "payload": self.payload,
            "sample_id": self.sample_id,
            "created_at": self.created_at,
        }


class RecorderBase:
    """Base recorder that batches events and flushes them on demand."""

    def __init__(
        self,
        run_id: str,
        *,
        created_by: str = "gage-eval",
        min_flush_events: int = 100,
        min_flush_seconds: float = 10.0,
    ) -> None:
        self.run_id = run_id
        self.created_by = created_by
        self._events: List[TraceEvent] = []
        self._written = 0
        self._lock = threading.Lock()
        self._sample_id: ContextVar[Optional[str]] = ContextVar("recorder_sample_id", default=None)
        self._min_flush_events = max(1, min_flush_events)
        self._min_flush_seconds = max(0.1, min_flush_seconds)
        self._last_flush_ts = time.time()

    @contextlib.contextmanager
    def use_sample(self, sample_id: Optional[str]):
        token = self._sample_id.set(sample_id)
        try:
            yield
        finally:
            self._sample_id.reset(token)

    def record_event(self, event: str, payload: Dict[str, Any], *, sample_id: Optional[str] = None) -> None:
        sample_ref = sample_id if sample_id is not None else self._sample_id.get()
        trace_event = TraceEvent(
            run_id=self.run_id,
            event_id=len(self._events),
            event=event,
            payload=payload,
            sample_id=sample_ref,
            created_at=time.time(),
        )
        with self._lock:
            self._events.append(trace_event)
            should_flush = (
                (len(self._events) - self._written) >= self._min_flush_events
                or time.time() - self._last_flush_ts >= self._min_flush_seconds
            )
        if should_flush:
            self.flush_events()

    def get_events(self) -> Sequence[TraceEvent]:
        with self._lock:
            return list(self._events)

    def flush_events(self) -> None:
        with self._lock:
            if self._written >= len(self._events):
                return
            events_to_write = self._events[self._written :]
            self._written = len(self._events)
            self._last_flush_ts = time.time()
        self._flush_events_internal(events_to_write)
        logger.debug(
            "Recorder '{}' flushed {} events (written={})",
            self.__class__.__name__,
            len(events_to_write),
            self._written,
        )

    def close(self) -> None:
        self.flush_events()

    def _flush_events_internal(self, events: Sequence[TraceEvent]) -> None:  # pragma: no cover - abstract
        raise NotImplementedError


@registry.asset(
    "reporting_sinks",
    "inmemory",
    desc="将事件保存在内存中（测试用途）",
    tags=("memory",),
)
class InMemoryRecorder(RecorderBase):
    """Recorder that keeps everything in memory (useful for tests)."""

    def __init__(self, run_id: str, **kwargs):
        super().__init__(run_id, **kwargs)
        self._buffer: List[TraceEvent] = []

    def _flush_events_internal(self, events: Sequence[TraceEvent]) -> None:
        self._buffer.extend(events)

    def buffered_events(self) -> List[Dict[str, Any]]:
        return [event.to_dict() for event in self._buffer]


@registry.asset(
    "reporting_sinks",
    "file",
    desc="将事件写入本地 JSONL 文件",
    tags=("file", "local"),
)
class FileRecorder(RecorderBase):
    """Recorder that appends events to a JSONL file."""

    def __init__(self, run_id: str, *, output_path: Path, **kwargs) -> None:
        super().__init__(run_id, **kwargs)
        self.output_path = output_path
        self.output_path.parent.mkdir(parents=True, exist_ok=True)

    def _flush_events_internal(self, events: Sequence[TraceEvent]) -> None:
        if not events:
            return
        with self.output_path.open("a", encoding="utf-8") as handle:
            for event in events:
                handle.write(json.dumps(event.to_dict(), ensure_ascii=False))
                handle.write("\n")
        logger.info("FileRecorder wrote {} events to {}", len(events), self.output_path)


@registry.asset(
    "reporting_sinks",
    "http",
    desc="推送事件到 HTTP 端点，支持回退",
    tags=("http", "remote"),
)
class HTTPRecorder(RecorderBase):
    """Recorder that ships events to an HTTP endpoint with optional fallback."""

    def __init__(
        self,
        run_id: str,
        *,
        url: str,
        batch_size: int = 50,
        timeout: float = 10.0,
        fail_threshold_pct: float = 5.0,
        fallback: Optional[RecorderBase] = None,
        **kwargs,
    ) -> None:
        super().__init__(run_id, **kwargs)
        self.url = url
        self.batch_size = max(1, batch_size)
        self.timeout = timeout
        self.fail_threshold_pct = max(0.0, fail_threshold_pct)
        self.fallback = fallback
        self._session = requests.Session()
        self._requests_sent = 0
        self._requests_failed = 0
        self._failover_triggered = False

    def _flush_events_internal(self, events: Sequence[TraceEvent]) -> None:
        if not events:
            return
        if self._failover_triggered:
            self._flush_via_fallback(events)
            return

        for chunk in _chunk(events, self.batch_size):
            payload = {"events": [event.to_dict() for event in chunk]}
            self._requests_sent += 1
            try:
                response = self._session.post(self.url, json=payload, timeout=self.timeout)
                response.raise_for_status()
            except requests.RequestException as exc:  # pragma: no cover - network branch
                self._requests_failed += 1
                logger.warning("HTTPRecorder error ({}). Falling back to local recorder.", exc)
                self._flush_via_fallback(chunk)
                self._check_failover()
            else:
                logger.debug("HTTPRecorder flushed {} events to {}", len(chunk), self.url)

    def _flush_via_fallback(self, events: Sequence[TraceEvent]) -> None:
        if self.fallback is None:
            logger.error("HTTPRecorder fallback not configured; dropping {} events", len(events))
            return
        self.fallback._flush_events_internal(events)

    def _check_failover(self) -> None:
        if self._requests_sent == 0:
            return
        failure_pct = (self._requests_failed / self._requests_sent) * 100
        if failure_pct >= self.fail_threshold_pct:
            self._failover_triggered = True
            logger.error(
                "HTTPRecorder failure rate {:.2f}% exceeded threshold {:.2f}%. Permanent failover to local recorder.",
                failure_pct,
                self.fail_threshold_pct,
            )


def _chunk(events: Sequence[TraceEvent], size: int) -> Iterable[Sequence[TraceEvent]]:
    for idx in range(0, len(events), size):
        yield events[idx : idx + size]
