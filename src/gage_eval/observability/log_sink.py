"""Loguru sink that forwards logs into ObservabilityTrace."""

from __future__ import annotations

import atexit
import os
import threading
import time
import weakref
from collections import deque
from dataclasses import dataclass
from typing import Any, Deque, Dict, Optional, TYPE_CHECKING

from loguru import logger

from gage_eval.observability.config import get_observability_config

if TYPE_CHECKING:  # pragma: no cover
    from gage_eval.observability.trace import ObservabilityTrace


@dataclass(slots=True)
class _QueuedLogRecord:
    run_id: str
    payload: Dict[str, Any]
    sample_id: Optional[str]


@dataclass(slots=True)
class _TraceBinding:
    trace_ref: "weakref.ReferenceType[ObservabilityTrace]"
    registered_at: float
    last_seen_at: float
    closing: bool = False


@dataclass(frozen=True, slots=True)
class LogSinkDrainResult:
    remaining_queue_size: int
    dropped_on_close: int
    closed_cleanly: bool


_GLOBAL_SINK: Optional["ObservabilityLogSink"] = None
_GLOBAL_SINK_ID: Optional[int] = None
_SINK_LOCK = threading.Lock()


class ObservabilityLogSink:
    """Background sink that batches log entries and emits them via ObservabilityTrace."""

    def __init__(
        self,
        *,
        max_queue_size: Optional[int] = None,
        flush_interval_s: Optional[float] = None,
        batch_size: Optional[int] = None,
        zombie_route_ttl_s: Optional[float] = None,
        route_sweep_interval_s: Optional[float] = None,
    ) -> None:
        self._queue: Deque[_QueuedLogRecord] = deque()
        self._max_queue = max(1, max_queue_size or _env_int("GAGE_EVAL_LOG_SINK_MAX_QUEUE", 1024))
        self._flush_interval = max(0.05, flush_interval_s or _env_float("GAGE_EVAL_LOG_SINK_FLUSH_INTERVAL_S", 0.25))
        self._batch_size = max(1, batch_size or _env_int("GAGE_EVAL_LOG_SINK_BATCH_SIZE", 64))
        self._zombie_route_ttl_s = max(
            1.0,
            zombie_route_ttl_s or _env_float("GAGE_EVAL_LOG_SINK_ZOMBIE_ROUTE_TTL_S", 300.0),
        )
        self._route_sweep_interval_s = max(
            1.0,
            route_sweep_interval_s or _env_float("GAGE_EVAL_LOG_SINK_ROUTE_SWEEP_INTERVAL_S", 30.0),
        )
        self._routes: Dict[str, _TraceBinding] = {}
        self._pending_by_run: Dict[str, int] = {}
        self._dropped_by_run: Dict[str, int] = {}
        self._backpressure_notified_runs: set[str] = set()
        self._draining_runs: set[str] = set()
        self._orphaned_pending_since: Dict[str, float] = {}
        self._last_sweep_at = time.monotonic()
        self._cv = threading.Condition()
        self._closed = False
        self._worker = threading.Thread(target=self._worker_loop, name="ObservableLogSink", daemon=True)
        self._worker.start()
        atexit.register(self.close)

    @property
    def closed(self) -> bool:
        with self._cv:
            return self._closed

    def register_trace(self, trace: "ObservabilityTrace") -> None:
        if not trace.accepts_new_events():
            return
        now = time.monotonic()
        with self._cv:
            self._routes[trace.run_id] = _TraceBinding(
                trace_ref=weakref.ref(trace),
                registered_at=now,
                last_seen_at=now,
                closing=False,
            )
            self._draining_runs.discard(trace.run_id)
            self._orphaned_pending_since.pop(trace.run_id, None)
            self._maybe_reap_locked(now)
            self._cv.notify_all()

    def unregister_trace(self, run_id: str) -> None:
        with self._cv:
            self._cleanup_route_locked(run_id)
            self._cv.notify_all()

    def flush_run(
        self,
        run_id: str,
        *,
        timeout_s: float,
        close_mode: str,
    ) -> LogSinkDrainResult:
        deadline = time.monotonic() + max(0.0, timeout_s)
        dropped_on_close = 0
        with self._cv:
            binding = self._routes.get(run_id)
            if binding is not None:
                binding.closing = True
                binding.last_seen_at = time.monotonic()
            self._draining_runs.add(run_id)
            self._cv.notify_all()

            if close_mode == "best_effort":
                dropped_on_close += self._drop_queued_entries_locked(run_id)
                remaining = self._pending_by_run.get(run_id, 0)
                self._draining_runs.discard(run_id)
                return LogSinkDrainResult(
                    remaining_queue_size=remaining,
                    dropped_on_close=dropped_on_close,
                    closed_cleanly=remaining == 0,
                )

            while self._pending_by_run.get(run_id, 0) > 0:
                remaining_timeout = deadline - time.monotonic()
                if remaining_timeout <= 0:
                    break
                self._cv.wait(timeout=min(remaining_timeout, 0.05))

            remaining = self._pending_by_run.get(run_id, 0)
            if remaining > 0:
                dropped_on_close += self._drop_queued_entries_locked(run_id)
                remaining = self._pending_by_run.get(run_id, 0)
            self._draining_runs.discard(run_id)
            return LogSinkDrainResult(
                remaining_queue_size=remaining,
                dropped_on_close=dropped_on_close,
                closed_cleanly=remaining == 0,
            )

    def __call__(self, message) -> None:
        cfg = get_observability_config()
        if not cfg.enabled:
            return
        record = message.record
        extra = record.get("extra", {})
        if _should_skip_observability(record):
            return
        run_id = extra.get("trace_run_id")
        if not run_id:
            return
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
        now = time.monotonic()
        with self._cv:
            if self._closed:
                return
            binding = self._routes.get(str(run_id))
            if binding is None or binding.closing:
                return
            binding.last_seen_at = now
            if len(self._queue) >= self._max_queue:
                self._dropped_by_run[str(run_id)] = self._dropped_by_run.get(str(run_id), 0) + 1
                if str(run_id) not in self._backpressure_notified_runs:
                    self._emit_backpressure_locked(str(run_id))
                    self._backpressure_notified_runs.add(str(run_id))
                return
            self._queue.append(_QueuedLogRecord(run_id=str(run_id), payload=payload, sample_id=sample_id))
            self._pending_by_run[str(run_id)] = self._pending_by_run.get(str(run_id), 0) + 1
            self._cv.notify_all()

    def close(self) -> None:
        with self._cv:
            if self._closed:
                return
            self._closed = True
            self._cv.notify_all()
        self._worker.join(timeout=1.0)

    def route_count(self) -> int:
        with self._cv:
            return len(self._routes)

    def pending_count(self, run_id: str) -> int:
        with self._cv:
            return self._pending_by_run.get(run_id, 0)

    def orphan_count(self) -> int:
        with self._cv:
            return len(self._orphaned_pending_since)

    def reap_zombie_routes(self, now: Optional[float] = None) -> None:
        with self._cv:
            self._maybe_reap_locked(time.monotonic() if now is None else now, force=True)

    def _worker_loop(self) -> None:
        while True:
            with self._cv:
                while not self._queue and not self._closed:
                    self._maybe_reap_locked(time.monotonic())
                    self._cv.wait(timeout=self._flush_interval)
                if not self._queue and self._closed:
                    return
                batch_size = self._effective_batch_size_locked()
                batch: list[_QueuedLogRecord] = []
                while self._queue and len(batch) < batch_size:
                    batch.append(self._queue.popleft())
                self._maybe_reap_locked(time.monotonic())

            for entry in batch:
                self._deliver(entry)

    def _deliver(self, entry: _QueuedLogRecord) -> None:
        trace = None
        with self._cv:
            binding = self._routes.get(entry.run_id)
            if binding is not None:
                binding.last_seen_at = time.monotonic()
                trace = binding.trace_ref()
                if trace is None:
                    self._cleanup_route_locked(entry.run_id)
            else:
                trace = None
        try:
            if trace is not None:
                trace.emit_log_event(entry.payload, sample_id=entry.sample_id)
        except Exception:
            logger.bind(skip_observability=True).exception(
                "ObservabilityLogSink failed to emit log event for run {}",
                entry.run_id,
            )
        finally:
            with self._cv:
                self._decrement_pending_locked(entry.run_id)
                self._maybe_reap_locked(time.monotonic())

    def _effective_batch_size_locked(self) -> int:
        if not self._draining_runs:
            return self._batch_size
        draining_pending = max((self._pending_by_run.get(run_id, 0) for run_id in self._draining_runs), default=0)
        if draining_pending <= 0:
            return self._batch_size
        return max(self._batch_size, min(draining_pending, self._batch_size * 4))

    def _drop_queued_entries_locked(self, run_id: str) -> int:
        if not self._queue:
            return 0
        kept: Deque[_QueuedLogRecord] = deque()
        dropped = 0
        while self._queue:
            entry = self._queue.popleft()
            if entry.run_id == run_id:
                dropped += 1
            else:
                kept.append(entry)
        self._queue = kept
        if dropped:
            self._decrement_pending_locked(run_id, dropped)
            self._dropped_by_run[run_id] = self._dropped_by_run.get(run_id, 0) + dropped
        return dropped

    def _decrement_pending_locked(self, run_id: str, count: int = 1) -> None:
        pending = max(0, self._pending_by_run.get(run_id, 0) - count)
        if pending == 0:
            self._pending_by_run.pop(run_id, None)
            self._backpressure_notified_runs.discard(run_id)
            self._orphaned_pending_since.pop(run_id, None)
        else:
            self._pending_by_run[run_id] = pending
        self._cv.notify_all()

    def _cleanup_route_locked(self, run_id: str, *, clear_pending: bool = False) -> None:
        self._routes.pop(run_id, None)
        self._draining_runs.discard(run_id)
        self._backpressure_notified_runs.discard(run_id)
        self._dropped_by_run.pop(run_id, None)
        if clear_pending or self._pending_by_run.get(run_id, 0) <= 0:
            self._pending_by_run.pop(run_id, None)
            self._orphaned_pending_since.pop(run_id, None)
        else:
            self._orphaned_pending_since.setdefault(run_id, time.monotonic())

    def _maybe_reap_locked(self, now: float, *, force: bool = False) -> None:
        if not force and now - self._last_sweep_at < self._route_sweep_interval_s:
            return
        self._last_sweep_at = now
        for run_id, binding in list(self._routes.items()):
            trace = binding.trace_ref()
            pending = self._pending_by_run.get(run_id, 0)
            if trace is None and pending <= 0:
                self._cleanup_route_locked(run_id, clear_pending=True)
                continue
            if binding.closing and now - binding.last_seen_at >= self._zombie_route_ttl_s:
                dropped = self._drop_queued_entries_locked(run_id)
                remaining = self._pending_by_run.pop(run_id, 0)
                self._cleanup_route_locked(run_id, clear_pending=True)
                if dropped or remaining:
                    logger.bind(skip_observability=True).warning(
                        "ObservabilityLogSink reaped zombie route run_id={} (dropped={}, remaining={})",
                        run_id,
                        dropped,
                        remaining,
                    )
        for run_id, orphaned_since in list(self._orphaned_pending_since.items()):
            remaining = self._pending_by_run.get(run_id, 0)
            if remaining <= 0:
                self._pending_by_run.pop(run_id, None)
                self._orphaned_pending_since.pop(run_id, None)
                self._backpressure_notified_runs.discard(run_id)
                self._dropped_by_run.pop(run_id, None)
                continue
            if now - orphaned_since < self._zombie_route_ttl_s:
                continue
            self._pending_by_run.pop(run_id, None)
            self._orphaned_pending_since.pop(run_id, None)
            self._backpressure_notified_runs.discard(run_id)
            self._dropped_by_run.pop(run_id, None)
            logger.bind(skip_observability=True).warning(
                "ObservabilityLogSink reaped orphan pending state run_id={} (remaining={})",
                run_id,
                remaining,
            )

    def _emit_backpressure_locked(self, run_id: str) -> None:
        binding = self._routes.get(run_id)
        if binding is None:
            return
        trace = binding.trace_ref()
        if trace is None:
            return
        dropped = self._dropped_by_run.get(run_id, 0)
        if dropped <= 0:
            return
        try:
            trace.emit(
                "log_backpressure",
                {"dropped": dropped, "queue_size": self._max_queue, "run_id": run_id},
            )
        except Exception:
            logger.bind(skip_observability=True).exception(
                "ObservabilityLogSink failed to emit backpressure event for run {}",
                run_id,
            )
        finally:
            self._dropped_by_run[run_id] = 0


def configure_observable_log_sink() -> Optional[ObservabilityLogSink]:
    """Install the global log sink if enabled and return it."""

    if not _log_sink_enabled():
        return None
    global _GLOBAL_SINK, _GLOBAL_SINK_ID
    with _SINK_LOCK:
        if _GLOBAL_SINK is None or _GLOBAL_SINK.closed:
            sink = ObservabilityLogSink()
            sink_id = logger.add(sink, level=os.environ.get("GAGE_EVAL_LOG_SINK_LEVEL", "INFO"))
            _GLOBAL_SINK = sink
            _GLOBAL_SINK_ID = sink_id
        return _GLOBAL_SINK


def register_observable_trace(trace: "ObservabilityTrace") -> None:
    if not trace.accepts_new_events():
        return
    sink = configure_observable_log_sink()
    if sink is not None:
        sink.register_trace(trace)


def get_observable_log_sink() -> Optional[ObservabilityLogSink]:
    with _SINK_LOCK:
        return _GLOBAL_SINK


def _log_sink_enabled() -> bool:
    value = os.environ.get("GAGE_EVAL_ENABLE_LOG_SINK")
    if value is None:
        return True
    return value.lower() in {"1", "true", "yes", "on"}


def is_log_sink_active() -> bool:
    sink = get_observable_log_sink()
    return sink is not None and not sink.closed


def _should_skip_observability(record: Dict[str, Any]) -> bool:
    extra = record.get("extra", {})
    return bool(extra.get("skip_observability"))


def _env_int(name: str, default: int) -> int:
    value = os.environ.get(name)
    if value is None:
        return default
    try:
        return int(value)
    except ValueError:
        logger.bind(skip_observability=True).warning(
            "Invalid integer for {}={}; using default {}",
            name,
            value,
            default,
        )
        return default


def _env_float(name: str, default: float) -> float:
    value = os.environ.get(name)
    if value is None:
        return default
    try:
        return float(value)
    except ValueError:
        logger.bind(skip_observability=True).warning(
            "Invalid float for {}={}; using default {}",
            name,
            value,
            default,
        )
        return default
