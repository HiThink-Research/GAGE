"""Recorder implementations inspired by OpenAI Evals."""

from __future__ import annotations

import contextlib
import json
import random
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


@dataclass(frozen=True, slots=True)
class ObservabilityHealth:
    """Structured health snapshot for the active observability pipeline."""

    degraded: bool
    mode: str
    primary_sink: str
    active_sink: str
    failure_count: int
    backlog_events: int
    last_error_stage: Optional[str] = None
    last_error_type: Optional[str] = None
    last_error_message: Optional[str] = None
    degraded_since: Optional[float] = None
    events_flushed_total: int = 0
    recorder_compactions_total: int = 0
    recorder_retained_events: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "observability_degraded": self.degraded,
            "observability_mode": self.mode,
            "primary_sink": self.primary_sink,
            "active_sink": self.active_sink,
            "failure_count": self.failure_count,
            "backlog_events": self.backlog_events,
            "last_error_stage": self.last_error_stage,
            "last_error_type": self.last_error_type,
            "last_error_message": self.last_error_message,
            "degraded_since": self.degraded_since,
            "events_flushed_total": self.events_flushed_total,
            "recorder_compactions_total": self.recorder_compactions_total,
            "recorder_retained_events": self.recorder_retained_events,
        }


@dataclass(frozen=True, slots=True)
class RecorderCloseFailure:
    """Structured close failure emitted by recorder shutdown aggregation."""

    recorder_name: str
    error_type: str
    error_message: str


class RecorderCloseError(RuntimeError):
    """Raised when one or more recorder close operations fail."""

    def __init__(self, failures: Sequence[RecorderCloseFailure]) -> None:
        self.failures = tuple(failures)
        super().__init__(
            "; ".join(
                f"{failure.recorder_name}:{failure.error_type}:{failure.error_message}"
                for failure in self.failures
            )
        )


@dataclass(frozen=True, slots=True)
class RecorderBufferStats:
    """Aggregated recorder buffer statistics for observability reporting."""

    events_flushed_total: int = 0
    recorder_compactions_total: int = 0
    recorder_retained_events: int = 0


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
        self._next_event_id = 0
        self._lock = threading.Lock()
        self._flush_lock = threading.Lock()
        self._sample_id: ContextVar[Optional[str]] = ContextVar("recorder_sample_id", default=None)
        self._min_flush_events = max(1, min_flush_events)
        self._min_flush_seconds = max(0.1, min_flush_seconds)
        self._last_flush_ts = time.time()
        self._events_flushed_total = 0
        self._compactions_total = 0

    @contextlib.contextmanager
    def use_sample(self, sample_id: Optional[str]):
        token = self._sample_id.set(sample_id)
        try:
            yield
        finally:
            self._sample_id.reset(token)

    def _build_trace_event(
        self,
        event: str,
        payload: Dict[str, Any],
        *,
        sample_id: Optional[str] = None,
    ) -> TraceEvent:
        sample_ref = sample_id if sample_id is not None else self._sample_id.get()
        return TraceEvent(
            run_id=self.run_id,
            event_id=self.allocate_event_id(),
            event=event,
            payload=payload,
            sample_id=sample_ref,
            created_at=time.time(),
        )

    def allocate_event_id(self) -> int:
        with self._lock:
            event_id = self._next_event_id
            self._next_event_id += 1
            return event_id

    def record_event(self, event: str, payload: Dict[str, Any], *, sample_id: Optional[str] = None) -> None:
        trace_event = self._build_trace_event(event, payload, sample_id=sample_id)
        self.record_trace_event(trace_event)

    def record_trace_event(self, trace_event: TraceEvent) -> None:
        with self._lock:
            self._observe_event_id_locked(trace_event.event_id)
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

    def pending_events(self) -> Sequence[TraceEvent]:
        with self._lock:
            return list(self._events[self._written :])

    def write_events(self, events: Sequence[TraceEvent]) -> None:
        self._flush_events_internal(events)

    def flush_events(self) -> None:
        with self._flush_lock:
            with self._lock:
                if self._written >= len(self._events):
                    return
                start_idx = self._written
                snapshot_len = len(self._events)
                events_to_write = tuple(self._events[start_idx:snapshot_len])
            self.write_events(events_to_write)
            with self._lock:
                committed_end = min(snapshot_len, len(self._events))
                if committed_end > 0:
                    del self._events[:committed_end]
                    self._compactions_total += 1
                self._written = 0
                self._events_flushed_total += len(events_to_write)
                self._last_flush_ts = time.time()
                retained_count = len(self._events)
        logger.debug(
            "Recorder '{}' flushed {} events (retained={})",
            self.__class__.__name__,
            len(events_to_write),
            retained_count,
        )

    def close(self) -> None:
        self.flush_events()

    def _observe_event_id_locked(self, event_id: int) -> None:
        if event_id >= self._next_event_id:
            self._next_event_id = event_id + 1

    def buffer_stats(self) -> RecorderBufferStats:
        with self._lock:
            return RecorderBufferStats(
                events_flushed_total=self._events_flushed_total,
                recorder_compactions_total=self._compactions_total,
                recorder_retained_events=len(self._events),
            )

    def _flush_events_internal(self, events: Sequence[TraceEvent]) -> None:  # pragma: no cover - abstract
        raise NotImplementedError


@registry.asset(
    "reporting_sinks",
    "inmemory",
    desc="In-memory event recorder (tests)",
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
    desc="Local JSONL event recorder",
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


class NoopRecorder(RecorderBase):
    """Recorder that drops all events after the observability pipeline gives up."""

    def record_event(self, event: str, payload: Dict[str, Any], *, sample_id: Optional[str] = None) -> None:
        return

    def record_trace_event(self, trace_event: TraceEvent) -> None:
        return

    def pending_events(self) -> Sequence[TraceEvent]:
        return ()

    def flush_events(self) -> None:
        return

    def close(self) -> None:
        return

    def _flush_events_internal(self, events: Sequence[TraceEvent]) -> None:
        return


@registry.asset(
    "reporting_sinks",
    "http",
    desc="HTTP event recorder with fallback support",
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
        max_retries: int = 2,
        base_retry_delay_ms: float = 50.0,
        max_retry_delay_ms: float = 500.0,
        backoff_multiplier: float = 2.0,
        fallback: Optional[RecorderBase] = None,
        **kwargs,
    ) -> None:
        super().__init__(run_id, **kwargs)
        self.url = url
        self.batch_size = max(1, batch_size)
        self.timeout = timeout
        self.fail_threshold_pct = max(0.0, fail_threshold_pct)
        self.max_retries = max(0, max_retries)
        self.base_retry_delay_s = max(0.0, base_retry_delay_ms / 1000.0)
        self.max_retry_delay_s = max(self.base_retry_delay_s, max_retry_delay_ms / 1000.0)
        self.backoff_multiplier = max(1.0, backoff_multiplier)
        self.fallback = fallback
        self._session = requests.Session()
        self._requests_sent = 0
        self._requests_failed = 0
        self._failover_triggered = False
        self._sleep = time.sleep
        self._random_uniform = random.uniform
        self._session_closed = False

    def _flush_events_internal(self, events: Sequence[TraceEvent]) -> None:
        if not events:
            return
        if self._failover_triggered:
            self._flush_via_fallback(events, None)
            return

        for chunk in _chunk(events, self.batch_size):
            self._requests_sent += 1
            self._flush_chunk_with_retry(chunk)

    def _flush_chunk_with_retry(self, events: Sequence[TraceEvent]) -> None:
        last_exc: Optional[requests.RequestException] = None
        for attempt in range(self.max_retries + 1):
            try:
                self._post_events(events)
            except requests.RequestException as exc:  # pragma: no cover - network branch
                last_exc = exc
                if attempt < self.max_retries and self._is_retryable_error(exc):
                    delay_s = self._compute_retry_delay(attempt)
                    logger.bind(skip_observability=True).warning(
                        "HTTPRecorder transient error ({}). Retrying in {:.3f}s ({}/{})",
                        exc,
                        delay_s,
                        attempt + 1,
                        self.max_retries,
                    )
                    if delay_s > 0:
                        self._sleep(delay_s)
                    continue
                self._requests_failed += 1
                logger.bind(skip_observability=True).warning(
                    "HTTPRecorder error ({}). Falling back to local recorder.",
                    exc,
                )
                self._flush_via_fallback(events, exc)
                self._check_failover()
                return
            else:
                logger.bind(skip_observability=True).debug(
                    "HTTPRecorder flushed {} events to {}",
                    len(events),
                    self.url,
                )
                return
        if last_exc is not None:  # pragma: no cover - defensive guard
            raise last_exc

    def _post_events(self, events: Sequence[TraceEvent]) -> None:
        payload = {"events": [event.to_dict() for event in events]}
        response = self._session.post(self.url, json=payload, timeout=self.timeout)
        response.raise_for_status()

    def _flush_via_fallback(self, events: Sequence[TraceEvent], exc: Optional[requests.RequestException]) -> None:
        if self.fallback is None:
            if exc is not None:
                raise exc
            raise RuntimeError("HTTPRecorder fallback not configured")
        self.fallback._flush_events_internal(events)

    def _check_failover(self) -> None:
        if self._requests_sent == 0:
            return
        failure_pct = (self._requests_failed / self._requests_sent) * 100
        if failure_pct >= self.fail_threshold_pct:
            self._failover_triggered = True
            logger.bind(skip_observability=True).error(
                "HTTPRecorder failure rate {:.2f}% exceeded threshold {:.2f}%. Permanent failover to local recorder.",
                failure_pct,
                self.fail_threshold_pct,
            )

    def _is_retryable_error(self, exc: requests.RequestException) -> bool:
        if isinstance(exc, (requests.Timeout, requests.ConnectionError)):
            return True
        if isinstance(exc, requests.HTTPError) and exc.response is not None:
            status = exc.response.status_code
            return status == 429 or 500 <= status < 600
        return False

    def _compute_retry_delay(self, attempt: int) -> float:
        capped = min(
            self.max_retry_delay_s,
            self.base_retry_delay_s * (self.backoff_multiplier**attempt),
        )
        return self._random_uniform(0.0, capped)

    def close(self) -> None:
        flush_error: Optional[Exception] = None
        try:
            super().close()
        except Exception as exc:
            flush_error = exc
        finally:
            if not self._session_closed:
                close_session = getattr(self._session, "close", None)
                try:
                    if callable(close_session):
                        close_session()
                except Exception as exc:  # pragma: no cover - defensive close path
                    if flush_error is None:
                        flush_error = exc
                finally:
                    self._session_closed = True
        if flush_error is not None:
            raise flush_error


class ResilientRecorder:
    """Facade that keeps recorder failures from bubbling into the main flow."""

    def __init__(self, primary: RecorderBase, *, fallback: Optional[RecorderBase] = None) -> None:
        self.run_id = primary.run_id
        self._primary = primary
        self._fallback = fallback if fallback is not primary else None
        self._noop = NoopRecorder(run_id=primary.run_id)
        self._active: RecorderBase = primary
        self._mode = "primary"
        self._failure_count = 0
        self._last_error_stage: Optional[str] = None
        self._last_error_type: Optional[str] = None
        self._last_error_message: Optional[str] = None
        self._degraded_since: Optional[float] = None
        self._backlog_events = 0
        self._state_lock = threading.Lock()

    @contextlib.contextmanager
    def use_sample(self, sample_id: Optional[str]):
        with self._active_recorder().use_sample(sample_id):
            yield

    def allocate_event_id(self) -> int:
        return self._primary.allocate_event_id()

    def record_event(self, event: str, payload: Dict[str, Any], *, sample_id: Optional[str] = None) -> None:
        trace_event = self._primary._build_trace_event(event, payload, sample_id=sample_id)
        self.record_trace_event(trace_event)

    def record_trace_event(self, trace_event: TraceEvent) -> None:
        recorder = self._active_recorder()
        try:
            recorder.record_trace_event(trace_event)
        except Exception as exc:
            self._handle_failure(stage="record", exc=exc, current_event=trace_event)
        else:
            self._sync_backlog(recorder)

    def flush_events(self) -> None:
        recorder = self._active_recorder()
        try:
            recorder.flush_events()
        except Exception as exc:
            self._handle_failure(stage="flush", exc=exc)
        else:
            self._sync_backlog(recorder)

    def close(self) -> None:
        self.flush_events()
        close_failures: list[RecorderCloseFailure] = []
        for recorder in self._iter_recorders():
            try:
                recorder.close()
            except Exception as exc:  # pragma: no cover - defensive close path
                close_failures.append(
                    RecorderCloseFailure(
                        recorder_name=_sink_name(recorder),
                        error_type=type(exc).__name__,
                        error_message=str(exc),
                    )
                )
                with self._state_lock:
                    self._failure_count += 1
                    self._last_error_stage = "close"
                    self._last_error_type = type(exc).__name__
                    self._last_error_message = str(exc)
                    if self._degraded_since is None:
                        self._degraded_since = time.time()
                logger.bind(skip_observability=True).warning(
                    "Observability recorder close failed for {} with {}",
                    _sink_name(recorder),
                    type(exc).__name__,
                )
        if close_failures:
            raise RecorderCloseError(tuple(close_failures))

    def health_snapshot(self) -> ObservabilityHealth:
        buffer_stats = self._aggregate_buffer_stats()
        with self._state_lock:
            return ObservabilityHealth(
                degraded=self._mode != "primary",
                mode=self._mode,
                primary_sink=_sink_name(self._primary),
                active_sink=_sink_name(self._active),
                failure_count=self._failure_count,
                backlog_events=self._backlog_events,
                last_error_stage=self._last_error_stage,
                last_error_type=self._last_error_type,
                last_error_message=self._last_error_message,
                degraded_since=self._degraded_since,
                events_flushed_total=buffer_stats.events_flushed_total,
                recorder_compactions_total=buffer_stats.recorder_compactions_total,
                recorder_retained_events=buffer_stats.recorder_retained_events,
            )

    def _handle_failure(
        self,
        *,
        stage: str,
        exc: Exception,
        current_event: Optional[TraceEvent] = None,
    ) -> None:
        with self._state_lock:
            failed_recorder = self._active
            backlog = tuple(failed_recorder.pending_events()) if hasattr(failed_recorder, "pending_events") else ()
            delivery_batch = _merge_events(backlog, current_event)
            self._failure_count += 1
            self._last_error_stage = stage
            self._last_error_type = type(exc).__name__
            self._last_error_message = str(exc)
            if self._degraded_since is None:
                self._degraded_since = time.time()
            self._backlog_events = len(delivery_batch)
            target_mode, target = self._next_target_locked()
            self._mode = target_mode
            self._active = target
        self._log_failure(stage=stage, exc=exc, target_mode=target_mode, target=target)

        if not delivery_batch:
            self._sync_backlog(target)
            return
        try:
            target.write_events(delivery_batch)
        except Exception as fallback_exc:
            with self._state_lock:
                self._failure_count += 1
                self._last_error_stage = stage
                self._last_error_type = type(fallback_exc).__name__
                self._last_error_message = str(fallback_exc)
                self._mode = "noop"
                self._active = self._noop
                self._backlog_events = len(delivery_batch)
            self._log_failure(stage=stage, exc=fallback_exc, target_mode="noop", target=self._noop)
        else:
            if target is self._noop:
                with self._state_lock:
                    self._backlog_events = len(delivery_batch)
            else:
                self._sync_backlog(target)

    def _active_recorder(self) -> RecorderBase:
        with self._state_lock:
            return self._active

    def _next_target_locked(self) -> tuple[str, RecorderBase]:
        if self._mode == "primary" and self._fallback is not None:
            return "fallback", self._fallback
        return "noop", self._noop

    def _iter_recorders(self) -> Sequence[RecorderBase]:
        recorders = [self._primary]
        if self._fallback is not None:
            recorders.append(self._fallback)
        return tuple(recorders)

    def _aggregate_buffer_stats(self) -> RecorderBufferStats:
        totals = RecorderBufferStats()
        for recorder in self._iter_recorders():
            stats = recorder.buffer_stats()
            totals = RecorderBufferStats(
                events_flushed_total=totals.events_flushed_total + stats.events_flushed_total,
                recorder_compactions_total=totals.recorder_compactions_total + stats.recorder_compactions_total,
                recorder_retained_events=totals.recorder_retained_events + stats.recorder_retained_events,
            )
        return totals

    def _sync_backlog(self, recorder: RecorderBase) -> None:
        backlog_events = self._pending_count(recorder)
        with self._state_lock:
            if self._active is recorder and self._mode != "noop":
                self._backlog_events = backlog_events

    def _pending_count(self, recorder: RecorderBase) -> int:
        pending = recorder.pending_events()
        return len(pending)

    def _log_failure(self, *, stage: str, exc: Exception, target_mode: str, target: RecorderBase) -> None:
        logger.bind(skip_observability=True).warning(
            "Observability recorder failed during {} with {}. Switching to {} ({})",
            stage,
            type(exc).__name__,
            target_mode,
            _sink_name(target),
        )


def _chunk(events: Sequence[TraceEvent], size: int) -> Iterable[Sequence[TraceEvent]]:
    for idx in range(0, len(events), size):
        yield events[idx : idx + size]


def _merge_events(
    events: Sequence[TraceEvent],
    current_event: Optional[TraceEvent],
) -> tuple[TraceEvent, ...]:
    merged = list(events)
    if current_event is not None and not any(event.event_id == current_event.event_id for event in merged):
        merged.append(current_event)
    return tuple(merged)


def _sink_name(recorder: RecorderBase) -> str:
    name = recorder.__class__.__name__
    if name.endswith("Recorder"):
        name = name[: -len("Recorder")]
    return name.lower()
