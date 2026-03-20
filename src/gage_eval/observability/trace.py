"""Observability utilities."""

from __future__ import annotations

import os
import time
from collections import deque
from dataclasses import dataclass
from datetime import datetime
from contextlib import contextmanager
from pathlib import Path
from threading import Condition, Lock
from typing import Any, Deque, Dict, List, Optional

from loguru import logger
from gage_eval.reporting.recorders import (
    HTTPRecorder,
    FileRecorder,
    InMemoryRecorder,
    ObservabilityHealth,
    RecorderBase,
    ResilientRecorder,
    TraceEvent,
)
from gage_eval.observability.config import get_observability_config
from gage_eval.observability.log_sink import (
    LogSinkDrainResult,
    configure_observable_log_sink,
    get_observable_log_sink,
    register_observable_trace,
)
from gage_eval.utils.run_identity import RunIdentity, build_run_identity


@dataclass(frozen=True, slots=True)
class ObservabilityCloseResult:
    """Structured close result produced after draining log sinks and recorders."""

    closed_cleanly: bool
    close_mode: str
    remaining_queue_size: int
    dropped_on_close: int
    warning: Optional[str] = None
    error_type: Optional[str] = None
    error_message: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        payload = {
            "observability_closed_cleanly": self.closed_cleanly,
            "observability_close_mode": self.close_mode,
            "observability_close_remaining_queue": self.remaining_queue_size,
            "observability_dropped_on_close": self.dropped_on_close,
        }
        if self.warning:
            payload["observability_close_warning"] = self.warning
        if self.error_type:
            payload["observability_close_error_type"] = self.error_type
        if self.error_message:
            payload["observability_close_error_message"] = self.error_message
        return payload


class ObservabilityTrace:
    """Thread-safe event bus backed by Recorder implementations."""

    def __init__(self, recorder: Optional[RecorderBase | ResilientRecorder] = None, run_id: Optional[str] = None) -> None:
        self.run_identity: RunIdentity = build_run_identity(run_id)
        self.run_id = self.run_identity.run_id
        self._trace_buffer_max_events = max(0, _env_int("GAGE_EVAL_TRACE_BUFFER_MAX_EVENTS", 2048))
        self._events: Deque[Dict[str, Any]] = deque(maxlen=self._trace_buffer_max_events or None)
        self._events_emitted_total = 0
        self._events_dropped_by_ring_buffer = 0
        self._lock = Lock()
        self._lifecycle_cv = Condition(Lock())
        self._next_event_id = 0
        self._recorder = self._wrap_recorder(recorder) if recorder is not None else self._build_default_recorder()
        self._lifecycle_state = "open"
        self._close_result: Optional[ObservabilityCloseResult] = None

    def emit(self, event: str, payload: Dict[str, Any], sample_id: Optional[str] = None) -> None:
        self._emit_internal(event, payload, sample_id=sample_id, allow_during_closing=False)

    def emit_log_event(self, payload: Dict[str, Any], sample_id: Optional[str] = None) -> None:
        """Emit a sink-delivered log event while the trace is draining."""

        self._emit_internal("log", payload, sample_id=sample_id, allow_during_closing=True)

    def _emit_internal(
        self,
        event: str,
        payload: Dict[str, Any],
        *,
        sample_id: Optional[str] = None,
        allow_during_closing: bool,
    ) -> None:
        state = self.lifecycle_state
        if state == "closed" or (state == "closing" and not allow_during_closing):
            logger.bind(skip_observability=True).warning(
                "Trace[{}] dropped event={} because lifecycle_state={}",
                self.run_id,
                event,
                state,
            )
            return
        trace_event = self._create_trace_event(event, payload, sample_id=sample_id)
        with self._lock:
            self._events_emitted_total += 1
            if self._trace_buffer_max_events > 0:
                if len(self._events) >= self._trace_buffer_max_events:
                    self._events_dropped_by_ring_buffer += 1
                self._events.append(trace_event.to_dict())
        logger.trace("Trace[{}] event={} sample={}", self.run_id, event, sample_id)
        self._recorder.record_trace_event(trace_event)

    def emit_tool_documentation(self, payload: Dict[str, Any], sample_id: Optional[str] = None) -> None:
        """Emit tool documentation metrics for observability."""

        self.emit("tool_documentation_built", payload, sample_id=sample_id)

    @contextmanager
    def use_sample(self, sample_id: Optional[str]):
        with self._recorder.use_sample(sample_id):
            yield

    @contextmanager
    def activate(self):
        if self.lifecycle_state != "open":
            raise RuntimeError(f"Trace[{self.run_id}] cannot activate from state {self.lifecycle_state}")
        configure_observable_log_sink()
        register_observable_trace(self)
        with logger.contextualize(trace_run_id=self.run_id):
            yield

    def flush(self) -> None:
        self._recorder.flush_events()
        logger.bind(skip_observability=True).debug("Trace[{}] flushed events", self.run_id)

    def close(
        self,
        *,
        close_mode: Optional[str] = None,
        drain_timeout_s: Optional[float] = None,
        cache_store=None,
    ) -> ObservabilityCloseResult:
        with self._lifecycle_cv:
            while self._lifecycle_state == "closing" and self._close_result is None:
                self._lifecycle_cv.wait(timeout=0.05)
            if self._lifecycle_state == "closed" and self._close_result is not None:
                result = self._close_result
                if cache_store is not None:
                    try:
                        payload = self.health_snapshot()
                        payload.update(result.to_dict())
                        cache_store.merge_summary_fields(payload)
                    except Exception as exc:  # pragma: no cover - defensive
                        logger.bind(skip_observability=True).warning(
                            "Trace[{}] failed to patch summary with cached close result: {}",
                            self.run_id,
                            exc,
                        )
                return result
            self._lifecycle_state = "closing"

        mode = (close_mode or os.environ.get("GAGE_EVAL_LOG_SINK_CLOSE_MODE", "drain")).strip().lower() or "drain"
        if mode not in {"drain", "best_effort"}:
            mode = "drain"
        timeout_s = drain_timeout_s
        if timeout_s is None:
            timeout_s = _env_float("GAGE_EVAL_LOG_SINK_DRAIN_TIMEOUT_S", 2.0)

        warning = None
        if mode == "best_effort":
            warning = "best_effort mode may leave observability data incomplete"

        drain_result = LogSinkDrainResult(remaining_queue_size=0, dropped_on_close=0, closed_cleanly=True)
        close_error: Optional[Exception] = None
        sink = get_observable_log_sink()
        if sink is not None:
            try:
                drain_result = sink.flush_run(
                    self.run_id,
                    timeout_s=max(0.0, float(timeout_s)),
                    close_mode=mode,
                )
            except Exception as exc:  # pragma: no cover - defensive
                close_error = exc
                drain_result = LogSinkDrainResult(
                    remaining_queue_size=0,
                    dropped_on_close=0,
                    closed_cleanly=False,
                )
                logger.bind(skip_observability=True).warning(
                    "Trace[{}] failed to drain log sink: {}",
                    self.run_id,
                    exc,
                )

        try:
            self._recorder.close()
        except Exception as exc:  # pragma: no cover - defensive
            if close_error is None:
                close_error = exc
            logger.bind(skip_observability=True).warning(
                "Trace[{}] failed to close recorder: {}",
                self.run_id,
                exc,
            )
        finally:
            sink = get_observable_log_sink()
            if sink is not None:
                sink.unregister_trace(self.run_id)

        result = ObservabilityCloseResult(
            closed_cleanly=drain_result.closed_cleanly and close_error is None,
            close_mode=mode,
            remaining_queue_size=drain_result.remaining_queue_size,
            dropped_on_close=drain_result.dropped_on_close,
            warning=warning,
            error_type=type(close_error).__name__ if close_error is not None else None,
            error_message=str(close_error) if close_error is not None else None,
        )
        if cache_store is not None:
            try:
                payload = self.health_snapshot()
                payload.update(result.to_dict())
                cache_store.merge_summary_fields(payload)
            except Exception as exc:  # pragma: no cover - defensive
                logger.bind(skip_observability=True).warning(
                    "Trace[{}] failed to patch summary with close result: {}",
                    self.run_id,
                    exc,
                )
        with self._lifecycle_cv:
            self._lifecycle_state = "closed"
            self._close_result = result
            self._lifecycle_cv.notify_all()
        return result

    @property
    def lifecycle_state(self) -> str:
        with self._lifecycle_cv:
            return self._lifecycle_state

    def accepts_new_events(self) -> bool:
        return self.lifecycle_state == "open"

    @property
    def events(self) -> List[Dict[str, Any]]:
        with self._lock:
            return list(self._events)

    @property
    def recorder(self) -> RecorderBase | ResilientRecorder:
        return self._recorder

    def health_snapshot(self) -> Dict[str, Any]:
        snapshot: ObservabilityHealth = self._recorder.health_snapshot()
        with self._lock:
            retained_events = len(self._events)
            emitted_total = self._events_emitted_total
            dropped_events = self._events_dropped_by_ring_buffer
        payload = snapshot.to_dict()
        payload.update(
            {
                "events_emitted_total": emitted_total,
                "events_retained_in_memory": retained_events,
                "events_dropped_by_ring_buffer": dropped_events,
            }
        )
        return payload

    def force_log(self, sample_id: Optional[str]) -> None:
        """Ensure future stages sample the provided sample_id regardless of rate."""

        if not sample_id:
            return
        get_observability_config().force_log(sample_id)

    def _build_default_recorder(self) -> ResilientRecorder:
        save_dir = Path(os.environ.get("GAGE_EVAL_SAVE_DIR", "./runs")) / self.run_id
        file_recorder = FileRecorder(run_id=self.run_id, output_path=save_dir / "events.jsonl")

        http_url = os.environ.get("GAGE_EVAL_REPORT_HTTP_URL")
        if http_url:
            batch = int(os.environ.get("GAGE_EVAL_REPORT_HTTP_BATCH", "50"))
            fail_pct = float(os.environ.get("GAGE_EVAL_REPORT_HTTP_FAIL_PCT", "5"))
            timeout = float(os.environ.get("GAGE_EVAL_REPORT_HTTP_TIMEOUT", "10"))
            max_retries = int(os.environ.get("GAGE_EVAL_REPORT_HTTP_MAX_RETRIES", "2"))
            base_retry_delay_ms = float(os.environ.get("GAGE_EVAL_REPORT_HTTP_RETRY_BASE_MS", "50"))
            max_retry_delay_ms = float(os.environ.get("GAGE_EVAL_REPORT_HTTP_RETRY_MAX_MS", "500"))
            backoff_multiplier = float(os.environ.get("GAGE_EVAL_REPORT_HTTP_RETRY_MULTIPLIER", "2.0"))
            primary = HTTPRecorder(
                run_id=self.run_id,
                url=http_url,
                batch_size=batch,
                timeout=timeout,
                fail_threshold_pct=fail_pct,
                max_retries=max_retries,
                base_retry_delay_ms=base_retry_delay_ms,
                max_retry_delay_ms=max_retry_delay_ms,
                backoff_multiplier=backoff_multiplier,
                fallback=file_recorder,
            )
            return ResilientRecorder(primary, fallback=file_recorder)
        inmemory = os.environ.get("GAGE_EVAL_INMEMORY_TRACE")
        if inmemory:
            return ResilientRecorder(InMemoryRecorder(run_id=self.run_id))
        return ResilientRecorder(file_recorder)

    def _wrap_recorder(self, recorder: RecorderBase | ResilientRecorder) -> ResilientRecorder:
        if isinstance(recorder, ResilientRecorder):
            return recorder
        fallback = recorder.fallback if isinstance(recorder, HTTPRecorder) else None
        return ResilientRecorder(recorder, fallback=fallback)

    def _create_trace_event(
        self,
        event: str,
        payload: Dict[str, Any],
        *,
        sample_id: Optional[str] = None,
    ) -> TraceEvent:
        event_id = self._allocate_event_id()
        return TraceEvent(
            run_id=self.run_id,
            event_id=event_id,
            event=event,
            payload=payload,
            sample_id=sample_id,
            created_at=time.time(),
        )

    def _allocate_event_id(self) -> int:
        allocate = getattr(self._recorder, "allocate_event_id", None)
        if callable(allocate):
            return int(allocate())
        with self._lock:
            event_id = self._next_event_id
            self._next_event_id += 1
            return event_id

    @staticmethod
    def _generate_run_id() -> str:
        return build_run_identity().run_id


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


def _env_int(name: str, default: int) -> int:
    value = os.environ.get(name)
    if value is None:
        return default
    try:
        return int(value)
    except ValueError:
        logger.bind(skip_observability=True).warning(
            "Invalid int for {}={}; using default {}",
            name,
            value,
            default,
        )
        return default
