"""Observability utilities."""

from __future__ import annotations

import os
import time
from datetime import datetime
from contextlib import contextmanager
from pathlib import Path
from threading import Lock
from typing import Any, Dict, List, Optional

from loguru import logger
from gage_eval.reporting.recorders import (
    HTTPRecorder,
    FileRecorder,
    InMemoryRecorder,
    RecorderBase,
    ResilientRecorder,
    TraceEvent,
)
from gage_eval.observability.config import get_observability_config
from gage_eval.observability.log_sink import configure_observable_log_sink
from gage_eval.utils.run_identity import RunIdentity, build_run_identity


class ObservabilityTrace:
    """Thread-safe event bus backed by Recorder implementations."""

    def __init__(self, recorder: Optional[RecorderBase | ResilientRecorder] = None, run_id: Optional[str] = None) -> None:
        self.run_identity: RunIdentity = build_run_identity(run_id)
        self.run_id = self.run_identity.run_id
        self._events: List[Dict[str, Any]] = []
        self._lock = Lock()
        self._next_event_id = 0
        self._recorder = self._wrap_recorder(recorder) if recorder is not None else self._build_default_recorder()
        configure_observable_log_sink(self)

    def emit(self, event: str, payload: Dict[str, Any], sample_id: Optional[str] = None) -> None:
        trace_event = self._create_trace_event(event, payload, sample_id=sample_id)
        with self._lock:
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

    def flush(self) -> None:
        self._recorder.flush_events()
        logger.bind(skip_observability=True).debug("Trace[{}] flushed events", self.run_id)

    @property
    def events(self) -> List[Dict[str, Any]]:
        with self._lock:
            return list(self._events)

    @property
    def recorder(self) -> RecorderBase | ResilientRecorder:
        return self._recorder

    def health_snapshot(self) -> Dict[str, Any]:
        return self._recorder.health_snapshot().to_dict()

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
