"""Observability utilities."""

from __future__ import annotations

import os
import time
import uuid
from datetime import datetime
from contextlib import contextmanager
from pathlib import Path
from threading import Lock
from typing import Any, Dict, List, Optional

from loguru import logger
from gage_eval.reporting.recorders import HTTPRecorder, RecorderBase, FileRecorder, InMemoryRecorder
from gage_eval.observability.config import get_observability_config
from gage_eval.observability.log_sink import configure_observable_log_sink


class ObservabilityTrace:
    """Thread-safe event bus backed by Recorder implementations."""

    def __init__(self, recorder: Optional[RecorderBase] = None, run_id: Optional[str] = None) -> None:
        self.run_id = run_id or self._generate_run_id()
        self._recorder = recorder or self._build_default_recorder()
        self._events: List[Dict[str, Any]] = []
        self._lock = Lock()
        configure_observable_log_sink(self)

    def emit(self, event: str, payload: Dict[str, Any], sample_id: Optional[str] = None) -> None:
        with self._lock:
            self._events.append(
                {"event": event, "payload": payload, "ts": time.time(), "sample_id": sample_id}
            )
        logger.trace("Trace[{}] event={} sample={}", self.run_id, event, sample_id)
        self._recorder.record_event(event, payload, sample_id=sample_id)

    def emit_tool_documentation(self, payload: Dict[str, Any], sample_id: Optional[str] = None) -> None:
        """Emit tool documentation metrics for observability."""

        self.emit("tool_documentation_built", payload, sample_id=sample_id)

    @contextmanager
    def use_sample(self, sample_id: Optional[str]):
        with self._recorder.use_sample(sample_id):
            yield

    def flush(self) -> None:
        self._recorder.flush_events()
        logger.debug("Trace[{}] flushed events", self.run_id)

    @property
    def events(self) -> List[Dict[str, Any]]:
        with self._lock:
            return list(self._events)

    @property
    def recorder(self) -> RecorderBase:
        return self._recorder

    def force_log(self, sample_id: Optional[str]) -> None:
        """Ensure future stages sample the provided sample_id regardless of rate."""

        if not sample_id:
            return
        get_observability_config().force_log(sample_id)

    def _build_default_recorder(self) -> RecorderBase:
        save_dir = Path(os.environ.get("GAGE_EVAL_SAVE_DIR", "./runs")) / self.run_id
        file_recorder = FileRecorder(run_id=self.run_id, output_path=save_dir / "events.jsonl")

        http_url = os.environ.get("GAGE_EVAL_REPORT_HTTP_URL")
        if http_url:
            batch = int(os.environ.get("GAGE_EVAL_REPORT_HTTP_BATCH", "50"))
            fail_pct = float(os.environ.get("GAGE_EVAL_REPORT_HTTP_FAIL_PCT", "5"))
            timeout = float(os.environ.get("GAGE_EVAL_REPORT_HTTP_TIMEOUT", "10"))
            return HTTPRecorder(
                run_id=self.run_id,
                url=http_url,
                batch_size=batch,
                timeout=timeout,
                fail_threshold_pct=fail_pct,
                fallback=file_recorder,
            )
        inmemory = os.environ.get("GAGE_EVAL_INMEMORY_TRACE")
        if inmemory:
            return InMemoryRecorder(run_id=self.run_id)
        return file_recorder

    @staticmethod
    def _generate_run_id() -> str:
        # NOTE: Use a human-friendly timestamp (MMddHHMMSS) while keeping second-level
        # precision to reduce collisions.
        return datetime.now().strftime("%m%d%H%M%S")
