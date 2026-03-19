from __future__ import annotations

import requests

import pytest

from gage_eval.reporting.recorders import HTTPRecorder, InMemoryRecorder, RecorderBase, TraceEvent


class _AlwaysFailFlushRecorder(RecorderBase):
    def _flush_events_internal(self, events: tuple[TraceEvent, ...] | list[TraceEvent]) -> None:
        raise RuntimeError("flush boom")


class _Response:
    def __init__(self, status_code: int) -> None:
        self.status_code = status_code

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            error = requests.HTTPError(f"status {self.status_code}")
            error.response = self
            raise error


class _FakeHTTPSession:
    def __init__(self, outcomes: list[object]) -> None:
        self._outcomes = list(outcomes)
        self.calls = 0

    def post(self, url, json=None, timeout=None):  # noqa: ANN001, D401
        _ = url, json, timeout
        self.calls += 1
        outcome = self._outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome


@pytest.mark.fast
def test_flush_failure_keeps_pending_events() -> None:
    recorder = _AlwaysFailFlushRecorder(
        run_id="commit-semantics",
        min_flush_events=100,
        min_flush_seconds=10_000.0,
    )
    recorder.record_event("runtime_ready", {"ok": True}, sample_id="sample-1")

    with pytest.raises(RuntimeError, match="flush boom"):
        recorder.flush_events()

    pending = recorder.pending_events()

    assert len(pending) == 1
    assert pending[0].event == "runtime_ready"
    assert recorder._written == 0


@pytest.mark.fast
def test_http_recorder_retries_with_capped_backoff_and_jitter() -> None:
    fallback = InMemoryRecorder(run_id="http-retry-fallback")
    recorder = HTTPRecorder(
        run_id="http-retry",
        url="http://example.test/events",
        fallback=fallback,
        max_retries=2,
        base_retry_delay_ms=50,
        max_retry_delay_ms=75,
        min_flush_events=100,
        min_flush_seconds=10_000.0,
    )
    recorder._session = _FakeHTTPSession([_Response(503), _Response(503), _Response(200)])
    sleeps: list[float] = []
    recorder._sleep = sleeps.append
    recorder._random_uniform = lambda low, high: high

    recorder.write_events(
        [
            TraceEvent(
                run_id="http-retry",
                event_id=0,
                event="runtime_ready",
                payload={"ok": True},
                sample_id="sample-1",
                created_at=0.0,
            )
        ]
    )

    assert recorder._session.calls == 3
    assert sleeps == [0.05, 0.075]
    assert fallback.buffered_events() == []
    assert recorder._requests_failed == 0


@pytest.mark.fast
def test_http_recorder_does_not_retry_non_retryable_http_errors() -> None:
    fallback = InMemoryRecorder(run_id="http-no-retry-fallback")
    recorder = HTTPRecorder(
        run_id="http-no-retry",
        url="http://example.test/events",
        fallback=fallback,
        max_retries=2,
        min_flush_events=100,
        min_flush_seconds=10_000.0,
    )
    recorder._session = _FakeHTTPSession([_Response(400)])
    sleeps: list[float] = []
    recorder._sleep = sleeps.append

    recorder.write_events(
        [
            TraceEvent(
                run_id="http-no-retry",
                event_id=0,
                event="runtime_ready",
                payload={"ok": True},
                sample_id="sample-1",
                created_at=0.0,
            )
        ]
    )

    buffered = fallback.buffered_events()

    assert recorder._session.calls == 1
    assert sleeps == []
    assert recorder._requests_failed == 1
    assert len(buffered) == 1
    assert buffered[0]["event"] == "runtime_ready"
