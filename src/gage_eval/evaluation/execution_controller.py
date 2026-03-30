"""Execution control primitives for sample and metric concurrency."""

from __future__ import annotations

from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from enum import Enum
import threading
from typing import Any, Callable, Dict, Optional

from gage_eval.observability.trace import ObservabilityTrace


class FailurePolicy(str, Enum):
    """Supported sample loop failure behaviors."""

    FAIL_FAST = "fail_fast"
    GRACEFUL = "graceful"
    BEST_EFFORT = "best_effort"

    @classmethod
    def parse(cls, value: object | None) -> "FailurePolicy":
        """Return a normalized failure policy enum."""

        if isinstance(value, FailurePolicy):
            return value
        normalized = str(value or cls.FAIL_FAST.value).strip().lower()
        for policy in cls:
            if policy.value == normalized:
                return policy
        raise ValueError(
            f"Unsupported failure_policy '{value}'. Expected one of: "
            + ", ".join(policy.value for policy in cls)
        )


@dataclass(frozen=True)
class SampleLoopOutcome:
    """Structured result describing sample-loop completion or abort."""

    status: str
    failure_policy: str
    processed_samples: int
    failed_sample_id: Optional[str]
    error_type: Optional[str]
    error_message: Optional[str]
    cancelled_samples: int
    completed_after_first_error: int
    sample_workers: int
    metric_workers: int
    max_inflight: int
    legacy_ff_mode: bool = False
    metric_inline_fallbacks: int = 0

    def to_summary_payload(self) -> Dict[str, Any]:
        """Return a JSON-serializable execution summary payload."""

        return {
            "status": self.status,
            "failure_policy": self.failure_policy,
            "processed_samples": self.processed_samples,
            "failed_sample_id": self.failed_sample_id,
            "error_type": self.error_type,
            "error_message": self.error_message,
            "cancelled_samples": self.cancelled_samples,
            "completed_after_first_error": self.completed_after_first_error,
            "sample_workers": self.sample_workers,
            "metric_workers": self.metric_workers,
            "max_inflight": self.max_inflight,
            "legacy_ff_mode": self.legacy_ff_mode,
            "metric_inline_fallbacks": self.metric_inline_fallbacks,
        }


class SampleLoopExecutionError(RuntimeError):
    """Wraps the primary sample-loop failure with structured outcome data."""

    def __init__(self, outcome: SampleLoopOutcome, error: BaseException) -> None:
        self.outcome = outcome
        self.error = error
        message = str(error) or error.__class__.__name__
        super().__init__(message)
        self.__cause__ = error


class TaskExecutionController:
    """Owns task-scoped failure policy and concurrency coordination."""

    def __init__(
        self,
        *,
        sample_workers: int,
        metric_workers: int,
        failure_policy: FailurePolicy | str | None = None,
        legacy_ff_mode: bool = False,
        report_partial_on_failure: bool = True,
        inline_sample_execution: bool = False,
    ) -> None:
        self._sample_workers = max(1, int(sample_workers))
        self._metric_workers = max(0, int(metric_workers))
        self._failure_policy = FailurePolicy.parse(failure_policy)
        self._legacy_ff_mode = bool(legacy_ff_mode)
        self._report_partial_on_failure = bool(report_partial_on_failure)
        self._inline_sample_execution = bool(inline_sample_execution)
        self._sample_executor = ThreadPoolExecutor(
            max_workers=self._sample_workers,
            thread_name_prefix="gage-sample",
        )
        self._metric_executor = (
            ThreadPoolExecutor(
                max_workers=self._metric_workers,
                thread_name_prefix="gage-metric",
            )
            if self._metric_workers > 1
            else None
        )
        self._metric_capacity = (
            threading.BoundedSemaphore(self._metric_workers)
            if self._metric_workers > 1
            else None
        )
        self._lock = threading.Lock()
        self._pending_samples: Dict[Future[Any], tuple[str, threading.Event]] = {}
        self._first_error: Optional[BaseException] = None
        self._first_error_sample_id: Optional[str] = None
        self._cancelled_samples = 0
        self._completed_after_first_error = 0
        self._metric_inline_fallbacks = 0
        self._closed = False
        self._abort = threading.Event()

    @property
    def failure_policy(self) -> FailurePolicy:
        return self._failure_policy

    @property
    def sample_workers(self) -> int:
        return self._sample_workers

    @property
    def metric_workers(self) -> int:
        return self._metric_workers

    @property
    def report_partial_on_failure(self) -> bool:
        return self._report_partial_on_failure

    @property
    def first_error(self) -> Optional[BaseException]:
        return self._first_error

    def submit_sample(
        self,
        fn: Callable[..., Any],
        *args: Any,
        sample_id: str,
        **kwargs: Any,
    ) -> Future[Any]:
        """Submit one sample task to the shared sample lane."""

        if self._inline_sample_execution:
            return self._run_sample_inline(
                fn,
                *args,
                sample_id=sample_id,
                **kwargs,
            )

        started = threading.Event()

        def _runner() -> Any:
            started.set()
            try:
                return fn(*args, **kwargs)
            except BaseException as exc:
                self.record_failure(sample_id, exc)
                raise

        future = self._sample_executor.submit(_runner)
        with self._lock:
            self._pending_samples[future] = (sample_id, started)
        future.add_done_callback(self._sample_done_callback)
        return future

    def submit_metric(
        self,
        fn: Callable[..., Any],
        *args: Any,
        sample_id: Optional[str] = None,
        trace: Optional[ObservabilityTrace] = None,
        **kwargs: Any,
    ) -> Future[Any]:
        """Submit one metric task to the metric lane or execute inline."""

        if self._metric_executor is None or self._metric_capacity is None:
            return self._run_metric_inline(
                fn,
                *args,
                sample_id=sample_id,
                trace=trace,
                **kwargs,
            )
        if not self._metric_capacity.acquire(blocking=False):
            return self._run_metric_inline(
                fn,
                *args,
                sample_id=sample_id,
                trace=trace,
                **kwargs,
            )
        try:
            future = self._metric_executor.submit(fn, *args, **kwargs)
        except BaseException:
            self._metric_capacity.release()
            raise
        future.add_done_callback(lambda _: self._metric_capacity.release())
        return future

    def should_stop_submitting(self) -> bool:
        """Return whether new samples should stop being submitted."""

        return self._abort.is_set() and self._failure_policy is not FailurePolicy.BEST_EFFORT

    def record_failure(self, sample_id: Optional[str], exc: BaseException) -> None:
        """Persist the first sample failure and trigger abort when needed."""

        with self._lock:
            if self._first_error is not None:
                return
            self._first_error = exc
            self._first_error_sample_id = sample_id
            if self._failure_policy is not FailurePolicy.BEST_EFFORT:
                self._abort.set()

    def cancel_pending_samples(self) -> int:
        """Cancel sample tasks that have not started yet."""

        cancelled = 0
        with self._lock:
            handles = list(self._pending_samples.items())
        for future, (_sample_id, started) in handles:
            if started.is_set():
                continue
            if future.cancel():
                cancelled += 1
        if cancelled:
            with self._lock:
                self._cancelled_samples += cancelled
        return cancelled

    def record_queue_cancellations(self, count: int) -> None:
        """Account for queued-but-never-submitted samples after abort."""

        if count <= 0:
            return
        with self._lock:
            self._cancelled_samples += int(count)

    def snapshot(
        self,
        *,
        processed_samples: int,
        max_inflight: int,
    ) -> SampleLoopOutcome:
        """Return the current structured loop outcome."""

        with self._lock:
            error = self._first_error
            failed_sample_id = self._first_error_sample_id
            cancelled_samples = self._cancelled_samples
            completed_after_first_error = self._completed_after_first_error
            metric_inline_fallbacks = self._metric_inline_fallbacks
        if error is None:
            status = "completed"
            error_type = None
            error_message = None
        elif self._failure_policy is FailurePolicy.BEST_EFFORT:
            status = "completed_with_failures"
            error_type = error.__class__.__name__
            error_message = str(error)
        else:
            status = "aborted"
            error_type = error.__class__.__name__
            error_message = str(error)
        return SampleLoopOutcome(
            status=status,
            failure_policy=self._failure_policy.value,
            processed_samples=int(processed_samples),
            failed_sample_id=failed_sample_id,
            error_type=error_type,
            error_message=error_message,
            cancelled_samples=cancelled_samples,
            completed_after_first_error=completed_after_first_error,
            sample_workers=self._sample_workers,
            metric_workers=self._metric_workers,
            max_inflight=int(max_inflight),
            legacy_ff_mode=self._legacy_ff_mode,
            metric_inline_fallbacks=metric_inline_fallbacks,
        )

    def shutdown(self) -> None:
        """Release underlying executors exactly once."""

        with self._lock:
            if self._closed:
                return
            self._closed = True
        self._sample_executor.shutdown(wait=True, cancel_futures=False)
        if self._metric_executor is not None:
            self._metric_executor.shutdown(wait=True, cancel_futures=False)

    def _run_metric_inline(
        self,
        fn: Callable[..., Any],
        *args: Any,
        sample_id: Optional[str] = None,
        trace: Optional[ObservabilityTrace] = None,
        **kwargs: Any,
    ) -> Future[Any]:
        future: Future[Any] = Future()
        with self._lock:
            self._metric_inline_fallbacks += 1
        if trace is not None:
            trace.emit(
                "metric_lane_fallback_inline",
                {
                    "sample_id": sample_id,
                    "metric_workers": self._metric_workers,
                    "failure_policy": self._failure_policy.value,
                },
                sample_id=sample_id,
            )
        try:
            future.set_result(fn(*args, **kwargs))
        except BaseException as exc:
            future.set_exception(exc)
        return future

    def _run_sample_inline(
        self,
        fn: Callable[..., Any],
        *args: Any,
        sample_id: str,
        **kwargs: Any,
    ) -> Future[Any]:
        future: Future[Any] = Future()
        started = threading.Event()
        with self._lock:
            self._pending_samples[future] = (sample_id, started)
        future.add_done_callback(self._sample_done_callback)
        started.set()
        try:
            future.set_result(fn(*args, **kwargs))
        except BaseException as exc:
            self.record_failure(sample_id, exc)
            future.set_exception(exc)
        return future

    def _sample_done_callback(self, future: Future[Any]) -> None:
        with self._lock:
            sample_id, _started = self._pending_samples.pop(future, (None, None))
            first_error_sample_id = self._first_error_sample_id
            should_count_completion = (
                first_error_sample_id is not None
                and sample_id is not None
                and sample_id != first_error_sample_id
                and not future.cancelled()
                and future.exception() is None
            )
            if should_count_completion:
                self._completed_after_first_error += 1
