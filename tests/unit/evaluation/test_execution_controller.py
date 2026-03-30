from __future__ import annotations

import threading

import pytest

from gage_eval.evaluation.execution_controller import (
    FailurePolicy,
    SampleLoopExecutionError,
    TaskExecutionController,
)
from gage_eval.observability.trace import ObservabilityTrace
from gage_eval.reporting.recorders import InMemoryRecorder


@pytest.mark.fast
def test_fail_fast_controller_cancels_pending_samples() -> None:
    controller = TaskExecutionController(
        sample_workers=1,
        metric_workers=0,
        failure_policy=FailurePolicy.FAIL_FAST,
    )

    blocker = threading.Event()

    def wait_for_release() -> str:
        blocker.wait(timeout=5)
        return "done"

    first = controller.submit_sample(
        lambda: (_ for _ in ()).throw(RuntimeError("boom")),
        sample_id="s1",
    )
    second = controller.submit_sample(wait_for_release, sample_id="s2")

    with pytest.raises(RuntimeError, match="boom"):
        first.result()

    cancelled = controller.cancel_pending_samples()
    blocker.set()

    outcome = controller.snapshot(processed_samples=0, max_inflight=2)
    controller.shutdown()

    assert cancelled == 1
    assert second.cancelled() is True
    assert outcome.status == "aborted"
    assert outcome.failed_sample_id == "s1"
    assert outcome.cancelled_samples == 1


@pytest.mark.fast
def test_best_effort_controller_keeps_accepting_samples_after_failure() -> None:
    controller = TaskExecutionController(
        sample_workers=1,
        metric_workers=0,
        failure_policy=FailurePolicy.BEST_EFFORT,
    )

    first = controller.submit_sample(
        lambda: (_ for _ in ()).throw(RuntimeError("boom")),
        sample_id="s1",
    )
    second = controller.submit_sample(lambda: "ok", sample_id="s2")

    with pytest.raises(RuntimeError, match="boom"):
        first.result()

    assert second.result() == "ok"
    assert controller.should_stop_submitting() is False

    outcome = controller.snapshot(processed_samples=1, max_inflight=2)
    controller.shutdown()

    assert outcome.status == "completed_with_failures"
    assert outcome.completed_after_first_error >= 1


@pytest.mark.fast
def test_metric_lane_falls_back_inline_when_saturated() -> None:
    controller = TaskExecutionController(
        sample_workers=1,
        metric_workers=2,
        failure_policy=FailurePolicy.FAIL_FAST,
    )
    trace = ObservabilityTrace(recorder=InMemoryRecorder(run_id="metric-inline"))

    release = threading.Event()

    def wait_metric() -> str:
        release.wait(timeout=5)
        return "held"

    held_one = controller.submit_metric(wait_metric, sample_id="s1", trace=trace)
    held_two = controller.submit_metric(wait_metric, sample_id="s1", trace=trace)
    inline = controller.submit_metric(lambda: "inline", sample_id="s1", trace=trace)

    assert inline.result() == "inline"
    release.set()
    assert held_one.result() == "held"
    assert held_two.result() == "held"

    outcome = controller.snapshot(processed_samples=0, max_inflight=1)
    controller.shutdown()

    assert outcome.metric_inline_fallbacks >= 1
    assert any(event["event"] == "metric_lane_fallback_inline" for event in trace.events)


@pytest.mark.fast
def test_failure_policy_parse_rejects_unknown_values() -> None:
    with pytest.raises(ValueError, match="Unsupported failure_policy"):
        FailurePolicy.parse("explode")
