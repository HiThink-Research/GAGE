from __future__ import annotations

from types import SimpleNamespace

import pytest

from gage_eval.evaluation.sample_loop import SampleLoop
from gage_eval.observability.trace import ObservabilityTrace
from gage_eval.reporting.recorders import InMemoryRecorder


class _Session:
    def __init__(self) -> None:
        self.calls = []

    def execute_support_step(self, step) -> None:
        self.calls.append(("support", step))

    def execute_inference(self) -> None:
        self.calls.append(("inference",))

    def execute_arena(self) -> None:
        self.calls.append(("arena",))

    def execute_judge(self) -> None:
        self.calls.append(("judge",))

    def execute_auto_eval(self, sample_id: str) -> None:
        self.calls.append(("auto_eval", sample_id))


class _FailingSession(_Session):
    def execute_inference(self) -> None:
        raise RuntimeError("boom")


class _RoleManagerStub:
    def __init__(self, adapters):
        self._adapters = adapters

    def get_adapter(self, adapter_id: str):
        return self._adapters.get(adapter_id)


def _trace() -> ObservabilityTrace:
    return ObservabilityTrace(recorder=InMemoryRecorder(run_id="step-dispatch"))


def _plan(**kwargs):
    defaults = {
        "metadata": {"task_id": "task-1"},
        "inference_role": None,
        "arena_role": None,
        "judge_role": None,
    }
    defaults.update(kwargs)
    return SimpleNamespace(**defaults)


@pytest.mark.fast
def test_sample_loop_rejects_unregistered_steps() -> None:
    sample_loop = SampleLoop([], concurrency=1)
    trace = _trace()

    with pytest.raises(ValueError, match="Unsupported configured step 'hook'"):
        sample_loop._dispatch_step(
            _Session(),
            {"step": "hook"},
            _plan(),
            _RoleManagerStub({}),
            trace,
            "sample-1",
        )

    failed = [event for event in trace.events if event["event"] == "step_execution_failed"]
    assert failed and failed[-1]["payload"]["step_type"] == "hook"


@pytest.mark.fast
def test_sample_loop_rejects_global_steps_in_sample_execution() -> None:
    sample_loop = SampleLoop([], concurrency=1)
    trace = _trace()

    with pytest.raises(ValueError, match="global and cannot run inside sample execution"):
        sample_loop._dispatch_step(
            _Session(),
            {"step": "report"},
            _plan(),
            _RoleManagerStub({}),
            trace,
            "sample-1",
        )


@pytest.mark.fast
def test_sample_loop_fails_fast_when_required_adapter_binding_is_missing() -> None:
    sample_loop = SampleLoop([], concurrency=1)
    trace = _trace()

    with pytest.raises(RuntimeError, match="requires a resolved adapter_id"):
        sample_loop._dispatch_step(
            _Session(),
            {"step": "inference"},
            _plan(inference_role=None),
            _RoleManagerStub({}),
            trace,
            "sample-1",
        )

    failed = [event for event in trace.events if event["event"] == "step_execution_failed"]
    assert failed and failed[-1]["payload"]["error_type"] == "RuntimeError"


@pytest.mark.fast
def test_sample_loop_emits_started_and_completed_events_for_successful_dispatch() -> None:
    sample_loop = SampleLoop([], concurrency=1)
    trace = _trace()
    session = _Session()

    sample_loop._dispatch_step(
        session,
        {"step": "inference", "adapter_id": "dut"},
        _plan(inference_role="dut"),
        _RoleManagerStub({"dut": object()}),
        trace,
        "sample-1",
    )

    assert session.calls == [("inference",)]
    events = [event["event"] for event in trace.events]
    assert "step_execution_started" in events
    assert "step_execution_completed" in events


@pytest.mark.fast
def test_sample_loop_surfaces_step_handler_errors() -> None:
    sample_loop = SampleLoop([], concurrency=1)
    trace = _trace()

    with pytest.raises(RuntimeError, match="boom"):
        sample_loop._dispatch_step(
            _FailingSession(),
            {"step": "inference", "adapter_id": "dut"},
            _plan(inference_role="dut"),
            _RoleManagerStub({"dut": object()}),
            trace,
            "sample-1",
        )

    failed = [event for event in trace.events if event["event"] == "step_execution_failed"]
    assert failed and failed[-1]["payload"]["error"] == "boom"
