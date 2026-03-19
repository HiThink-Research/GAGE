from __future__ import annotations

from contextlib import contextmanager
from typing import Iterator

import pytest

from gage_eval.observability.trace import ObservabilityTrace
from gage_eval.pipeline.steps.inference import InferenceStep
from gage_eval.pipeline.steps.judge import JudgeStep
from gage_eval.reporting.recorders import InMemoryRecorder


class _StubRole:
    def __init__(self, output: dict) -> None:
        self._output = output

    def invoke(self, _payload: dict, _trace: ObservabilityTrace) -> dict:
        return dict(self._output)


class _StubRoleManager:
    def __init__(self, output: dict) -> None:
        self._output = output

    @contextmanager
    def borrow_role(self, _adapter_id: str) -> Iterator[_StubRole]:
        yield _StubRole(self._output)


def _build_trace(run_id: str) -> ObservabilityTrace:
    recorder = InMemoryRecorder(run_id=run_id)
    return ObservabilityTrace(recorder=recorder, run_id=run_id)


@pytest.mark.fast
def test_inference_step_raises_runtime_error_for_backend_error() -> None:
    trace = _build_trace("inference-backend-error")
    step = InferenceStep(adapter_id="dut-adapter")

    with pytest.raises(RuntimeError, match="inference backend returned error: boom"):
        step.execute({"id": "sample-1"}, _StubRoleManager({"error": "boom"}), trace)

    assert [event["event"] for event in trace.events] == ["inference_start", "inference_error"]
    assert trace.events[1]["payload"] == {
        "adapter_id": "dut-adapter",
        "error_type": "backend_error",
        "failure_reason": "backend_returned_error",
        "error": "boom",
    }


@pytest.mark.fast
def test_judge_step_raises_runtime_error_for_backend_error() -> None:
    trace = _build_trace("judge-backend-error")
    step = JudgeStep(adapter_id="judge-adapter")

    with pytest.raises(RuntimeError, match="judge backend returned error: boom"):
        step.execute({"sample": {"id": "sample-1"}}, _StubRoleManager({"error": "boom"}), trace)

    assert [event["event"] for event in trace.events] == ["judge_start", "judge_error"]
    assert trace.events[1]["payload"] == {
        "adapter_id": "judge-adapter",
        "error_type": "backend_error",
        "failure_reason": "backend_returned_error",
        "error": "boom",
    }
