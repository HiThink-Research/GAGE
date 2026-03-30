from __future__ import annotations

from contextlib import contextmanager

import pytest

from gage_eval.pipeline.steps.judge import JudgeStep


class _ErrorRole:
    def invoke(self, payload, trace):
        return {"error": "judge boom"}


class _RoleManager:
    @contextmanager
    def borrow_role(self, adapter_id, *, execution_context=None):
        yield _ErrorRole()


def test_judge_step_raises_on_backend_error(mock_trace) -> None:
    step = JudgeStep("judge_main")
    payload = {
        "sample": {"id": "sample-1"},
        "model_output": {"answer": "x"},
        "trace": mock_trace,
    }

    with pytest.raises(RuntimeError, match="judge backend returned error: judge boom"):
        step.execute(payload, _RoleManager(), mock_trace)

    events = [item["event"] for item in mock_trace.events]
    assert "judge_error" in events
    assert "judge_end" not in events
