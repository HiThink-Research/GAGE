from __future__ import annotations

from typing import Any, Dict

from gage_eval.evaluation.sample_envelope import (
    ensure_predict_result_slot,
    get_arena_trace,
    set_arena_trace,
)
from gage_eval.evaluation.task_planner import StepExecutionContext
from gage_eval.observability.trace import ObservabilityTrace


def test_sample_envelope_arena_trace_helpers_normalize_predict_result() -> None:
    sample: Dict[str, Any] = {"predict_result": {"unexpected": True}}
    slot = ensure_predict_result_slot(sample, index=0)
    assert isinstance(slot, dict)
    assert isinstance(sample["predict_result"], list)
    assert len(sample["predict_result"]) == 1

    trace_payload = {"schema": "gage.trace/v1", "steps": [{"step_index": 1}]}
    set_arena_trace(sample, trace_payload, index=0)
    assert sample["predict_result"][0]["arena_trace"]["schema"] == "gage.trace/v1"
    loaded = get_arena_trace(sample, index=0)
    assert loaded is not None
    assert loaded["steps"][0]["step_index"] == 1


def test_execute_arena_writes_arena_trace_to_predict_result_zero() -> None:
    class _DummySupport:
        def execute(self, *args, **kwargs):  # noqa: ANN002, ANN003
            return None

    class _DummyInference:
        def execute(self, *args, **kwargs):  # noqa: ANN002, ANN003
            return {}

    class _DummyArena:
        def execute(self, *args, **kwargs):  # noqa: ANN002, ANN003
            return {
                "result": "draw",
                "arena_trace": {
                    "schema": "gage.trace/v1",
                    "steps": [{"step_index": 2, "trace_state": "done"}],
                },
            }

    class _DummyJudge:
        def execute(self, *args, **kwargs):  # noqa: ANN002, ANN003
            return {}

    sample: Dict[str, Any] = {"id": "sample_1", "predict_result": [], "eval_result": {}}
    context = StepExecutionContext(
        sample=sample,
        support=_DummySupport(),
        inference=_DummyInference(),
        arena=_DummyArena(),
        judge=_DummyJudge(),
        auto_eval_step=None,
        trace=ObservabilityTrace(run_id="arena-trace-test"),
        role_manager=object(),
    )

    context.execute_arena()

    assert isinstance(sample["predict_result"], list)
    assert sample["predict_result"]
    trace_payload = sample["predict_result"][0].get("arena_trace")
    assert isinstance(trace_payload, dict)
    assert trace_payload["steps"][0]["step_index"] == 2

