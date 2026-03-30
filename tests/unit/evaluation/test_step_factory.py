from __future__ import annotations

import pytest

import gage_eval.evaluation.step_factory as step_factory_module
from gage_eval.evaluation.step_factory import StepFactory
from gage_eval.pipeline.steps.auto_eval import AutoEvalStep


@pytest.mark.fast
def test_step_factory_freezes_task_scope_steps_and_snapshots_support_config() -> None:
    support_steps = [
        {
            "step": "support",
            "adapter_id": "helper",
            "params": {"tool": "search"},
        }
    ]
    auto_eval_step = AutoEvalStep(metric_specs=())

    bundle = StepFactory().build_bundle(
        support_steps=support_steps,
        inference_role="dut",
        arena_role="arena",
        judge_role="judge",
        auto_eval_step=auto_eval_step,
    )

    support_steps[0]["adapter_id"] = "mutated"
    support_steps[0]["params"]["tool"] = "mutated"

    assert bundle.support is not None
    assert bundle.inference is not None
    assert bundle.arena is not None
    assert bundle.judge is not None
    assert bundle.support.is_frozen is True
    assert bundle.inference.is_frozen is True
    assert bundle.arena.is_frozen is True
    assert bundle.judge.is_frozen is True
    assert bundle.support._steps[0]["adapter_id"] == "helper"
    assert bundle.support._steps[0]["params"]["tool"] == "search"
    assert bundle.auto_eval_step is auto_eval_step
    assert bundle.auto_eval_step is not None
    assert bundle.auto_eval_step.is_frozen is False

    with pytest.raises(AttributeError, match="task-scope frozen"):
        bundle.inference.execution_state = {}


@pytest.mark.fast
def test_step_factory_fails_fast_when_step_construction_errors(monkeypatch: pytest.MonkeyPatch) -> None:
    class BrokenInferenceStep:
        def __init__(self, adapter_id: str) -> None:
            raise RuntimeError(f"broken constructor for {adapter_id}")

    monkeypatch.setattr(step_factory_module, "InferenceStep", BrokenInferenceStep)

    with pytest.raises(
        ValueError,
        match="Failed to build task-scoped step 'inference': broken constructor for dut",
    ):
        StepFactory().build_bundle(
            support_steps=(),
            inference_role="dut",
            arena_role=None,
            judge_role=None,
            auto_eval_step=None,
        )
