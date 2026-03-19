from __future__ import annotations

import pytest

from gage_eval.evaluation.task_planner import TaskPlanner


@pytest.mark.fast
def test_task_planner_preserves_ordered_steps() -> None:
    planner = TaskPlanner()
    steps = (
        {"step": "support", "adapter_id": "helper-a"},
        {"step": "support", "adapter_id": "helper-b"},
        {"step": "inference", "adapter_id": "dut"},
    )

    planner.configure_custom_steps(steps)
    plan = planner.prepare_plan({"id": "sample-1"})

    assert tuple(plan.steps) == steps
    assert tuple(plan.support_steps) == steps[:2]
    assert plan.inference_role == "dut"


@pytest.mark.fast
def test_task_planner_rejects_duplicate_singleton_steps() -> None:
    planner = TaskPlanner()

    with pytest.raises(ValueError, match="only one occurrence is supported"):
        planner.configure_custom_steps(
            (
                {"step": "inference", "adapter_id": "dut-a"},
                {"step": "inference", "adapter_id": "dut-b"},
            )
        )


@pytest.mark.fast
def test_task_planner_requires_non_inference_adapter_binding() -> None:
    planner = TaskPlanner()

    with pytest.raises(ValueError, match="step 'support' requires adapter_id"):
        planner.configure_custom_steps(({"step": "support"},))


@pytest.mark.fast
def test_task_planner_requires_prerequisite_step() -> None:
    planner = TaskPlanner()

    with pytest.raises(ValueError, match="requires a preceding inference/arena/judge step"):
        planner.configure_custom_steps(
            (
                {"step": "auto_eval"},
                {"step": "inference", "adapter_id": "dut"},
            )
        )


@pytest.mark.fast
def test_task_planner_accepts_role_ref_binding_for_sample_steps() -> None:
    planner = TaskPlanner()
    steps = (
        {"step": "support", "role_ref": "helper"},
        {"step": "inference", "role_ref": "dut"},
    )

    planner.configure_custom_steps(steps)
    plan = planner.prepare_plan({"id": "sample-1"})

    assert tuple(plan.support_steps) == steps[:1]
    assert plan.inference_role == "dut"
