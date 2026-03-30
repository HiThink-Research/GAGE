from __future__ import annotations

import pytest

from gage_eval.evaluation.task_planner import TaskPlanner
from gage_eval.observability.trace import ObservabilityTrace


@pytest.mark.fast
def test_task_planner_reuses_task_scope_bundle_across_plans_and_contexts() -> None:
    planner = TaskPlanner()
    planner.configure_custom_steps(
        (
            {"step": "support", "adapter_id": "helper"},
            {"step": "inference", "adapter_id": "dut"},
        )
    )

    plan_one = planner.prepare_plan({"id": "sample-1"})
    plan_two = planner.prepare_plan({"id": "sample-2"})

    assert plan_one.step_bundle is not None
    assert plan_one.step_bundle is plan_two.step_bundle

    ctx_one = plan_one.create_context(
        {"id": "sample-1"},
        ObservabilityTrace(run_id="ctx-one"),
        object(),
    )
    ctx_two = plan_two.create_context(
        {"id": "sample-2"},
        ObservabilityTrace(run_id="ctx-two"),
        object(),
    )

    assert ctx_one.support is ctx_two.support
    assert ctx_one.inference is ctx_two.inference
    assert ctx_one.auto_eval_step is ctx_two.auto_eval_step is None


@pytest.mark.fast
def test_task_planner_refreshes_task_scope_bundle_after_metric_registration() -> None:
    planner = TaskPlanner()
    planner.configure_custom_steps(
        (
            {"step": "inference", "adapter_id": "dut"},
            {"step": "auto_eval"},
        )
    )

    plan_before = planner.prepare_plan({"id": "sample-before"})
    assert plan_before.auto_eval_enabled is True
    assert plan_before.step_bundle is not None
    assert plan_before.step_bundle.auto_eval_step is None

    planner.configure_metrics(())
    plan_after = planner.prepare_plan({"id": "sample-after"})

    assert plan_after.step_bundle is not None
    assert plan_after.step_bundle is not plan_before.step_bundle
    assert plan_after.auto_eval_step is planner.get_auto_eval_step()
    assert plan_after.step_bundle.auto_eval_step is planner.get_auto_eval_step()

    ctx = plan_after.create_context(
        {"id": "sample-after"},
        ObservabilityTrace(run_id="ctx-after"),
        object(),
    )
    assert ctx.auto_eval_step is planner.get_auto_eval_step()
