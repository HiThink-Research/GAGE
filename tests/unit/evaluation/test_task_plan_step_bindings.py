from __future__ import annotations

import pytest

from gage_eval.config.pipeline_config import CustomPipelineStep, RoleAdapterSpec
from gage_eval.evaluation.task_plan import _infer_step_bindings, _validate_task_steps


def _role_map(**adapters: str):
    return {
        adapter_id: RoleAdapterSpec(adapter_id=adapter_id, role_type=role_type)
        for adapter_id, role_type in adapters.items()
    }


@pytest.mark.fast
def test_infer_step_bindings_resolves_unique_dut_inference() -> None:
    role_map = _role_map(dut="dut_model")
    steps = (CustomPipelineStep(step_type="inference"),)

    resolved = _infer_step_bindings(steps, role_map, task_id="task-1")

    assert resolved[0].adapter_id == "dut"


@pytest.mark.fast
def test_infer_step_bindings_rejects_ambiguous_inference_binding() -> None:
    role_map = _role_map(dut_a="dut_model", dut_b="dut_model")
    steps = (CustomPipelineStep(step_type="inference"),)

    with pytest.raises(ValueError, match="multiple DUT role adapters"):
        _infer_step_bindings(steps, role_map, task_id="task-1")


@pytest.mark.fast
def test_validate_task_steps_requires_non_inference_adapter_binding() -> None:
    role_map = _role_map(helper="helper_model")
    steps = (CustomPipelineStep(step_type="support"),)

    with pytest.raises(ValueError, match="step 'support' requires adapter_id"):
        _validate_task_steps(steps, role_map, task_id="task-1")


@pytest.mark.fast
def test_validate_task_steps_rejects_global_report_step() -> None:
    role_map = _role_map(dut="dut_model")
    steps = (CustomPipelineStep(step_type="report"),)

    with pytest.raises(ValueError, match="global step 'report'"):
        _validate_task_steps(steps, role_map, task_id="task-1")


@pytest.mark.fast
def test_validate_task_steps_preserves_unknown_adapter_key_error() -> None:
    role_map = _role_map(dut="dut_model")
    steps = (CustomPipelineStep(step_type="support", adapter_id="missing"),)

    with pytest.raises(KeyError, match="references unknown role adapter 'missing'"):
        _validate_task_steps(steps, role_map, task_id="task-1")


@pytest.mark.fast
def test_validate_task_steps_requires_prerequisite_step() -> None:
    role_map = _role_map(dut="dut_model")
    steps = (
        CustomPipelineStep(step_type="auto_eval"),
        CustomPipelineStep(step_type="inference", adapter_id="dut"),
    )

    with pytest.raises(ValueError, match="requires a preceding inference/arena/judge step"):
        _validate_task_steps(steps, role_map, task_id="task-1")
