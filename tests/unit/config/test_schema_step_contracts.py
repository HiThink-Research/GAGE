from __future__ import annotations

import pytest

from gage_eval.config.schema import SchemaValidationError, normalize_pipeline_payload
from gage_eval.registry import registry


def _base_payload() -> dict:
    return {
        "datasets": [{"dataset_id": "ds"}],
        "role_adapters": [
            {"adapter_id": "dut", "role_type": "dut_model"},
            {"adapter_id": "judge", "role_type": "judge_model"},
        ],
        "tasks": [{"task_id": "task-1", "dataset_id": "ds"}],
    }


@pytest.mark.fast
def test_schema_rejects_unregistered_step() -> None:
    payload = _base_payload()
    payload["custom"] = {"steps": [{"step": "hook"}]}

    with pytest.raises(SchemaValidationError, match="unregistered step 'hook'"):
        normalize_pipeline_payload(payload)


@pytest.mark.fast
def test_schema_rejects_global_report_step_inside_sample_pipeline() -> None:
    payload = _base_payload()
    payload["custom"] = {"steps": [{"step": "report"}]}

    with pytest.raises(SchemaValidationError, match="global step 'report'"):
        normalize_pipeline_payload(payload)


@pytest.mark.fast
def test_schema_requires_adapter_for_non_inference_sample_steps() -> None:
    payload = _base_payload()
    payload["custom"] = {"steps": [{"step": "support"}]}

    with pytest.raises(SchemaValidationError, match="step 'support' requires adapter_id"):
        normalize_pipeline_payload(payload)


@pytest.mark.fast
def test_schema_requires_auto_eval_prerequisite_step() -> None:
    payload = _base_payload()
    payload["custom"] = {
        "steps": [
            {"step": "auto_eval"},
            {"step": "inference", "adapter_id": "dut"},
        ]
    }

    with pytest.raises(SchemaValidationError, match="step 'auto_eval' requires a preceding inference/arena/judge step"):
        normalize_pipeline_payload(payload)


@pytest.mark.fast
def test_schema_allows_inference_without_adapter_for_later_inference_binding() -> None:
    payload = _base_payload()
    payload["custom"] = {"steps": [{"step": "inference"}]}

    normalized = normalize_pipeline_payload(payload)

    assert normalized["custom"]["steps"][0]["step"] == "inference"


@pytest.mark.fast
def test_schema_accepts_role_ref_binding_for_sample_steps() -> None:
    payload = _base_payload()
    payload["custom"] = {"steps": [{"step": "support", "role_ref": "judge"}]}

    normalized = normalize_pipeline_payload(payload)

    assert normalized["custom"]["steps"][0]["role_ref"] == "judge"


@pytest.mark.fast
def test_schema_accepts_builtin_prompt_reference_without_forcing_global_registration() -> None:
    payload = _base_payload()
    payload["custom"] = {"steps": [{"step": "inference"}]}
    payload["role_adapters"][0]["prompt_id"] = "dut/general@v1"

    prompt_was_present = True
    try:
        registry.get("prompts", "dut/general@v1")
    except KeyError:
        prompt_was_present = False

    normalized = normalize_pipeline_payload(payload)

    assert normalized["role_adapters"][0]["prompt_id"] == "dut/general@v1"
    if not prompt_was_present:
        with pytest.raises(KeyError):
            registry.get("prompts", "dut/general@v1")
