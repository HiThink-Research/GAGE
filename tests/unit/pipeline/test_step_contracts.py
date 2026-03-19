from __future__ import annotations

import pytest

from gage_eval.pipeline.step_contracts import (
    clear_step_contract_catalog_cache,
    collect_step_sequence_issues,
    get_step_adapter_id,
    get_step_contract_catalog,
    get_step_type,
)
from gage_eval.pipeline.steps.base import StepKind


@pytest.mark.fast
def test_catalog_exposes_builtin_sample_and_global_steps() -> None:
    clear_step_contract_catalog_cache()

    catalog = get_step_contract_catalog()

    inference = catalog.require("inference")
    report = catalog.require("report")
    support = catalog.require("support")

    assert inference.step_kind is StepKind.SAMPLE
    assert inference.requires_adapter is True
    assert inference.executor_name == "execute_inference"
    assert inference.allow_multiple is False
    assert report.step_kind is StepKind.GLOBAL
    assert report.executor_name is None
    assert support.allow_multiple is True


@pytest.mark.fast
def test_catalog_keeps_unregistered_hook_out_of_contracts() -> None:
    clear_step_contract_catalog_cache()

    catalog = get_step_contract_catalog()

    assert catalog.get("hook") is None


@pytest.mark.fast
def test_step_helpers_support_dict_and_dataclass_like_objects() -> None:
    class _Step:
        step_type = "inference"
        adapter_id = "dut"

    assert get_step_type({"step": "support", "adapter_id": "helper"}) == "support"
    assert get_step_adapter_id({"step": "support", "role_ref": "helper"}) == "helper"
    assert get_step_type(_Step()) == "inference"
    assert get_step_adapter_id(_Step()) == "dut"


@pytest.mark.fast
def test_collect_step_sequence_issues_accepts_role_ref_binding() -> None:
    issues = collect_step_sequence_issues(
        [{"step": "support", "role_ref": "helper"}],
        owner_label="custom step",
        adapter_ids=("helper",),
    )

    assert issues == ()


@pytest.mark.fast
def test_collect_step_sequence_issues_reports_missing_adapter() -> None:
    issues = collect_step_sequence_issues(
        [{"step": "support"}],
        owner_label="custom step",
    )

    assert len(issues) == 1
    assert issues[0].code == "missing_adapter"
    assert "requires adapter_id" in issues[0].message


@pytest.mark.fast
def test_collect_step_sequence_issues_reports_missing_prerequisite() -> None:
    issues = collect_step_sequence_issues(
        [
            {"step": "auto_eval"},
            {"step": "inference", "adapter_id": "dut"},
        ],
        owner_label="custom step",
        adapter_ids=("dut",),
    )

    assert len(issues) == 1
    assert issues[0].code == "missing_prerequisite"
    assert "requires a preceding inference/arena/judge step" in issues[0].message
