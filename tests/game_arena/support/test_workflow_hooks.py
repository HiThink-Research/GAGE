from __future__ import annotations

import importlib
import sys

import pytest

from gage_eval.role.arena.support.context import SupportContext
from gage_eval.role.arena.support.hooks import SupportHook
from gage_eval.role.arena.support.registry import SupportWorkflowRegistry
from gage_eval.role.arena.support.workflow import GameSupportWorkflow
from gage_eval.registry import import_asset_from_manifest, registry


class _RecordingUnit:
    def __init__(self, label: str) -> None:
        self.label = label

    def invoke(self, context: SupportContext) -> SupportContext:
        context.unit_trace.append(self.label)
        return context


def test_support_hook_lifecycle_order_is_stable() -> None:
    assert list(SupportHook) == [
        SupportHook.AFTER_OBSERVE,
        SupportHook.BEFORE_DECIDE,
        SupportHook.AFTER_DECIDE,
        SupportHook.BEFORE_APPLY,
        SupportHook.AFTER_APPLY,
        SupportHook.ON_FINALIZE,
    ]


def test_support_hook_preserves_string_enum_semantics() -> None:
    assert isinstance(SupportHook.BEFORE_APPLY, str)
    assert SupportHook.BEFORE_APPLY == "before_apply"
    assert str(SupportHook.BEFORE_APPLY) == "before_apply"
    assert SupportHook("before_apply") is SupportHook.BEFORE_APPLY


def test_support_workflow_runs_only_units_for_selected_hook() -> None:
    workflow = GameSupportWorkflow(
        workflow_id="test",
        units_by_hook={
            SupportHook.AFTER_OBSERVE: [_RecordingUnit("after_observe")],
            SupportHook.BEFORE_DECIDE: [_RecordingUnit("before_decide")],
        },
    )

    context = SupportContext()
    result = workflow.run(SupportHook.BEFORE_DECIDE, context)

    assert result is context
    assert result.unit_trace == ["before_decide"]


def test_support_workflow_continue_degrade_policy_records_error_and_keeps_running() -> None:
    class _FailingUnit:
        def invoke(self, context: SupportContext) -> SupportContext:
            raise RuntimeError("support unit failed")

    workflow = GameSupportWorkflow(
        workflow_id="degrade",
        metadata={"workflow_kind": "support"},
        degrade_policy="continue",
        units_by_hook={
            SupportHook.BEFORE_APPLY: [_FailingUnit(), _RecordingUnit("after-failure")]
        },
    )

    context = SupportContext(payload={"action": "A1"}, state={})
    result = workflow.run(SupportHook.BEFORE_APPLY, context)

    assert result is context
    assert result.unit_trace == ["after-failure"]
    assert workflow.metadata["workflow_kind"] == "support"
    assert result.state["support_errors"][0]["error_code"] == "support_workflow_failure"
    assert result.state["support_errors"][0]["hook"] == "before_apply"


def test_support_workflow_registry_builds_noop_workflow() -> None:
    workflow = SupportWorkflowRegistry().build("noop_support_v1")

    assert workflow.workflow_id == "noop_support_v1"
    assert workflow.units_by_hook == {}


def test_support_registry_module_does_not_eager_import_specs() -> None:
    registry_module = importlib.import_module("gage_eval.role.arena.support.registry")
    workflow = registry_module.SupportWorkflowRegistry().build("arena/default")
    context = SupportContext(payload={"action": "1"})
    result = workflow.run(SupportHook.BEFORE_APPLY, context)

    assert workflow.workflow_id == "arena/default"
    assert result is context
    assert result.payload["action"] == "1"


def test_support_workflow_default_builds_after_support_specs_import() -> None:
    registry_module = importlib.import_module("gage_eval.role.arena.support.registry")
    importlib.import_module("gage_eval.role.arena.support.specs")

    workflow = registry_module.SupportWorkflowRegistry().build("arena/default")
    context = SupportContext(payload={"action": "2"})
    result = workflow.run(SupportHook.BEFORE_APPLY, context)

    assert workflow.workflow_id == "arena/default"
    assert result.payload["action"] == "2"


def test_support_workflow_default_bootstraps_from_manifest() -> None:
    clone = registry.clone()
    clone._entries["support_workflows"].pop("arena/default", None)
    clone._objects["support_workflows"].pop("arena/default", None)

    with registry.route_to(clone):
        report = import_asset_from_manifest(
            "support_workflows",
            "arena/default",
            registry=clone,
            source="unit-test",
        )
        assert report.ok
        workflow = SupportWorkflowRegistry().build("arena/default")

    context = SupportContext(payload={"action": "3"})
    result = workflow.run(SupportHook.BEFORE_APPLY, context)

    assert workflow.workflow_id == "arena/default"
    assert result.payload["action"] == "3"


def test_support_workflow_registry_materializes_clone_local_override_for_default_unit() -> None:
    support_specs = importlib.import_module("gage_eval.role.arena.support.specs")
    registry_module = importlib.import_module("gage_eval.role.arena.support.registry")

    clone = registry.clone()
    clone.register(
        "support_units",
        "arena/default",
        support_specs.SupportUnitSpec(
            unit_id="arena/default",
            impl="continuous_action_shaping",
            defaults={"low": -2.0, "high": 2.0},
        ),
        desc="Clone-local override for the default support unit",
    )

    workflow = registry_module.SupportWorkflowRegistry(registry_view=clone).build(
        "arena/default"
    )

    assert SupportHook.BEFORE_APPLY in workflow.units_by_hook
    context = SupportContext(payload={"action": [3.0, -4.0]})
    result = workflow.run(SupportHook.BEFORE_APPLY, context)

    assert result.payload["action"] == [2.0, -2.0]


def test_support_workflow_registry_materializes_metadata_and_unit_kind() -> None:
    support_specs = importlib.import_module("gage_eval.role.arena.support.specs")
    registry_module = importlib.import_module("gage_eval.role.arena.support.registry")

    clone = registry.clone()
    clone.register(
        "support_units",
        "test/shaping",
        support_specs.SupportUnitSpec(
            unit_id="test/shaping",
            impl="continuous_action_shaping",
            unit_kind="execution_support",
            metadata={"slot": "fast_loop"},
            defaults={"low": -1.0, "high": 1.0},
        ),
        desc="Support unit with explicit kind metadata",
    )
    clone.register(
        "support_workflows",
        "test/workflow",
        support_specs.SupportWorkflowSpec(
            workflow_id="test/workflow",
            hook_bindings={"before_apply": ("test/shaping",)},
            metadata={"workflow_kind": "slow_fast_loop"},
            degrade_policy="continue",
        ),
        desc="Support workflow with explicit metadata",
    )

    workflow = registry_module.SupportWorkflowRegistry(registry_view=clone).build(
        "test/workflow"
    )

    assert workflow.metadata["workflow_kind"] == "slow_fast_loop"
    assert workflow.degrade_policy == "continue"
    assert workflow.unit_metadata[SupportHook.BEFORE_APPLY][0].unit_id == "test/shaping"
    assert workflow.unit_metadata[SupportHook.BEFORE_APPLY][0].unit_kind == "execution_support"
    assert workflow.unit_metadata[SupportHook.BEFORE_APPLY][0].metadata == {"slot": "fast_loop"}


def test_support_workflow_registry_rejects_unsupported_spec_like_assets() -> None:
    class _SpecLike:
        workflow_id = "arena_v2/unsupported"
        unit_ids = ("u1",)

    clone = registry.clone()
    with registry.route_to(clone):
        registry.register(
            "support_workflows",
            "arena_v2/unsupported",
            _SpecLike(),
            desc="Unsupported support workflow asset used in tests",
        )
        with pytest.raises(KeyError, match="support_units:u1"):
            SupportWorkflowRegistry().build("arena_v2/unsupported")
