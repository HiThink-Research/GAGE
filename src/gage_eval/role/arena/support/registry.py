"""Support workflow registry bindings."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

from gage_eval.role.arena.support.hooks import (
    SupportDegradePolicy,
    SupportHook,
    SupportUnitKind,
)
from gage_eval.role.arena.support.units.action_shaping import (
    ContinuousActionShapingUnit,
)
from gage_eval.role.arena.support.workflow import (
    GameSupportWorkflow,
    SupportUnitRuntimeMetadata,
)
from gage_eval.registry import registry

DEFAULT_SUPPORT_WORKFLOW = SimpleNamespace(
    workflow_id="arena/default",
    unit_ids=("arena/default",),
    hook_bindings={"before_apply": ("arena/default",)},
    metadata={"workflow_kind": "game_support"},
    degrade_policy=SupportDegradePolicy.FAIL_FAST.value,
    defaults={},
)

def register_runtime_assets(*, registry_target=None) -> None:
    target = registry_target or registry
    target.register(
        "support_workflows",
        "arena/default",
        DEFAULT_SUPPORT_WORKFLOW,
        desc="Default support workflow runtime for GameArena",
    )


@registry.asset("support_workflows", "noop_support_v1", desc="No-op support workflow")
def build_noop_support() -> GameSupportWorkflow:
    return GameSupportWorkflow(workflow_id="noop_support_v1", units_by_hook={})


register_runtime_assets(registry_target=registry)


class _NoOpSupportUnit:
    def invoke(self, context):
        return context


def _supports_invoke(asset: Any) -> bool:
    return hasattr(asset, "invoke") and callable(getattr(asset, "invoke"))


def _is_spec_like(asset: Any, *, attrs: tuple[str, ...]) -> bool:
    return all(hasattr(asset, attr) for attr in attrs)


def _materialize_support_unit(
    asset: Any,
    *,
    unit_id: str,
    registry_view,
) -> object:
    if isinstance(asset, _NoOpSupportUnit):
        return asset
    if isinstance(asset, ContinuousActionShapingUnit) or _supports_invoke(asset):
        return asset

    if callable(asset):
        unit = asset()
        if isinstance(unit, ContinuousActionShapingUnit) or _supports_invoke(unit):
            return unit
        raise TypeError(
            f"Support unit builder '{unit_id}' must return "
            f"an invokable support unit (got '{type(unit).__name__}')"
        )

    if _is_spec_like(asset, attrs=("unit_id", "impl", "defaults")):
        impl = str(getattr(asset, "impl")).strip()
        defaults = getattr(asset, "defaults", {}) or {}
        if not impl or impl in {
            "noop_support_v1",
            "default",
            "placeholder://arena/support_units/default",
        }:
            return _NoOpSupportUnit()
        if impl in {
            "continuous_action_shaping",
            "placeholder://arena/support_units/continuous_action_shaping",
        }:
            low = float(defaults.get("low", float("-inf")))
            high = float(defaults.get("high", float("inf")))
            return ContinuousActionShapingUnit(low=low, high=high)
        raise KeyError(
            f"Unsupported support unit implementation '{impl}' for unit '{unit_id}'"
        )

    raise TypeError(
        f"Support unit '{unit_id}' must be an invokable support unit or callable "
        f"returning one (got '{type(asset).__name__}')"
    )


def _build_support_unit_metadata(asset: Any, *, unit_id: str) -> SupportUnitRuntimeMetadata:
    if _is_spec_like(asset, attrs=("unit_id", "impl", "defaults")):
        unit_kind = str(
            getattr(asset, "unit_kind", SupportUnitKind.EXECUTION_SUPPORT.value)
            or SupportUnitKind.EXECUTION_SUPPORT.value
        )
        metadata = dict(getattr(asset, "metadata", {}) or {})
        return SupportUnitRuntimeMetadata(
            unit_id=str(getattr(asset, "unit_id")),
            unit_kind=unit_kind,
            metadata=metadata,
        )
    return SupportUnitRuntimeMetadata(unit_id=unit_id)


def _materialize_support_workflow_spec(
    asset: Any,
    *,
    workflow_id: str,
    registry_view,
) -> GameSupportWorkflow:
    if _is_spec_like(asset, attrs=("workflow_id", "unit_ids")):
        defaults = getattr(asset, "defaults", {}) or {}
        raw_bindings = getattr(asset, "hook_bindings", {}) or {}
        hook_bindings: dict[SupportHook, tuple[str, ...]] = {}
        if isinstance(raw_bindings, dict) and raw_bindings:
            for hook_name, unit_ids in raw_bindings.items():
                hook = SupportHook(str(hook_name))
                hook_bindings[hook] = tuple(
                    str(unit_id) for unit_id in unit_ids if unit_id
                )
        else:
            unit_ids = tuple(str(unit_id) for unit_id in getattr(asset, "unit_ids", ()) if unit_id)
            hook_name = defaults.get("hook", SupportHook.BEFORE_APPLY)
            hook_bindings[SupportHook(str(hook_name))] = unit_ids

        units_by_hook: dict[SupportHook, list[object]] = {}
        unit_metadata: dict[SupportHook, tuple[SupportUnitRuntimeMetadata, ...]] = {}
        for hook, unit_ids in hook_bindings.items():
            units: list[object] = []
            metadata_entries: list[SupportUnitRuntimeMetadata] = []
            for unit_id in unit_ids:
                asset_obj = _lookup_support_unit_asset(registry_view, unit_id=unit_id)
                units.append(
                    _materialize_support_unit(
                        asset_obj,
                        unit_id=unit_id,
                        registry_view=registry_view,
                    )
                )
                metadata_entries.append(
                    _build_support_unit_metadata(asset_obj, unit_id=unit_id)
                )
            if units:
                units_by_hook[hook] = units
                unit_metadata[hook] = tuple(metadata_entries)

        return GameSupportWorkflow(
            workflow_id=str(getattr(asset, "workflow_id")),
            units_by_hook=units_by_hook,
            metadata=dict(getattr(asset, "metadata", {}) or {}),
            degrade_policy=str(
                getattr(asset, "degrade_policy", SupportDegradePolicy.FAIL_FAST.value)
                or SupportDegradePolicy.FAIL_FAST.value
            ),
            unit_metadata=unit_metadata,
        )
    raise TypeError(
        f"Support workflow '{workflow_id}' must be a workflow spec-like object"
    )


def _lookup_support_unit_asset(registry_view, *, unit_id: str) -> Any:
    try:
        return registry_view.get("support_units", unit_id)
    except KeyError:
        if unit_id == "arena/default":
            return _NoOpSupportUnit()
        raise


def _materialize_support_workflow(
    asset: Any,
    *,
    workflow_id: str,
    registry_view,
) -> GameSupportWorkflow:
    if isinstance(asset, GameSupportWorkflow):
        return asset
    if callable(asset):
        workflow = asset()
        if isinstance(workflow, GameSupportWorkflow):
            return workflow
        raise TypeError(
            f"Support workflow builder '{workflow_id}' must return "
            f"'GameSupportWorkflow' (got '{type(workflow).__name__}')"
        )
    return _materialize_support_workflow_spec(
        asset,
        workflow_id=workflow_id,
        registry_view=registry_view,
    )


class SupportWorkflowRegistry:
    def __init__(self, *, registry_view=None) -> None:
        self._registry = registry_view or registry

    def build(self, workflow_id: str) -> GameSupportWorkflow:
        return _materialize_support_workflow(
            self._registry.get("support_workflows", workflow_id),
            workflow_id=workflow_id,
            registry_view=self._registry,
        )
