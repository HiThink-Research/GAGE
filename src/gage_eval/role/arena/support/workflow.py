"""Support workflow runner."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

from gage_eval.role.arena.support.context import SupportContext
from gage_eval.role.arena.support.hooks import (
    SupportDegradePolicy,
    SupportHook,
    SupportUnit,
    SupportUnitKind,
)


@dataclass(frozen=True)
class SupportUnitRuntimeMetadata:
    unit_id: str
    unit_kind: str = SupportUnitKind.EXECUTION_SUPPORT.value
    metadata: dict[str, object] = field(default_factory=dict)


@dataclass
class GameSupportWorkflow:
    workflow_id: str
    units_by_hook: dict[SupportHook, list[SupportUnit]] = field(default_factory=dict)
    metadata: dict[str, object] = field(default_factory=dict)
    degrade_policy: str = SupportDegradePolicy.FAIL_FAST.value
    unit_metadata: dict[SupportHook, tuple[SupportUnitRuntimeMetadata, ...]] = field(
        default_factory=dict
    )

    def run(self, hook: SupportHook, context: SupportContext) -> SupportContext:
        current = context
        units = list(self.units_by_hook.get(hook, []))
        metadata = self._resolve_unit_metadata(hook, units)
        for index, unit in enumerate(units):
            unit_meta = metadata[index]
            try:
                result = unit.invoke(current)
            except Exception as exc:
                if self._should_continue_after_error():
                    errors = current.state.setdefault("support_errors", [])
                    if isinstance(errors, list):
                        errors.append(
                            {
                                "error_code": "support_workflow_failure",
                                "workflow_id": self.workflow_id,
                                "hook": str(hook),
                                "unit_id": unit_meta.unit_id,
                                "unit_kind": unit_meta.unit_kind,
                                "message": str(exc),
                            }
                        )
                    continue
                raise
            if isinstance(result, SupportContext):
                current = result
        return current

    def _resolve_unit_metadata(
        self,
        hook: SupportHook,
        units: Sequence[SupportUnit],
    ) -> tuple[SupportUnitRuntimeMetadata, ...]:
        existing = tuple(self.unit_metadata.get(hook, ()))
        if len(existing) >= len(units):
            return existing
        generated = list(existing)
        for index in range(len(existing), len(units)):
            generated.append(
                SupportUnitRuntimeMetadata(
                    unit_id=f"{self.workflow_id}:{hook}:{index}",
                )
            )
        return tuple(generated)

    def _should_continue_after_error(self) -> bool:
        return str(self.degrade_policy or "").strip().lower() == SupportDegradePolicy.CONTINUE
