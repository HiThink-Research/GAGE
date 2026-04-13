"""Runtime observation workflow contracts and registry helpers."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from gage_eval.registry import registry


@dataclass(frozen=True)
class ObservationWorkflow:
    """Minimal runtime observation workflow.

    The default workflow is intentionally a no-op so registry-managed specs can
    resolve without over-coupling the game-kit layer to any particular player or
    rendering implementation.
    """

    workflow_id: str
    defaults: dict[str, object] = field(default_factory=dict)

    def build(self, raw_observation: object, session: object) -> object:
        del session
        return raw_observation


@registry.asset(
    "observation_workflows",
    "noop_observation_v1",
    desc="No-op observation workflow",
)
def build_noop_observation_workflow() -> ObservationWorkflow:
    return ObservationWorkflow(workflow_id="noop_observation_v1")


def _materialize_observation_workflow(asset: Any, *, workflow_id: str) -> ObservationWorkflow:
    if isinstance(asset, ObservationWorkflow):
        return asset

    if callable(asset):
        workflow = asset()
        if isinstance(workflow, ObservationWorkflow):
            return workflow
        raise TypeError(
            f"Observation workflow builder '{workflow_id}' must return "
            f"'ObservationWorkflow' (got '{type(workflow).__name__}')"
        )

    if all(hasattr(asset, attr) for attr in ("workflow_id", "impl", "defaults")):
        impl = str(getattr(asset, "impl")).strip()
        defaults = getattr(asset, "defaults", {}) or {}
        if not impl or impl.startswith("placeholder://") or impl == "noop_observation_v1":
            return ObservationWorkflow(
                workflow_id=str(getattr(asset, "workflow_id")),
                defaults=dict(defaults),
            )
        raise KeyError(
            f"Unsupported observation workflow implementation '{getattr(asset, 'impl')}' "
            f"for workflow '{workflow_id}'"
        )

    raise TypeError(
        f"Observation workflow '{workflow_id}' must be an 'ObservationWorkflow' or callable "
        f"returning one (got '{type(asset).__name__}')"
    )


class ObservationWorkflowRegistry:
    """Build registered observation workflow assets into runtime workflows."""

    _BUILTIN_NOOP_IDS = {"noop_observation_v1", "arena/default"}

    def __init__(self, *, registry_view=None) -> None:
        self._registry = registry_view or registry

    def build(self, workflow_id: str) -> ObservationWorkflow:
        try:
            asset = self._registry.get("observation_workflows", workflow_id)
        except KeyError:
            if workflow_id in self._BUILTIN_NOOP_IDS:
                return ObservationWorkflow(workflow_id=workflow_id)
            raise
        return _materialize_observation_workflow(asset, workflow_id=workflow_id)
