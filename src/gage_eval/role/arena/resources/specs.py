from __future__ import annotations

from dataclasses import dataclass
from dataclasses import field

from gage_eval.role.arena.resources.handles import RuntimeHandle
from gage_eval.role.arena.resources.runtime_bridge import RuntimeBridge


@dataclass
class ArenaResources:
    resource_spec: object = field(default_factory=dict)
    resource_categories: tuple[str, ...] = ()
    lifecycle_phase: str = "allocated"
    lifecycle_events: list[dict[str, object]] = field(default_factory=list)
    resource_artifacts: dict[str, object] = field(default_factory=dict)
    errors: list[dict[str, object]] = field(default_factory=list)
    visualization: object | None = None
    game_runtime: RuntimeHandle | None = None
    game_bridge: RuntimeBridge | None = None
    compute: object | None = None
    output: object | None = None
    sandbox: object | None = None

    def record_lifecycle(
        self,
        phase: str,
        *,
        resource_category: str | None = None,
        details: dict[str, object] | None = None,
    ) -> None:
        self.lifecycle_phase = str(phase)
        event: dict[str, object] = {"phase": self.lifecycle_phase}
        if resource_category is not None:
            event["resource_category"] = resource_category
        if details:
            event["details"] = dict(details)
        self.lifecycle_events.append(event)
