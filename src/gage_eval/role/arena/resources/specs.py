from __future__ import annotations

from dataclasses import dataclass
from dataclasses import field

from gage_eval.role.arena.resources.handles import RuntimeHandle
from gage_eval.role.arena.resources.runtime_bridge import RuntimeBridge


@dataclass
class ArenaResources:
    resource_spec: object = field(default_factory=dict)
    visualization: object | None = None
    game_runtime: RuntimeHandle | None = None
    game_bridge: RuntimeBridge | None = None
    output: object | None = None
