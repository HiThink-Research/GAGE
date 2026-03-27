from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Literal, Protocol

from gage_eval.role.arena.types import ArenaAction

PlayerKind = Literal["llm", "human", "agent", "dummy"]


@dataclass(frozen=True)
class PlayerBindingSpec:
    seat: str
    player_id: str
    player_kind: PlayerKind
    driver_id: str
    backend_id: str | None = None
    agent_role_id: str | None = None
    actions: tuple[str, ...] | None = None
    driver_params: dict[str, object] = field(default_factory=dict)


class BoundPlayer(Protocol):
    player_id: str
    display_name: str
    seat: str
    player_kind: PlayerKind
    metadata: Mapping[str, object]

    def next_action(self, observation) -> ArenaAction: ...


@dataclass
class BaseBoundPlayer:
    player_id: str
    display_name: str
    seat: str
    player_kind: PlayerKind
    metadata: Mapping[str, object] = field(default_factory=dict)


__all__ = [
    "BaseBoundPlayer",
    "BoundPlayer",
    "PlayerBindingSpec",
    "PlayerKind",
]
