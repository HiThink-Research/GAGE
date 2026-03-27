from __future__ import annotations

from dataclasses import dataclass, field

from gage_eval.role.arena.core.players import BaseBoundPlayer, PlayerBindingSpec
from gage_eval.role.arena.player_drivers.base import PlayerDriver
from gage_eval.role.arena.types import ArenaAction


@dataclass
class DummyBoundPlayer(BaseBoundPlayer):
    actions: tuple[str, ...] = ()
    _next_index: int = field(default=0, init=False, repr=False)

    def next_action(self, observation) -> ArenaAction:
        del observation
        if not self.actions:
            raise ValueError(f"Player '{self.player_id}' does not define dummy actions")
        if self._next_index < len(self.actions):
            move = self.actions[self._next_index]
        else:
            move = self.actions[-1]
        self._next_index += 1
        return ArenaAction(
            player=self.player_id,
            move=move,
            raw=move,
            metadata=dict(self.metadata),
        )


class DummyPlayerDriver(PlayerDriver):
    def bind(self, spec: PlayerBindingSpec, *, invocation=None) -> DummyBoundPlayer:
        del invocation
        params = self.resolve_params(spec)
        raw_actions = spec.actions or tuple(
            str(action)
            for action in params.get("actions", ())
            if action is not None
        )
        return DummyBoundPlayer(
            player_id=spec.player_id,
            display_name=spec.player_id,
            seat=spec.seat,
            player_kind=spec.player_kind,
            actions=tuple(raw_actions),
            metadata={"driver_id": self.driver_id, "seat": spec.seat},
        )
