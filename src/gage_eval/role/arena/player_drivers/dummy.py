from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

from gage_eval.role.arena.core.players import BaseBoundPlayer, PlayerBindingSpec
from gage_eval.role.arena.player_drivers.base import PlayerDriver
from gage_eval.role.arena.types import ArenaAction


@dataclass
class DummyBoundPlayer(BaseBoundPlayer):
    actions: tuple[str, ...] = ()
    _next_index: int = field(default=0, init=False, repr=False)

    def next_action(self, observation) -> ArenaAction:
        if not self.actions:
            raise ValueError(f"Player '{self.player_id}' does not define dummy actions")
        legal_moves = _extract_legal_moves(observation)
        legal_lookup = {move.lower(): move for move in legal_moves}
        while self._next_index < len(self.actions):
            raw_move = self.actions[self._next_index]
            self._next_index += 1
            move_text = str(raw_move).strip()
            if not move_text:
                continue
            if not legal_lookup:
                return ArenaAction(
                    player=self.player_id,
                    move=raw_move,
                    raw=raw_move,
                    metadata=dict(self.metadata),
                )
            resolved = legal_lookup.get(move_text.lower())
            if resolved is not None:
                return ArenaAction(
                    player=self.player_id,
                    move=resolved,
                    raw=resolved,
                    metadata=dict(self.metadata),
                )
        if not legal_moves:
            raise ValueError(f"Player '{self.player_id}' ran out of dummy actions and no legal moves remain")
        move = legal_moves[0]
        return ArenaAction(
            player=self.player_id,
            move=move,
            raw=move,
            metadata={**dict(self.metadata), "fallback": "first_legal"},
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


def _extract_legal_moves(observation: object) -> list[str]:
    items = getattr(observation, "legal_actions_items", None)
    if items is not None:
        return [str(item).strip() for item in items if str(item).strip()]
    if isinstance(observation, Mapping):
        legal_actions = observation.get("legal_actions")
        if isinstance(legal_actions, Mapping):
            raw_items = legal_actions.get("items")
            if isinstance(raw_items, (list, tuple)):
                return [str(item).strip() for item in raw_items if str(item).strip()]
        raw_moves = observation.get("legal_moves")
        if isinstance(raw_moves, (list, tuple)):
            return [str(item).strip() for item in raw_moves if str(item).strip()]
    return []
