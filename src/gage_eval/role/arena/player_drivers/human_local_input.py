from __future__ import annotations

from collections.abc import Mapping
from queue import Empty
from typing import Any

from gage_eval.role.arena.core.invocation import GameArenaInvocationContext
from gage_eval.role.arena.core.players import BaseBoundPlayer, PlayerBindingSpec
from gage_eval.role.arena.human_input_protocol import extract_action_text, parse_action_payload
from gage_eval.role.arena.player_drivers.base import PlayerDriver
from gage_eval.role.arena.types import ArenaAction, ArenaObservation


class LocalHumanBoundPlayer(BaseBoundPlayer):
    def __init__(
        self,
        *,
        action_queue: Any | None = None,
        timeout_ms: int | None = None,
        timeout_fallback_move: str | None = None,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self._action_queue = action_queue
        self._timeout_ms = timeout_ms
        self._timeout_fallback_move = timeout_fallback_move

    def next_action(self, observation: ArenaObservation) -> ArenaAction:
        payload = self._read_action_payload(observation)
        move = self._resolve_move(payload)
        if not move:
            fallback = self._resolve_timeout_fallback_move(observation)
            move = str(fallback or "").strip()
        if not move:
            raise ValueError(f"Human player '{self.player_id}' did not provide an action")
        metadata = {"driver_id": self.metadata.get("driver_id"), "player_type": "human"}
        payload_metadata = payload.get("metadata") if isinstance(payload, Mapping) else None
        if isinstance(payload_metadata, Mapping):
            metadata.update(dict(payload_metadata))
        return ArenaAction(
            player=self.player_id,
            move=move,
            raw=self._resolve_raw(payload, move),
            metadata=metadata,
        )

    def _read_action_payload(self, observation: ArenaObservation) -> dict[str, Any]:
        queue = self._action_queue
        if queue is not None:
            timeout_s = None
            if self._timeout_ms is not None:
                timeout_s = max(0.0, float(self._timeout_ms) / 1000.0)
            get = getattr(queue, "get", None)
            if callable(get):
                try:
                    queued = get(timeout=timeout_s) if timeout_s is not None else get()
                except Empty:
                    return {}
                return parse_action_payload(queued)
        return parse_action_payload(input(self._format_prompt(observation)))

    def _resolve_move(self, payload: Mapping[str, Any] | None) -> str:
        if not isinstance(payload, Mapping):
            return ""
        move = payload.get("action", payload.get("move"))
        if move is None:
            move = extract_action_text(payload)
        return str(move or "").strip()

    def _resolve_raw(self, payload: Mapping[str, Any] | None, move: str) -> str:
        if isinstance(payload, Mapping):
            raw = str(payload.get("raw") or "").strip()
            if raw:
                return raw
        return str(move or "").strip()

    def _resolve_timeout_fallback_move(self, observation: ArenaObservation) -> str | None:
        if self._timeout_fallback_move:
            return self._timeout_fallback_move
        legal_moves = list(observation.legal_actions_items)
        if legal_moves:
            return str(legal_moves[0])
        return None

    def _format_prompt(self, observation: ArenaObservation) -> str:
        legal_hint = ", ".join(str(item) for item in observation.legal_actions_items) or "none"
        return (
            f"Active player: {observation.active_player}\n"
            f"{observation.view_text}\n"
            f"Legal moves: {legal_hint}\n"
            "Enter exactly one legal move: "
        )


class LocalHumanInputDriver(PlayerDriver):
    def bind(
        self,
        spec: PlayerBindingSpec,
        *,
        invocation: GameArenaInvocationContext | None = None,
    ) -> LocalHumanBoundPlayer:
        params = self.resolve_params(spec)
        action_queue = params.get("action_queue")
        if action_queue is None and invocation is not None:
            action_queue = invocation.queue_for_player(spec.player_id)
        timeout_ms = _coerce_optional_int(params.get("timeout_ms"))
        timeout_fallback_move = params.get("timeout_fallback_move")
        return LocalHumanBoundPlayer(
            player_id=spec.player_id,
            display_name=spec.player_id,
            seat=spec.seat,
            player_kind=spec.player_kind,
            action_queue=action_queue,
            timeout_ms=timeout_ms,
            timeout_fallback_move=(
                None if timeout_fallback_move is None else str(timeout_fallback_move)
            ),
            metadata={"driver_id": self.driver_id, "seat": spec.seat},
        )


def _coerce_optional_int(value: object) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None
