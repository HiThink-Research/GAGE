"""Observation builders and info feeders for stable-retro environments."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Optional, Sequence

from gage_eval.role.arena.types import ArenaObservation


@dataclass(frozen=True)
class ActionSchema:
    """Defines the action schema constraints for retro moves."""

    hold_ticks_min: int = 1
    hold_ticks_max: int = 20
    default_hold_ticks: int = 6

    def format_prompt(self, legal_moves: Sequence[str]) -> str:
        """Format the action schema instructions for the player."""

        moves_hint = ", ".join([str(move) for move in legal_moves]) if legal_moves else "none"
        return (
            "Output ONE JSON object:\n"
            '{ "move": "<legal_move>", "hold_ticks": <int> }\n'
            f"hold_ticks range: {self.hold_ticks_min}-{self.hold_ticks_max} "
            f"(default {self.default_hold_ticks}).\n"
            f"Legal moves: {moves_hint}."
        )


class InfoFeeder:
    """Base class for info projection strategies."""

    def build(
        self,
        *,
        info_history: Sequence[dict[str, Any]],
        raw_info: dict[str, Any],
        token_budget: int,
    ) -> tuple[str, dict[str, Any]]:
        """Build info text and extra payload from raw info."""

        del info_history
        del token_budget
        return _truncate_text(_json_dumps(raw_info), token_budget), {}


class InfoLastFeeder(InfoFeeder):
    """Expose the last tick info payload."""

    def build(
        self,
        *,
        info_history: Sequence[dict[str, Any]],
        raw_info: dict[str, Any],
        token_budget: int,
    ) -> tuple[str, dict[str, Any]]:
        del info_history
        text = _truncate_text(_json_dumps(raw_info), token_budget)
        return text, {"info_last": dict(raw_info)}


class InfoDeltaFeeder(InfoFeeder):
    """Expose the delta between recent info payloads."""

    def __init__(self, *, window_size: int = 8) -> None:
        self._window_size = max(1, int(window_size))

    def build(
        self,
        *,
        info_history: Sequence[dict[str, Any]],
        raw_info: dict[str, Any],
        token_budget: int,
    ) -> tuple[str, dict[str, Any]]:
        history = list(info_history)[-self._window_size :]
        previous = history[-2] if len(history) >= 2 else {}
        delta = _numeric_delta(previous, raw_info)
        payload = {"info_last": dict(raw_info), "info_delta": delta, "window_size": self._window_size}
        text = _truncate_text(_json_dumps({"last": raw_info, "delta": delta}), token_budget)
        return text, payload


class ObservationBuilder:
    """Builds retro observations aligned with the section-15 contracts."""

    def __init__(self, *, info_feeder: InfoFeeder, action_schema: ActionSchema, token_budget: int) -> None:
        self._info_feeder = info_feeder
        self._action_schema = action_schema
        self._token_budget = max(1, int(token_budget))

    def build(
        self,
        *,
        player_id: str,
        active_player: str,
        legal_moves: Sequence[str],
        last_move: Optional[str],
        tick: int,
        decision_count: int,
        info_history: Sequence[dict[str, Any]],
        raw_info: dict[str, Any],
        reward_total: float,
    ) -> ArenaObservation:
        """Build an ArenaObservation plus the section-15 dict payload."""

        info_text, info_extra = self._info_feeder.build(
            info_history=info_history,
            raw_info=raw_info,
            token_budget=self._token_budget,
        )
        view_text = self._format_view_text(
            tick=tick,
            decision_count=decision_count,
            reward_total=reward_total,
            info_text=info_text,
        )
        action_hint = self._action_schema.format_prompt(legal_moves)
        observation_dict = _build_observation_dict(
            view_text=view_text,
            legal_actions=[str(move) for move in legal_moves],
            active_player=active_player,
            tick=tick,
            step=decision_count,
            info=raw_info,
            extra={"info_projection": info_extra, "action_schema": action_hint},
        )
        return ArenaObservation(
            board_text=view_text,
            legal_moves=list(legal_moves),
            active_player=active_player,
            last_move=last_move,
            metadata={
                "player_id": player_id,
                "observation_dict": observation_dict,
                "action_schema": action_hint,
                "token_budget": self._token_budget,
            },
            view=observation_dict.get("view"),
            context=observation_dict.get("context"),
            extra=observation_dict.get("extra", {}),
        )

    @staticmethod
    def _format_view_text(
        *,
        tick: int,
        decision_count: int,
        reward_total: float,
        info_text: str,
    ) -> str:
        lines = [
            f"Mode: tick",
            f"Decision step: {decision_count}",
            f"Tick: {tick}",
            f"Reward total: {reward_total:.3f}",
            "Info:",
            info_text or "{}",
        ]
        return "\n".join(lines)


def _numeric_delta(previous: dict[str, Any], current: dict[str, Any]) -> dict[str, float]:
    delta: dict[str, float] = {}
    for key, value in current.items():
        if key not in previous:
            continue
        if isinstance(value, (int, float)) and isinstance(previous[key], (int, float)):
            diff = float(value) - float(previous[key])
            if diff != 0.0:
                delta[str(key)] = diff
    return delta


def _json_dumps(payload: Any) -> str:
    try:
        return json.dumps(payload, ensure_ascii=False, sort_keys=True)
    except TypeError:
        return json.dumps(str(payload), ensure_ascii=False, sort_keys=True)


def _truncate_text(text: str, token_budget: int) -> str:
    if token_budget <= 0:
        return ""
    if len(text) <= token_budget * 4:
        return text
    return text[: token_budget * 4] + "..."


def _build_observation_dict(
    *,
    view_text: str,
    legal_actions: Sequence[str],
    active_player: str,
    tick: int,
    step: int,
    info: dict[str, Any],
    extra: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    merged_extra: dict[str, Any] = {"info": dict(info)}
    if extra:
        merged_extra.update(extra)
    return {
        "view": {"text": str(view_text)},
        "legal_actions": [str(item) for item in legal_actions],
        "context": {"mode": "tick", "step": int(step), "tick": int(tick)},
        "active_player": str(active_player),
        "extra": merged_extra,
    }


__all__ = [
    "ActionSchema",
    "InfoDeltaFeeder",
    "InfoFeeder",
    "InfoLastFeeder",
    "ObservationBuilder",
]
