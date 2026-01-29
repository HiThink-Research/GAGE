"""Observation utilities for stable-retro environments."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from gage_eval.role.arena.games.retro.temporal.data_contract_v15 import build_observation_dict
from gage_eval.role.arena.types import ArenaObservation


class InfoFeeder:
    """Base class for projecting tick info into a compact payload."""

    def build(self, info_history: Sequence[dict[str, Any]]) -> dict[str, Any]:
        """Build a compact info payload.

        Args:
            info_history: Ordered list of tick info dictionaries.

        Returns:
            Aggregated info payload.
        """

        return info_history[-1] if info_history else {}


@dataclass(frozen=True)
class InfoLastFeeder(InfoFeeder):
    """Expose the last tick info snapshot."""

    def build(self, info_history: Sequence[dict[str, Any]]) -> dict[str, Any]:
        return dict(info_history[-1]) if info_history else {}


@dataclass(frozen=True)
class InfoDeltaFeeder(InfoFeeder):
    """Expose the last tick and numeric deltas over the window."""

    window_size: int = 8

    def build(self, info_history: Sequence[dict[str, Any]]) -> dict[str, Any]:
        if not info_history:
            return {}
        window = list(info_history[-self.window_size :])
        last = dict(window[-1])
        delta: dict[str, Any] = {}
        if len(window) >= 2:
            prev = window[-2]
            for key, value in last.items():
                if isinstance(value, (int, float)) and isinstance(prev.get(key), (int, float)):
                    delta[key] = value - prev.get(key)
        return {"last": last, "delta": delta, "window": window}


@dataclass(frozen=True)
class ActionSchema:
    """Action schema description for retro prompts."""

    hold_ticks_min: int = 1
    hold_ticks_max: int = 20
    default_hold_ticks: int = 6


class ObservationBuilder:
    """Builds arena observations from retro tick history."""

    def __init__(
        self,
        *,
        info_feeder: InfoFeeder,
        action_schema: ActionSchema,
        token_budget: int = 200,
    ) -> None:
        """Initialize the builder.

        Args:
            info_feeder: Info feeder to aggregate tick info.
            action_schema: Action schema descriptor.
            token_budget: Approximate token budget for the info payload.
        """

        self._info_feeder = info_feeder
        self._action_schema = action_schema
        self._token_budget = max(50, int(token_budget))

    def build(
        self,
        *,
        player_id: str,
        active_player: str,
        legal_moves: Sequence[str],
        last_move: str | None,
        tick: int,
        decision_count: int,
        info_history: Sequence[dict[str, Any]],
        raw_info: dict[str, Any] | None,
        reward_total: float,
    ) -> ArenaObservation:
        """Construct an ArenaObservation for the decision point.

        Args:
            player_id: Player identifier.
            active_player: Active player identifier.
            legal_moves: List of legal macro moves.
            last_move: Last decision move.
            tick: Current tick index.
            decision_count: Number of decisions taken.
            info_history: Recent tick info history.
            raw_info: Most recent tick info snapshot from the environment.
            reward_total: Accumulated reward.

        Returns:
            ArenaObservation payload.
        """

        info_payload = self._info_feeder.build(info_history)
        info_text = self._truncate_info(info_payload)
        prompt_text = self._build_prompt_text(info_text, legal_moves)
        board_text = self._build_board_text(info_text, legal_moves)
        contract_v15 = build_observation_dict(
            view_text=prompt_text,
            legal_actions=legal_moves,
            active_player=active_player,
            tick=tick,
            step=decision_count,
            info=raw_info or {},
            extra={
                "board_text": board_text,
                "player_id": str(player_id),
                "last_move": last_move,
                "reward_total": float(reward_total),
                "info_projection": info_payload,
                "action_schema": {
                    "hold_ticks_default": int(self._action_schema.default_hold_ticks),
                    "hold_ticks_min": int(self._action_schema.hold_ticks_min),
                    "hold_ticks_max": int(self._action_schema.hold_ticks_max),
                },
            },
        )
        return ArenaObservation(
            board_text=board_text,
            legal_moves=list(legal_moves),
            active_player=str(active_player),
            last_move=last_move,
            metadata={
                "player_id": str(player_id),
                "tick": int(tick),
                "decision_count": int(decision_count),
                "reward_total": float(reward_total),
                "info": dict(raw_info or {}),
                "info_projection": info_payload,
                "prompt_text": prompt_text,
                "hold_ticks_default": int(self._action_schema.default_hold_ticks),
                "hold_ticks_min": int(self._action_schema.hold_ticks_min),
                "hold_ticks_max": int(self._action_schema.hold_ticks_max),
                "contract_v15": contract_v15,
            },
        )

    def _build_prompt_text(self, info_text: str, legal_moves: Sequence[str]) -> str:
        legal_hint = ", ".join(list(legal_moves)[:40]) if legal_moves else "none"
        schema = (
            "{\"move\": \"<legal_move>\", \"hold_ticks\": <int>}"
        )
        lines = [
            "You are controlling a real-time retro game at decision points.",
            "Output ONE JSON object on the last line.",
            f"Schema: {schema}",
            f"Legal moves: {legal_hint}",
            "hold_ticks must be an integer within the allowed range.",
            "Latest info (JSON):",
            info_text,
        ]
        return "\n".join(lines)

    def _build_board_text(self, info_text: str, legal_moves: Sequence[str]) -> str:
        legal_hint = ", ".join(list(legal_moves)[:40]) if legal_moves else "none"
        lines = [
            "Retro Observation:",
            f"Legal moves: {legal_hint}",
            "Info:",
            info_text,
        ]
        return "\n".join(lines)

    def _truncate_info(self, info_payload: Mapping[str, Any]) -> str:
        text = json.dumps(info_payload, ensure_ascii=True)
        max_chars = self._token_budget * 4
        if len(text) <= max_chars:
            return text
        return text[: max_chars - 3] + "..."
