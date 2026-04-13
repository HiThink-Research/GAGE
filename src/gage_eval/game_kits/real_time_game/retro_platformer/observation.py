"""Observation builders and info feeders for retro platformer environments."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Optional, Sequence

from gage_eval.role.arena.types import ArenaObservation, ArenaPromptSpec


@dataclass(frozen=True)
class ActionSchema:
    """Defines the action schema constraints for retro moves."""

    hold_ticks_min: int = 1
    hold_ticks_max: int = 20
    default_hold_ticks: int = 6

    def format_prompt(self, legal_moves: Sequence[str]) -> str:
        moves_hint = ", ".join([str(move) for move in legal_moves]) if legal_moves else "none"
        return (
            "Output ONE JSON object:\n"
            '{ "move": "<legal_move_or_key_combo>", "hold_ticks": <int> }\n'
            "Key aliases accepted for `move`: w/a/s/d (up/left/down/right), j=A(jump), "
            "k=B(run), l=select, enter=start.\n"
            "Key combos are allowed (e.g. `d+j+k`).\n"
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
        del info_history
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


class InfoNoneFeeder(InfoFeeder):
    """Disable info projection in the observation view text."""

    def build(
        self,
        *,
        info_history: Sequence[dict[str, Any]],
        raw_info: dict[str, Any],
        token_budget: int,
    ) -> tuple[str, dict[str, Any]]:
        del info_history
        del raw_info
        del token_budget
        return "", {}


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
        controls: Optional[dict[str, Any]] = None,
        image: Optional[dict[str, Any]] = None,
        game_type: str = "retro",
        env_id: Optional[str] = None,
    ) -> ArenaObservation:
        legal_move_items = [str(move) for move in legal_moves]
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
        controls_block = self._format_controls_block(controls)
        if controls_block:
            view_text = f"{view_text}\n\n{controls_block}"
        action_hint = self._action_schema.format_prompt(legal_move_items)
        action_schema_config = {
            "hold_ticks_min": int(self._action_schema.hold_ticks_min),
            "hold_ticks_max": int(self._action_schema.hold_ticks_max),
            "hold_ticks_default": int(self._action_schema.default_hold_ticks),
        }
        game_label = str(game_type or "retro")
        env_label = str(env_id or game_label)
        prompt_instruction = self._format_prompt_instruction(
            view_text=view_text,
            legal_moves=legal_move_items,
            action_schema=action_hint,
        )
        prompt_renderer_instruction = self._format_renderer_instruction(
            view_text=view_text,
            legal_moves=legal_move_items,
        )
        prompt_payload = self._build_prompt_payload(
            player_id=player_id,
            game_type=game_label,
            env_id=env_label,
            active_player=active_player,
            last_move=last_move,
            legal_moves=legal_move_items,
            action_schema=action_hint,
            action_schema_config=action_schema_config,
            tick=tick,
            decision_count=decision_count,
            reward_total=reward_total,
            info=dict(raw_info),
            controls=dict(controls or {}),
            view_text=view_text,
        )
        observation_dict = _build_observation_dict(
            view_text=view_text,
            image=image,
            legal_actions=legal_move_items,
            active_player=active_player,
            tick=tick,
            step=decision_count,
            info=raw_info,
            extra={
                "info_projection": info_extra,
                "action_schema": action_hint,
                "action_schema_config": action_schema_config,
                "controls": controls or {},
            },
        )
        metadata = {
            "game_type": game_label,
            "player_id": player_id,
            "observation_dict": observation_dict,
            "action_schema": action_hint,
            "action_schema_config": action_schema_config,
            "token_budget": self._token_budget,
            "last_move": last_move,
            "observation_extra": observation_dict.get("extra", {}),
        }
        if env_id:
            metadata["env_id"] = str(env_id)

        return ArenaObservation(
            board_text=view_text,
            legal_moves=legal_move_items,
            active_player=active_player,
            last_move=last_move,
            metadata=metadata,
            view=observation_dict.get("view"),
            legal_actions={"items": list(observation_dict.get("legal_actions") or [])},
            context=observation_dict.get("context"),
            prompt=ArenaPromptSpec(
                instruction=prompt_instruction,
                renderer_instruction=prompt_renderer_instruction,
                payload=prompt_payload,
            ),
        )

    @staticmethod
    def _format_prompt_instruction(
        *,
        view_text: str,
        legal_moves: Sequence[str],
        action_schema: str,
    ) -> str:
        legal_hint = ", ".join([str(move) for move in legal_moves]) if legal_moves else "none"
        lines = [
            "You are playing a retro game environment.",
            view_text,
            "",
            "Status:",
            f"- Legal moves: {legal_hint}",
            "",
            "Instructions:",
            *str(action_schema or "").splitlines(),
        ]
        return "\n".join(lines)

    @staticmethod
    def _format_renderer_instruction(*, view_text: str, legal_moves: Sequence[str]) -> str:
        legal_hint = ", ".join([str(move) for move in legal_moves]) if legal_moves else "none"
        lines = [
            "You are playing a retro game environment.",
            view_text,
            "",
            "Status:",
            f"- Legal moves: {legal_hint}",
            "",
            "Instructions:",
            "- Follow the system prompt for output format and policy.",
            "- Choose one move from the legal moves list.",
        ]
        return "\n".join(lines)

    @staticmethod
    def _build_prompt_payload(
        *,
        player_id: str,
        game_type: str,
        env_id: str,
        active_player: str,
        last_move: Optional[str],
        legal_moves: Sequence[str],
        action_schema: str,
        action_schema_config: dict[str, Any],
        tick: int,
        decision_count: int,
        reward_total: float,
        info: dict[str, Any],
        controls: dict[str, Any],
        view_text: str,
    ) -> dict[str, Any]:
        hold_ticks = {
            "min": action_schema_config.get("hold_ticks_min"),
            "max": action_schema_config.get("hold_ticks_max"),
            "default": action_schema_config.get("hold_ticks_default"),
        }
        return {
            "player_id": str(player_id),
            "game_type": str(game_type),
            "env_id": str(env_id),
            "mode": "tick",
            "scheduler_mode": "tick",
            "observation_mode": "tick",
            "legal_moves": [str(move) for move in legal_moves],
            "action_schema": str(action_schema),
            "action_schema_config": dict(action_schema_config),
            "hold_ticks": hold_ticks,
            "arena_observation": {
                "view_text": str(view_text),
                "legal_moves": [str(move) for move in legal_moves],
                "active_player": str(active_player),
                "last_action": None if last_move is None else str(last_move),
                "metadata": {
                    "game_type": str(game_type),
                    "env_id": str(env_id),
                    "player_id": str(player_id),
                    "reward_total": float(reward_total),
                    "info": dict(info),
                    "controls": dict(controls),
                },
                "context": {
                    "mode": "tick",
                    "step": int(decision_count),
                    "tick": int(tick),
                },
            },
        }

    @staticmethod
    def _format_view_text(
        *,
        tick: int,
        decision_count: int,
        reward_total: float,
        info_text: str,
    ) -> str:
        lines = [
            "Mode: tick",
            f"Decision step: {decision_count}",
            f"Tick: {tick}",
            f"Reward total: {reward_total:.3f}",
        ]
        normalized_info = str(info_text or "").strip()
        if normalized_info:
            lines.extend(["Info:", normalized_info])
        return "\n".join(lines)

    @staticmethod
    def _format_controls_block(controls: Optional[dict[str, Any]]) -> str:
        if not isinstance(controls, dict) or not controls:
            return ""
        keys_hint = controls.get("keys_hint")
        buttons = controls.get("buttons")
        move_aliases = controls.get("move_aliases")

        lines: list[str] = ["Controls:"]
        if isinstance(keys_hint, str) and keys_hint.strip():
            lines.append(f"- {keys_hint.strip()}")
        if isinstance(buttons, (list, tuple)) and buttons:
            button_preview = ", ".join([str(btn) for btn in buttons])
            lines.append(f"- Stable-retro buttons: {button_preview}")
        if isinstance(move_aliases, dict) and move_aliases:
            preview_items: list[str] = []
            for move, payload in move_aliases.items():
                if not isinstance(payload, dict):
                    continue
                key_combo = payload.get("keys_combo")
                if not isinstance(key_combo, str) or not key_combo:
                    continue
                buttons_payload = payload.get("buttons")
                if isinstance(buttons_payload, (list, tuple)) and buttons_payload:
                    button_combo = "+".join([str(btn) for btn in buttons_payload])
                    preview_items.append(f"{move}={key_combo} ({button_combo})")
                else:
                    preview_items.append(f"{move}={key_combo}")
            if preview_items:
                lines.append("- Legal move aliases: " + ", ".join(preview_items))
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
    image: Optional[dict[str, Any]],
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
    view_payload: dict[str, Any] = {"text": str(view_text)}
    if isinstance(image, dict) and image:
        view_payload["image"] = dict(image)
    return {
        "view": view_payload,
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
    "InfoNoneFeeder",
    "ObservationBuilder",
]
