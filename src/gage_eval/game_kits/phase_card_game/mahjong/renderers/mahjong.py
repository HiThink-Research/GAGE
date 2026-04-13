"""Mahjong renderer."""

from __future__ import annotations

from typing import Any

from gage_eval.game_kits.phase_card_game.mahjong.renderers.base import MahjongRenderer
from gage_eval.game_kits.phase_card_game.mahjong.types import MahjongGameResult
from gage_eval.registry import registry


@registry.asset("renderer_impls", "mahjong_replay_v1", desc="Mahjong replay renderer")
class StandardMahjongRenderer(MahjongRenderer):
    """Produces JSON-friendly frames and replay payloads."""

    def render_frame(self, frame: dict[str, Any]) -> dict[str, Any]:
        return {
            "active_player_id": frame.get("active_player_id"),
            "public_state": frame.get("public_state", {}),
            "private_state": frame.get("private_state", {}),
            "observer_player_id": frame.get("observer_player_id"),
            "legal_moves": list(frame.get("legal_moves", [])),
            "move_count": frame.get("move_count", 0),
            "last_move": frame.get("last_move"),
            "chat_log": list(frame.get("chat_log", [])),
            "timestamp_ms": frame.get("timestamp_ms"),
        }

    def save_replay(self, game_result: MahjongGameResult) -> dict[str, Any]:
        return {
            "winner": game_result.get("winner"),
            "reason": game_result.get("reason"),
            "moves": list(game_result.get("move_log", [])),
            "chat_log": list(game_result.get("chat_log", [])),
            "payoffs": list(game_result.get("payoffs", [])),
        }
