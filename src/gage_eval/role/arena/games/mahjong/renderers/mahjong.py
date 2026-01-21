"""Mahjong renderer."""

from __future__ import annotations

from typing import Any
from gage_eval.registry import registry
from gage_eval.role.arena.games.mahjong.renderers.base import MahjongRenderer
from gage_eval.role.arena.games.mahjong.types import MahjongGameResult

@registry.asset("renderer_impls", "mahjong_replay_v1", desc="Mahjong replay renderer")
class StandardMahjongRenderer(MahjongRenderer):
    """Produces JSON-friendly frames and replay payloads."""

    def render_frame(self, frame: dict[str, Any]) -> dict[str, Any]:
        return {
            "active_player_id": frame.get("active_player_id"),
            "public_state": frame.get("public_state", {}),
            "private_state": frame.get("private_state", {}),
        }
    
    def save_replay(self, game_result: MahjongGameResult) -> dict[str, Any]:
        return {
            "winner": game_result.get("winner"),
            "reason": game_result.get("reason"),
            "moves": list(game_result.get("move_log", [])),
            "chat_log": list(game_result.get("chat_log", [])),
            "payoffs": list(game_result.get("payoffs", [])),
        }
