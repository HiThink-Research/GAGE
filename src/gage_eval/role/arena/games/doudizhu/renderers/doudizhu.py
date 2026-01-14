"""Doudizhu renderer for lightweight frame and replay output."""

from __future__ import annotations

from typing import Any

from gage_eval.registry import registry
from gage_eval.role.arena.games.doudizhu.renderers.base import CardGameRenderer
from gage_eval.role.arena.games.doudizhu.types import CardGameResult


@registry.asset(
    "renderer_impls",
    "doudizhu_replay_v1",
    desc="Doudizhu replay renderer (JSON payloads)",
    tags=("doudizhu", "renderer", "card"),
)
class DoudizhuRenderer(CardGameRenderer):
    """Produces minimal JSON-friendly frames and replay payloads."""

    def render_frame(self, frame: dict[str, Any]) -> dict[str, Any]:
        """Return a lightweight frame payload for the current state.

        Args:
            frame: Frame payload containing public/private state.

        Returns:
            JSON-friendly frame payload.
        """

        return {
            "active_player_id": frame.get("active_player_id"),
            "public_state": frame.get("public_state", {}),
            "private_state": frame.get("private_state", {}),
        }

    def save_replay(self, game_result: CardGameResult) -> dict[str, Any]:
        """Build a replay payload based on the final game result.

        Args:
            game_result: Final game result payload.

        Returns:
            Replay payload.
        """

        return {
            "winner": game_result.get("winner"),
            "reason": game_result.get("reason"),
            "moves": list(game_result.get("move_log", [])),
            "chat_log": list(game_result.get("chat_log", [])),
            "payoffs": list(game_result.get("payoffs", [])),
        }
