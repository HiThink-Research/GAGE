"""Core factory for Mahjong card game."""

from __future__ import annotations

from typing import Any, Optional

from gage_eval.role.arena.games.mahjong.cores.base import AbstractGameCore
from gage_eval.role.arena.games.mahjong.cores.rlcard_mahjong import RLCardCore


def make_core(
    game_type: str, *, config: Optional[dict[str, Any]] = None
) -> AbstractGameCore:
    """Create a card game core for Mahjong.

    Args:
        game_type: Game identifier, expected to be "mahjong".
        config: Optional core configuration payload.

    Returns:
        Initialized RLCard core instance.
    """

    if str(game_type) != "mahjong":
        raise ValueError(f"Unsupported game_type for Mahjong core: {game_type}")
    return RLCardCore("mahjong", config=config)


__all__ = ["make_core"]
