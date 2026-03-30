"""Core factory for Mahjong card game."""

from __future__ import annotations

from typing import Any, Optional

from gage_eval.game_kits.phase_card_game.mahjong.cores.base import AbstractGameCore


def make_core(
    game_type: str, *, config: Optional[dict[str, Any]] = None
) -> AbstractGameCore:
    """Create a card game core for Mahjong."""

    if str(game_type) != "mahjong":
        raise ValueError(f"Unsupported game_type for Mahjong core: {game_type}")

    from gage_eval.game_kits.phase_card_game.mahjong.cores.rlcard_mahjong import (
        RLCardCore,
    )

    return RLCardCore("mahjong", config=config)


__all__ = ["make_core"]
