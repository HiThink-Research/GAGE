"""Core factory for Doudizhu card game."""

from __future__ import annotations

from typing import Any, Optional

from gage_eval.role.arena.games.doudizhu.cores.base import AbstractGameCore
from gage_eval.role.arena.games.doudizhu.cores.rlcard_core import RLCardCore


def make_core(
    game_type: str, *, config: Optional[dict[str, Any]] = None
) -> AbstractGameCore:
    """Create a card game core for Doudizhu.

    Args:
        game_type: Game identifier, expected to be "doudizhu".
        config: Optional core configuration payload.

    Returns:
        Initialized RLCard core instance.
    """

    if str(game_type) != "doudizhu":
        raise ValueError(f"Unsupported game_type for Doudizhu core: {game_type}")
    return RLCardCore("doudizhu", config=config)


__all__ = ["make_core"]
