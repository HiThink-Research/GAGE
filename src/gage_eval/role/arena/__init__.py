"""Game arena runtime package."""

from __future__ import annotations

from gage_eval.role.arena.core.arena_core import GameArenaCore
from gage_eval.role.arena.core.game_session import GameSession
from gage_eval.role.arena.core.types import ArenaSample, ArenaStopReason
from gage_eval.role.arena.output.models import ArenaOutput
from gage_eval.role.arena.output.writer import ArenaOutputWriter

__all__ = [
    "ArenaOutput",
    "ArenaOutputWriter",
    "ArenaSample",
    "ArenaStopReason",
    "GameArenaCore",
    "GameSession",
]
