"""Game arena runtime package."""

from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING

if TYPE_CHECKING:
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

_LAZY_EXPORTS = {
    "ArenaOutput": ("gage_eval.role.arena.output.models", "ArenaOutput"),
    "ArenaOutputWriter": ("gage_eval.role.arena.output.writer", "ArenaOutputWriter"),
    "ArenaSample": ("gage_eval.role.arena.core.types", "ArenaSample"),
    "ArenaStopReason": ("gage_eval.role.arena.core.types", "ArenaStopReason"),
    "GameArenaCore": ("gage_eval.role.arena.core.arena_core", "GameArenaCore"),
    "GameSession": ("gage_eval.role.arena.core.game_session", "GameSession"),
}


def __getattr__(name: str):
    try:
        module_name, attr_name = _LAZY_EXPORTS[name]
    except KeyError as exc:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from exc
    value = getattr(import_module(module_name), attr_name)
    globals()[name] = value
    return value
