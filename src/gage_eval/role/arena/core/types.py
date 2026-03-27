
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class ArenaStopReason(str, Enum):
    COMPLETED = "completed"
    ERROR = "error"
    TIMEOUT = "timeout"


@dataclass(frozen=True)
class ArenaSample:
    game_kit: str
    env: str | None
    scheduler: str | None = None
    players: tuple[dict[str, object], ...] = field(default_factory=tuple)
    runtime_overrides: dict[str, object] = field(default_factory=dict)
