"""Shared data contracts for arena games."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional, Sequence


@dataclass(frozen=True)
class ArenaObservation:
    """Observation payload delivered to a player (active_player uses player_id)."""

    board_text: str
    legal_moves: Sequence[str]
    active_player: str
    last_move: Optional[str] = None
    metadata: dict[str, Any] = field(default_factory=dict)
    view: Optional[dict[str, Any]] = None
    context: Optional[dict[str, Any]] = None
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ArenaAction:
    """Action payload returned by a player (player uses player_id)."""

    player: str
    move: str
    raw: str
    metadata: dict[str, Any] = field(default_factory=dict)
    hold_ticks: Optional[int] = None

    @property
    def extra(self) -> dict[str, Any]:
        """Expose metadata under the retro-friendly name."""

        return self.metadata


@dataclass(frozen=True)
class GameResult:
    """Outcome summary for a completed game."""

    winner: Optional[str]
    result: str
    reason: Optional[str]
    move_count: int
    illegal_move_count: int
    final_board: str
    move_log: Sequence[dict[str, Any]]
    status: Optional[str] = None
    rule_profile: Optional[str] = None
    win_direction: Optional[str] = None
    line_length: Optional[int] = None
    replay_path: Optional[str] = None
    scores: Optional[dict[str, float]] = None
    metrics: Optional[dict[str, Any]] = None
    extra: Optional[dict[str, Any]] = None

    def __post_init__(self) -> None:
        resolved_status = self.status or self.result
        if not self.status:
            object.__setattr__(self, "status", resolved_status)
        if not self.result:
            object.__setattr__(self, "result", resolved_status)
