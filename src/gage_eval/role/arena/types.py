"""Shared data contracts for arena games."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional, Sequence


@dataclass(frozen=True)
class ArenaObservation:
    """Observation payload delivered to a player (active_player uses player_id).

    This dataclass intentionally carries both the newer "view/legal_actions/context"
    fields as well as legacy convenience fields referenced by players/renderers.
    """

    board_text: str
    legal_moves: Sequence[str]
    active_player: str
    last_move: Optional[str] = None
    metadata: dict[str, Any] = field(default_factory=dict)
    view: Optional[dict[str, Any]] = None
    legal_actions: Optional[dict[str, Any]] = None
    context: Optional[dict[str, Any]] = None
    extra: dict[str, Any] = field(default_factory=dict)
    view_text: Optional[str] = None
    last_action: Optional[str] = None
    legal_actions_items: Sequence[str] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        """Populate compatibility fields derived from structured payloads."""

        # STEP 1: Derive view_text from the structured view payload when omitted.
        if self.view_text is None:
            view_text = None
            if isinstance(self.view, dict):
                candidate = self.view.get("text")
                if candidate is not None:
                    view_text = str(candidate)
            object.__setattr__(self, "view_text", view_text if view_text is not None else str(self.board_text))

        # STEP 2: Expose last_action as an alias of last_move when omitted.
        if self.last_action is None:
            object.__setattr__(self, "last_action", self.last_move)

        # STEP 3: Derive legal_actions_items from legal_actions["items"] or legal_moves.
        if not self.legal_actions_items:
            items: list[str] = []
            if isinstance(self.legal_actions, dict):
                raw_items = self.legal_actions.get("items")
                if isinstance(raw_items, (list, tuple)):
                    items = [str(item) for item in raw_items if item is not None]
            if not items:
                items = [str(move) for move in (self.legal_moves or []) if move is not None]
            object.__setattr__(self, "legal_actions_items", tuple(items))


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
