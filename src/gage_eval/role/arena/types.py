"""Shared data contracts for arena games."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, replace
from typing import Any, Optional, Sequence


@dataclass(frozen=True)
class ArenaPromptSpec:
    """Game-owned prompt package consumed by arena players."""

    instruction: str
    payload: dict[str, Any] = field(default_factory=dict)
    renderer_instruction: Optional[str] = None


@dataclass(frozen=True)
class ArenaObservation:
    """Observation payload delivered to a player (active_player uses player_id)."""

    board_text: str
    legal_moves: Sequence[str]
    active_player: str
    last_move: Optional[str] = None
    metadata: dict[str, Any] = field(default_factory=dict)
    view: Optional[dict[str, Any]] = None
    legal_actions: Optional[dict[str, Any]] = None
    context: Optional[dict[str, Any]] = None
    prompt: Optional[ArenaPromptSpec] = None

    @property
    def view_text(self) -> str:
        view = self.view or {}
        text = view.get("text")
        if text is None:
            return self.board_text
        return str(text)

    @property
    def legal_actions_items(self) -> Sequence[str]:
        legal_actions = self.legal_actions or {}
        items = legal_actions.get("items")
        if isinstance(items, Sequence) and not isinstance(items, (str, bytes)):
            return [str(item) for item in items]
        return list(self.legal_moves)

    @property
    def last_action(self) -> Optional[str]:
        if self.last_move:
            return self.last_move
        last_move = self.metadata.get("last_move")
        if last_move is None:
            return None
        return str(last_move)


@dataclass(frozen=True)
class ArenaAction:
    """Action payload returned by a player (player uses player_id)."""

    player: str
    move: str
    raw: str
    metadata: dict[str, Any] = field(default_factory=dict)


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
    rule_profile: Optional[str] = None
    win_direction: Optional[str] = None
    line_length: Optional[int] = None
    replay_path: Optional[str] = None
    arena_trace: Sequence[dict[str, Any]] = field(default_factory=tuple)


def attach_arena_trace(
    result: GameResult,
    arena_trace: Sequence[dict[str, Any]],
) -> GameResult:
    """Attach scheduler-produced arena trace to a GameResult.

    Args:
        result: The immutable GameResult instance returned by the environment.
        arena_trace: Ordered per-step trace entries produced by the scheduler.

    Returns:
        A new GameResult instance with ``arena_trace`` populated.
    """

    return replace(result, arena_trace=tuple(arena_trace))


@dataclass(frozen=True)
class ArenaTraceStep:
    """Canonical per-step arena trace contract."""

    step_index: int
    trace_state: str
    timestamp: int
    player_id: str
    action_raw: Any
    action_applied: Any
    t_obs_ready_ms: int
    t_action_submitted_ms: int
    timeout: bool
    is_action_legal: bool
    retry_count: int
    illegal_reason: Optional[str] = None
    info: Optional[dict[str, Any]] = None
    reward: Optional[dict[str, float]] = None
    timeline_id: Optional[str] = None
    deadline_ms: Optional[int] = None

    def to_dict(self) -> dict[str, Any]:
        """Serializes the trace step to dict."""

        return asdict(self)


@dataclass(frozen=True)
class ArenaFooter:
    """Canonical arena footer under `predict_result[0].game_arena`."""

    end_time_ms: int
    total_steps: int
    winner_player_id: Optional[str]
    termination_reason: str
    ranks: Optional[list[Any]] = None
    final_scores: Optional[dict[str, float]] = None
    episode_returns: Optional[dict[str, float]] = None

    def to_dict(self) -> dict[str, Any]:
        """Serializes footer to dict."""

        return asdict(self)


@dataclass(frozen=True)
class ArenaHeader:
    """Canonical arena header under `sample.metadata.game_arena`."""

    engine_id: str
    seed: int
    mode: str
    players: list[dict[str, Any]]
    start_time_ms: int

    def to_dict(self) -> dict[str, Any]:
        """Serializes header to dict."""

        return asdict(self)
