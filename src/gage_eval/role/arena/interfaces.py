"""Protocols for arena environments, players, and schedulers."""

from __future__ import annotations

from typing import Any, Iterable, Optional, Protocol, Sequence

from gage_eval.role.arena.types import ArenaAction, ArenaObservation, GameResult


class ArenaEnvironment(Protocol):
    """Defines the minimal interface for a game environment."""

    def reset(self) -> None:
        """Reset the environment to its initial state."""

    def get_active_player(self) -> str:
        """Return the player_id of the participant who should act next."""

    def observe(self, player: str) -> ArenaObservation:
        """Return an observation for the given player_id."""

    def apply(self, action: ArenaAction) -> Optional[GameResult]:
        """Apply an action and return GameResult if the game ends."""

    def is_terminal(self) -> bool:
        """Return True if the game has ended."""

    def build_result(self, *, status: str, reason: Optional[str]) -> GameResult:
        """Build a GameResult snapshot when the game ends."""


class RuleEngine(Protocol):
    """Defines a rule validation interface for arena games."""

    def validate(self, action: ArenaAction) -> tuple[bool, Optional[str]]:
        """Validate an action and return (is_legal, reason)."""

    def check_terminal(self) -> Optional[GameResult]:
        """Return GameResult if the game has ended."""


class Player(Protocol):
    """Defines the minimal player interface."""

    name: str

    def think(self, observation: ArenaObservation) -> ArenaAction:
        """Return an action based on the observation."""


class MoveParser(Protocol):
    """Defines the minimal parser interface for arena games."""

    def parse(
        self,
        text: str,
        *,
        legal_moves: Optional[Iterable[Any]] = None,
    ) -> Any:
        """Parse a move from raw text and optionally validate it."""

    def build_rethink_prompt(
        self,
        *,
        last_output: str,
        reason: str,
        legal_moves: Sequence[str],
    ) -> str:
        """Build a retry prompt for invalid moves."""


class Scheduler(Protocol):
    """Defines a scheduler that runs the game loop."""

    def run_loop(self, environment: ArenaEnvironment, players: Sequence[Player]) -> GameResult:
        """Run the game loop and return the final result."""
