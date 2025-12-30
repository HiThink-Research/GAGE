"""Turn-based scheduler for arena games."""

from __future__ import annotations

from typing import Optional, Sequence

from gage_eval.role.arena.interfaces import ArenaEnvironment, Player, Scheduler
from gage_eval.role.arena.types import GameResult


class TurnScheduler(Scheduler):
    """Stop-and-wait scheduler for turn-based games."""

    def __init__(self, *, max_turns: Optional[int] = None) -> None:
        """Initialize the turn scheduler.

        Args:
            max_turns: Optional maximum number of turns before forcing a draw.
        """

        self._max_turns = max_turns if max_turns is None else max(1, int(max_turns))

    def run_loop(self, environment: ArenaEnvironment, players: Sequence[Player]) -> GameResult:
        """Run the game loop until termination and return the final result."""

        players_by_name = {player.name: player for player in players}
        if not players_by_name:
            raise ValueError("TurnScheduler requires at least one player")

        environment.reset()
        turn = 0

        # STEP 1: Iterate until terminal state or max turns reached.
        while not environment.is_terminal():
            if self._max_turns is not None and turn >= self._max_turns:
                return environment.build_result(result="draw", reason="max_turns")

            active_player = environment.get_active_player()
            player = players_by_name.get(active_player)
            if player is None:
                raise KeyError(f"Missing player '{active_player}' for environment")

            observation = environment.observe(active_player)
            action = player.think(observation)
            outcome = environment.apply(action)
            if outcome is not None:
                return outcome

            turn += 1

        return environment.build_result(result="draw", reason="terminated")
