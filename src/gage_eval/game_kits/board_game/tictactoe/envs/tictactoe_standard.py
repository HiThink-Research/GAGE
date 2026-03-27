from __future__ import annotations

from typing import Any, Sequence

from gage_eval.role.arena.resources.runtime_bridge import attach_runtime_resources
from gage_eval.role.arena.games.tictactoe.env import TicTacToeArenaEnvironment


class TicTacToeStandardEnvironment:
    """GameKit adapter that reuses the legacy Tic-Tac-Toe local environment."""

    def __init__(
        self,
        *,
        board_size: int,
        coord_scheme: str,
        illegal_policy: dict[str, str | int] | None,
        player_specs: Sequence[object],
        start_player_id: str | None = None,
    ) -> None:
        player_ids = [str(getattr(player, "player_id")) for player in player_specs]
        player_names = {
            str(getattr(player, "player_id")): str(getattr(player, "display_name"))
            for player in player_specs
        }
        self._legacy = TicTacToeArenaEnvironment(
            board_size=board_size,
            player_ids=player_ids,
            player_names=player_names,
            start_player_id=start_player_id or (player_ids[0] if player_ids else None),
            coord_scheme=coord_scheme,
            illegal_policy=illegal_policy,
        )

    @classmethod
    def from_runtime(cls, *, sample, resolved, resources, player_specs):
        defaults = {
            **dict(resolved.game_kit.defaults),
            **dict(resolved.env_spec.defaults),
            **dict(sample.runtime_overrides or {}),
        }
        environment = cls(
            board_size=int(defaults.get("board_size", 3)),
            coord_scheme=str(defaults.get("coord_scheme", "ROW_COL")),
            illegal_policy=defaults.get("illegal_policy"),
            player_specs=player_specs,
            start_player_id=defaults.get("start_player_id"),
        )
        return attach_runtime_resources(environment, resources)

    def get_active_player(self) -> str:
        return self._legacy.get_active_player()

    def observe(self, player: str):
        return self._legacy.observe(player)

    def apply(self, action):
        return self._legacy.apply(action)

    def get_last_frame(self):
        return self._legacy.get_last_frame()

    def is_terminal(self) -> bool:
        return self._legacy.is_terminal()

    def build_result(self, *, result: str, reason: str | None):
        return self._legacy.build_result(result=result, reason=reason)


def build_tictactoe_standard_environment(*, sample, resolved, resources, player_specs) -> Any:
    return TicTacToeStandardEnvironment.from_runtime(
        sample=sample,
        resolved=resolved,
        resources=resources,
        player_specs=player_specs,
    )
