from __future__ import annotations

from typing import Any, Sequence

from gage_eval.role.arena.resources.runtime_bridge import attach_runtime_resources
from gage_eval.role.arena.games.gomoku.env import GomokuArenaEnvironment


class GomokuStandardEnvironment:
    """GameKit adapter that reuses the legacy Gomoku local environment."""

    def __init__(
        self,
        *,
        board_size: int,
        win_len: int,
        coord_scheme: str,
        rule_profile: str,
        win_directions: Sequence[str],
        illegal_policy: dict[str, str | int] | None,
        player_specs: Sequence[object],
        start_player_id: str | None = None,
    ) -> None:
        player_ids = [str(getattr(player, "player_id")) for player in player_specs]
        player_names = {
            str(getattr(player, "player_id")): str(getattr(player, "display_name"))
            for player in player_specs
        }
        self._legacy = GomokuArenaEnvironment(
            board_size=board_size,
            win_len=win_len,
            player_ids=player_ids,
            player_names=player_names,
            start_player_id=start_player_id or (player_ids[0] if player_ids else None),
            coord_scheme=coord_scheme,
            rule_profile=rule_profile,
            win_directions=win_directions,
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
            board_size=int(defaults.get("board_size", 15)),
            win_len=int(defaults.get("win_len", 5)),
            coord_scheme=str(defaults.get("coord_scheme", "A1")),
            rule_profile=str(defaults.get("rule_profile", "freestyle")),
            win_directions=tuple(
                str(direction)
                for direction in defaults.get(
                    "win_directions",
                    ("horizontal", "vertical", "diagonal", "anti_diagonal"),
                )
            ),
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

    def is_terminal(self) -> bool:
        return self._legacy.is_terminal()

    def build_result(self, *, result: str, reason: str | None):
        return self._legacy.build_result(result=result, reason=reason)


def build_gomoku_standard_environment(*, sample, resolved, resources, player_specs) -> Any:
    return GomokuStandardEnvironment.from_runtime(
        sample=sample,
        resolved=resolved,
        resources=resources,
        player_specs=player_specs,
    )
