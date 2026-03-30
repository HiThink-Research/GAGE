from __future__ import annotations

from typing import Any, Sequence

from gage_eval.game_kits.phase_card_game.mahjong.environment import MahjongArena
from gage_eval.role.arena.resources.runtime_bridge import attach_runtime_resources
from gage_eval.role.arena.replay_paths import resolve_invocation_run_sample_ids


class Riichi4pRealEnvironment:
    """GameKit adapter that reuses the real RLCard-backed Mahjong arena."""

    def __init__(
        self,
        *,
        player_specs: Sequence[object],
        replay_output_dir: str | None,
        replay_filename: str | None,
        run_id: str | None = None,
        sample_id: str | None = None,
        illegal_policy: dict[str, str | int] | None = None,
        replay_live: bool = False,
        chat_mode: str = "off",
        chat_every_n: int = 1,
    ) -> None:
        player_ids = [str(getattr(player, "player_id")) for player in player_specs]
        player_names = {
            str(getattr(player, "player_id")): str(getattr(player, "display_name"))
            for player in player_specs
        }
        self._arena = MahjongArena(
            game_type="mahjong",
            player_ids=player_ids,
            player_names=player_names,
            illegal_policy=illegal_policy,
            run_id=run_id,
            sample_id=sample_id,
            replay_output_dir=str(replay_output_dir) if replay_output_dir else None,
            replay_filename=str(replay_filename or "mahjong_riichi_4p_real_replay.json"),
            replay_live=bool(replay_live),
            chat_mode=str(chat_mode or "off"),
            chat_every_n=int(chat_every_n),
        )

    @classmethod
    def from_runtime(cls, *, sample, resolved, resources, player_specs, invocation_context=None):
        defaults = {
            **dict(resolved.game_kit.defaults),
            **dict(resolved.env_spec.defaults),
            **dict(sample.runtime_overrides or {}),
        }
        run_id, sample_id = resolve_invocation_run_sample_ids(
            invocation_context=invocation_context,
            run_id=defaults.get("run_id"),
            sample_id=defaults.get("sample_id"),
        )
        environment = cls(
            player_specs=player_specs,
            replay_output_dir=defaults.get("replay_output_dir"),
            replay_filename=defaults.get("replay_filename"),
            run_id=run_id,
            sample_id=sample_id,
            illegal_policy=defaults.get("illegal_policy"),
            replay_live=bool(defaults.get("replay_live", False)),
            chat_mode=str(defaults.get("chat_mode", "off")),
            chat_every_n=int(defaults.get("chat_every_n", 1)),
        )
        return attach_runtime_resources(environment, resources)

    def get_active_player(self) -> str:
        return self._arena.get_active_player()

    def observe(self, player: str):
        return self._arena.observe(player)

    def apply(self, action):
        return self._arena.apply(action)

    def get_last_frame(self):
        return self._arena.get_last_frame()

    def is_terminal(self) -> bool:
        return self._arena.is_terminal()

    def build_result(self, *, result: str, reason: str | None):
        return self._arena.build_result(result=result, reason=reason)


def build_riichi_4p_real_environment(
    *,
    sample,
    resolved,
    resources,
    player_specs,
    invocation_context=None,
) -> Any:
    return Riichi4pRealEnvironment.from_runtime(
        sample=sample,
        resolved=resolved,
        resources=resources,
        player_specs=player_specs,
        invocation_context=invocation_context,
    )
