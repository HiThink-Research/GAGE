from __future__ import annotations

from types import SimpleNamespace

from gage_eval.game_kits.real_time_game.vizdoom.environment import ViZDoomEnvConfig
from gage_eval.game_kits.real_time_game.vizdoom.envs.duel_map01 import DuelMap01Environment


def test_duel_map01_from_runtime_uses_vizdoom_env_defaults_for_reset_recovery() -> None:
    sample = SimpleNamespace(runtime_overrides={})
    resolved = SimpleNamespace(
        game_kit=SimpleNamespace(defaults={"backend_mode": "dummy"}),
        env_spec=SimpleNamespace(defaults={}),
    )
    resources = SimpleNamespace(game_runtime=None)
    player_specs = [
        SimpleNamespace(player_id="p0", display_name="doom_alpha"),
        SimpleNamespace(player_id="p1", display_name="doom_beta"),
    ]

    environment = DuelMap01Environment.from_runtime(
        sample=sample,
        resolved=resolved,
        resources=resources,
        player_specs=player_specs,
    )

    assert environment._cfg.reset_retry_count == ViZDoomEnvConfig.reset_retry_count
    assert (
        environment._cfg.death_check_warmup_steps
        == ViZDoomEnvConfig.death_check_warmup_steps
    )
