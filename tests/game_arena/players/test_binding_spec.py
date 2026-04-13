from __future__ import annotations

import importlib

from gage_eval.role.arena.core.types import ArenaSample


def test_runtime_binding_resolver_normalizes_minimal_player_configs(
    fake_game_kit_registry,
) -> None:
    runtime_binding = importlib.import_module("gage_eval.game_kits.runtime_binding")

    assert hasattr(runtime_binding, "PlayerDriverRegistry"), "PlayerDriverRegistry missing"

    resolver = runtime_binding.RuntimeBindingResolver(
        game_kits=fake_game_kit_registry,
        player_drivers=runtime_binding.PlayerDriverRegistry(),
    )
    sample = ArenaSample(
        game_kit="retro_platformer_players_v1",
        env="retro_mario",
        players=(
            {"seat": "player_0", "player_kind": "llm", "backend_id": "qwen_backend"},
            {"seat": "player_1", "player_kind": "human"},
            {"seat": "enemy_bot", "player_kind": "dummy", "actions": ["RIGHT", "FIRE"]},
        ),
    )

    resolved = resolver.resolve(sample)

    assert hasattr(resolved, "player_bindings"), "Resolved runtime binding missing player_bindings"
    assert [binding.driver_id for binding in resolved.player_bindings] == [
        "player_driver/llm_backend",
        "player_driver/human_local_input",
        "player_driver/dummy",
    ]
    assert [binding.player_id for binding in resolved.player_bindings] == [
        "player_0",
        "player_1",
        "enemy_bot",
    ]
