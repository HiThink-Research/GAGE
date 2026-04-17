from __future__ import annotations

from gage_eval.game_kits.registry import GameKitRegistry
from gage_eval.game_kits.runtime_binding import RuntimeBindingResolver
from gage_eval.role.arena.core.types import ArenaSample


def test_gymnasium_atari_space_invaders_resolves_single_player_frame_kit() -> None:
    resolver = RuntimeBindingResolver(game_kits=GameKitRegistry())

    resolved = resolver.resolve(ArenaSample(game_kit="gymnasium_atari", env="space_invaders"))

    assert resolved.game_kit.kit_id == "gymnasium_atari"
    assert resolved.env_spec.env_id == "space_invaders"
    assert resolved.scheduler.binding_id == "real_time_tick/default"
    assert resolved.game_kit.seat_spec == {"seats": ("pilot_0",)}
    assert resolved.visualization_spec.plugin_id == "arena.visualization.pettingzoo.frame_v1"
    assert resolved.parser == (
        "gage_eval.game_kits.aec_env_game.pettingzoo.action_codec.DiscreteActionParser"
    )
    assert resolved.input_mapper == (
        "gage_eval.game_kits.aec_env_game.pettingzoo.input_mapper.PettingZooDiscreteInputMapper"
    )
