from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from gage_eval.game_kits.phase_card_game.doudizhu.envs.classic_3p_real import (
    Classic3pRealEnvironment,
)
from gage_eval.game_kits.phase_card_game.mahjong.envs.riichi_4p_real import (
    Riichi4pRealEnvironment,
)
from gage_eval.role.arena.types import ArenaAction


def test_classic_3p_real_environment_uses_real_rlcard_opening_hands(tmp_path: Path) -> None:
    player_specs = [
        SimpleNamespace(player_id="player_0", display_name="Player 0"),
        SimpleNamespace(player_id="player_1", display_name="Player 1"),
        SimpleNamespace(player_id="player_2", display_name="Player 2"),
    ]

    environment = Classic3pRealEnvironment(
        player_specs=player_specs,
        replay_output_dir=str(tmp_path),
        replay_filename="doudizhu_real_replay.json",
        sample_id="doudizhu_real_sample",
        illegal_policy={"retry": 0, "on_fail": "random"},
        start_player_id="player_0",
    )

    observation = environment.observe("player_0")

    assert observation.metadata["private_state"]["self_id"] == "player_0"
    assert len(observation.metadata["private_state"]["current_hand"]) >= 17
    assert sum(observation.metadata["public_state"]["num_cards_left"].values()) >= 51
    assert observation.legal_actions_items

    first_move = observation.legal_actions_items[0]
    result = environment.apply(ArenaAction(player="player_0", move=first_move, raw=first_move))

    assert result is None or result.move_count >= 1


def test_riichi_4p_real_environment_uses_real_rlcard_opening_hands(tmp_path: Path) -> None:
    player_specs = [
        SimpleNamespace(player_id="east", display_name="East"),
        SimpleNamespace(player_id="south", display_name="South"),
        SimpleNamespace(player_id="west", display_name="West"),
        SimpleNamespace(player_id="north", display_name="North"),
    ]

    environment = Riichi4pRealEnvironment(
        player_specs=player_specs,
        replay_output_dir=str(tmp_path),
        replay_filename="mahjong_real_replay.json",
        sample_id="mahjong_real_sample",
        replay_live=True,
        illegal_policy={"retry": 0, "on_fail": "random"},
    )

    observation = environment.observe("east")

    assert observation.metadata["player_id"] == "east"
    assert len(observation.metadata["private_state"]["hand"]) >= 13
    assert sum(observation.metadata["public_state"]["num_cards_left"].values()) >= 52
    assert observation.legal_actions_items

    first_move = observation.legal_actions_items[0]
    result = environment.apply(ArenaAction(player="east", move=first_move, raw=first_move))

    assert result is None or result.move_count >= 1
