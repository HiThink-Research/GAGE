from queue import Queue

import pytest

from gage_eval.game_kits.phase_card_game.doudizhu.kit import build_doudizhu_game_kit
from gage_eval.game_kits.phase_card_game.doudizhu.environment import (
    DoudizhuArenaEnvironment,
)
from gage_eval.role.arena.types import ArenaAction

pytest.importorskip("rlcard")


def test_doudizhu_fast_finish_human_action():
    env = DoudizhuArenaEnvironment(
        chat_mode="all",
        fast_finish_action="finish",
        fast_finish_human_only=True,
        replay_live=False,
    )
    player_id = env.get_active_player()
    action = ArenaAction(
        player=player_id,
        move="finish",
        raw="finish",
        metadata={"player_type": "human"},
    )
    result = env.apply(action)
    assert result is not None
    assert result.winner == player_id
    assert result.result == "win"
    assert result.reason == "fast_finish"
    assert result.move_count == 1


def test_doudizhu_chat_queue_records_messages():
    chat_queue = Queue()
    env = DoudizhuArenaEnvironment(chat_mode="all", chat_queue=chat_queue, replay_live=False)
    chat_queue.put({"player_idx": 0, "text": "hello"})
    observation = env.observe(env.get_active_player())
    chat_log = observation.metadata.get("chat_log")
    assert isinstance(chat_log, list)
    assert chat_log
    assert chat_log[-1]["text"] == "hello"
    assert chat_log[-1]["player_id"] == "player_0"
    assert observation.prompt is not None
    assert observation.prompt.payload.get("game_type") == "doudizhu"


def test_doudizhu_gamekit_owns_parser_and_renderer_refs() -> None:
    game_kit = build_doudizhu_game_kit()

    assert game_kit.parser == "doudizhu_v1"
    assert game_kit.renderer == "doudizhu_replay_v1"
    assert [env_spec.parser for env_spec in game_kit.env_catalog] == [
        "doudizhu_v1",
        "doudizhu_v1",
    ]
    assert [env_spec.renderer for env_spec in game_kit.env_catalog] == [
        "doudizhu_replay_v1",
        "doudizhu_replay_v1",
    ]
