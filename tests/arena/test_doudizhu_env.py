from queue import Queue

import pytest

from gage_eval.role.arena.games.doudizhu.env import DoudizhuArenaEnvironment
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
