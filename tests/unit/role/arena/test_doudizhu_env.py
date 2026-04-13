from queue import Queue
from types import SimpleNamespace

import pytest

from gage_eval.game_kits.phase_card_game.doudizhu.kit import build_doudizhu_game_kit
from gage_eval.game_kits.phase_card_game.doudizhu.envs.classic_3p_real import (
    Classic3pRealEnvironment,
)
from gage_eval.game_kits.phase_card_game.doudizhu.environment import (
    DoudizhuArenaEnvironment,
)
from gage_eval.role.arena.core.invocation import GameArenaInvocationContext
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


def test_doudizhu_live_frame_drains_chat_queue_without_next_observation() -> None:
    chat_queue = Queue()
    env = DoudizhuArenaEnvironment(chat_mode="all", chat_queue=chat_queue, replay_live=False)

    chat_queue.put({"player_id": "player_0", "text": "bubble now"})
    last_frame = env.get_last_frame()

    assert isinstance(last_frame.get("chat_log"), list)
    assert last_frame["chat_log"] == [{"player_id": "player_0", "text": "bubble now"}]
    assert last_frame["ui_state"]["chat_log"] == [{"player_id": "player_0", "text": "bubble now"}]


def test_doudizhu_illegal_result_preserves_queued_chat_messages() -> None:
    chat_queue = Queue()
    env = DoudizhuArenaEnvironment(chat_mode="all", chat_queue=chat_queue, replay_live=False)

    active_player = env.get_active_player()
    chat_queue.put({"player_id": active_player, "text": "still visible"})
    result = env.apply(ArenaAction(player=active_player, move="not_a_move", raw="not_a_move"))

    assert result is not None
    assert "still visible" in result.final_board
    last_frame = env.get_last_frame()
    assert last_frame["chat_log"] == [{"player_id": active_player, "text": "still visible"}]


def test_doudizhu_observe_metadata_includes_ui_state_for_live_visualization() -> None:
    env = DoudizhuArenaEnvironment(replay_live=False)

    observation = env.observe(env.get_active_player())
    last_frame = env.get_last_frame()

    assert isinstance(observation.metadata.get("ui_state"), dict)
    assert observation.metadata["ui_state"] == last_frame["ui_state"]
    assert observation.metadata["ui_state"]["active_player_id"] == env.get_active_player()


def test_doudizhu_real_environment_reuses_runtime_chat_queue_for_live_visualization() -> None:
    shared_chat_queue = Queue()
    runtime_service_hub = SimpleNamespace(
        peek_action_server=lambda: SimpleNamespace(chat_queue=shared_chat_queue),
    )
    invocation_context = GameArenaInvocationContext(
        adapter_id="arena",
        sample_id="sample-1",
        runtime_service_hub=runtime_service_hub,
    )
    player_specs = tuple(
        SimpleNamespace(player_id=f"player_{index}", display_name=f"Player {index}")
        for index in range(3)
    )
    resolved = SimpleNamespace(
        game_kit=SimpleNamespace(defaults={}),
        env_spec=SimpleNamespace(defaults={}),
    )
    resources = SimpleNamespace(resource_spec=None, game_runtime=None, game_bridge=None)
    sample = SimpleNamespace(runtime_overrides={})

    env = Classic3pRealEnvironment.from_runtime(
        sample=sample,
        resolved=resolved,
        resources=resources,
        player_specs=player_specs,
        invocation_context=invocation_context,
    )

    assert env._arena._chat_queue is shared_chat_queue


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
