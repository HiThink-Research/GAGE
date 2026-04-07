from __future__ import annotations

import json
from queue import Queue

import pytest

import gage_eval.role.arena.player_drivers.human_local_input as human_local_input_module
from gage_eval.role.arena.core.players import PlayerBindingSpec
from gage_eval.role.arena.human_input_protocol import (
    LatestActionMailbox,
    build_action_payload,
    dump_action_payload,
)
from gage_eval.role.arena.player_drivers.human_local_input import LocalHumanInputDriver
from gage_eval.role.arena.types import ArenaObservation


def build_scheduler_owned_queued_command_player(*, action_queue: Queue[str]):
    return LocalHumanInputDriver(
        driver_id="player_driver/human_local_input",
        family="human",
    ).bind(
        PlayerBindingSpec(
            seat="player_0",
            player_id="player_0",
            player_kind="human",
            driver_id="player_driver/human_local_input",
            driver_params={
                "action_queue": action_queue,
                "input_semantics": "queued_command",
                "tick_interval_ms": 50,
                "timeout_fallback_move": "noop",
                "scheduler_owned_realtime": True,
            },
        )
    )


def test_local_human_input_driver_preserves_structured_raw_payload_and_metadata() -> None:
    action_queue: Queue[str] = Queue()
    action_queue.put(
        dump_action_payload(
            build_action_payload(
                action="right_jump",
                player_id="player_0",
                sample_id="retro-smoke",
                metadata={"hold_ticks": 6, "source": "arena_visual"},
            )
        )
    )
    driver = LocalHumanInputDriver(
        driver_id="player_driver/human_local_input",
        family="human",
    )
    player = driver.bind(
        PlayerBindingSpec(
            seat="player_0",
            player_id="player_0",
            player_kind="human",
            driver_id="player_driver/human_local_input",
            driver_params={"action_queue": action_queue},
        )
    )

    action = player.next_action(
        ArenaObservation(
            board_text="retro frame",
            legal_moves=("noop", "right", "right_jump"),
            active_player="player_0",
        )
    )

    assert action.move == "right_jump"
    assert json.loads(action.raw) == {
        "action": "right_jump",
        "hold_ticks": 6,
        "source": "arena_visual",
    }
    assert action.metadata == {
        "driver_id": "player_driver/human_local_input",
        "player_type": "human",
        "input_semantics": "queued_command",
        "hold_ticks": 6,
        "source": "arena_visual",
    }


def test_local_human_input_driver_reuses_last_stateful_action_until_changed() -> None:
    action_queue: Queue[str] = Queue()
    action_queue.put(
        dump_action_payload(
            build_action_payload(
                action="right",
                player_id="player_0",
                sample_id="retro-smoke",
                metadata={"hold_ticks": 1, "input_mode": "stateful"},
            )
        )
    )
    driver = LocalHumanInputDriver(
        driver_id="player_driver/human_local_input",
        family="human",
    )
    player = driver.bind(
        PlayerBindingSpec(
            seat="player_0",
            player_id="player_0",
            player_kind="human",
            driver_id="player_driver/human_local_input",
            driver_params={"action_queue": action_queue, "stateful_actions": True},
        )
    )
    observation = ArenaObservation(
        board_text="retro frame",
        legal_moves=("noop", "right", "right_jump"),
        active_player="player_0",
    )

    first_action = player.next_action(observation)
    second_action = player.next_action(observation)

    assert first_action.move == "right"
    assert second_action.move == "right"
    assert json.loads(second_action.raw) == {
        "action": "right",
        "hold_ticks": 1,
        "input_mode": "stateful",
    }

    action_queue.put(
        dump_action_payload(
            build_action_payload(
                action="noop",
                player_id="player_0",
                sample_id="retro-smoke",
                metadata={"input_mode": "stateful"},
            )
        )
    )

    released_action = player.next_action(observation)

    assert released_action.move == "noop"


def test_local_human_input_driver_continuous_state_mode_does_not_sleep_between_ticks(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    action_queue = LatestActionMailbox()
    action_queue.put(
        dump_action_payload(
            build_action_payload(
                action="right",
                player_id="player_0",
                sample_id="retro-smoke",
                metadata={"hold_ticks": 1, "input_mode": "stateful"},
            )
        )
    )
    driver = LocalHumanInputDriver(
        driver_id="player_driver/human_local_input",
        family="human",
    )
    player = driver.bind(
        PlayerBindingSpec(
            seat="player_0",
            player_id="player_0",
            player_kind="human",
            driver_id="player_driver/human_local_input",
            driver_params={"action_queue": action_queue, "stateful_actions": True, "timeout_ms": 16},
        )
    )
    observation = ArenaObservation(
        board_text="retro frame",
        legal_moves=("noop", "right", "right_jump"),
        active_player="player_0",
    )

    current_time = {"value": 100.0}
    def fake_monotonic() -> float:
        return current_time["value"]

    monkeypatch.setattr(human_local_input_module.time, "monotonic", fake_monotonic)

    first_action = player.next_action(observation)

    current_time["value"] = 100.005
    action_queue.put(
        dump_action_payload(
            build_action_payload(
                action="noop",
                player_id="player_0",
                sample_id="retro-smoke",
                metadata={"input_mode": "stateful"},
            )
        )
    )
    second_action = player.next_action(observation)

    assert first_action.move == "right"
    assert second_action.move == "noop"
    assert current_time["value"] == pytest.approx(100.005, abs=1e-9)


def test_local_human_input_driver_scheduler_owned_realtime_returns_latest_state_immediately() -> None:
    action_queue = LatestActionMailbox()
    action_queue.put(
        dump_action_payload(
            build_action_payload(
                action="right",
                player_id="player_0",
                sample_id="retro-smoke",
                metadata={"input_seq": 1},
            )
        )
    )
    player = LocalHumanInputDriver(
        driver_id="player_driver/human_local_input",
        family="human",
    ).bind(
        PlayerBindingSpec(
            seat="player_0",
            player_id="player_0",
            player_kind="human",
            driver_id="player_driver/human_local_input",
            driver_params={
                "action_queue": action_queue,
                "input_semantics": "continuous_state",
                "tick_interval_ms": 16,
                "timeout_fallback_move": "noop",
                "scheduler_owned_realtime": True,
            },
        )
    )
    observation = ArenaObservation(
        board_text="retro frame",
        legal_moves=("noop", "right", "right_jump"),
        active_player="player_0",
    )

    first_action = player.next_action(observation)
    second_action = player.next_action(observation)

    assert first_action.move == "right"
    assert second_action.move == "right"

    action_queue.put(
        dump_action_payload(
            build_action_payload(
                action="noop",
                player_id="player_0",
                sample_id="retro-smoke",
                metadata={"input_seq": 2},
            )
        )
    )

    third_action = player.next_action(observation)

    assert third_action.move == "noop"


def test_local_human_input_driver_scheduler_owned_queued_command_returns_no_action_for_empty_queue() -> None:
    action_queue: Queue[str] = Queue()
    player = build_scheduler_owned_queued_command_player(action_queue=action_queue)
    observation = ArenaObservation(
        board_text="realtime frame",
        legal_moves=("noop", "issue_command"),
        active_player="player_0",
    )

    assert player.poll_scheduler_owned_action(observation) is None


def test_local_human_input_driver_scheduler_owned_queued_command_drains_fifo_commands() -> None:
    action_queue: Queue[str] = Queue()
    action_queue.put(
        dump_action_payload(build_action_payload(action="bridge_input", player_id="player_0"))
    )
    action_queue.put(
        dump_action_payload(build_action_payload(action="issue_command", player_id="player_0"))
    )

    player = build_scheduler_owned_queued_command_player(action_queue=action_queue)
    observation = ArenaObservation(
        board_text="frame",
        legal_moves=("bridge_input", "issue_command"),
        active_player="player_0",
    )

    drained = player.drain_scheduler_owned_actions(observation=observation, max_items=4)

    assert [action.move for action in drained] == ["bridge_input", "issue_command"]


def test_local_human_input_driver_scheduler_owned_queued_command_returns_empty_list_when_queue_is_empty() -> None:
    player = build_scheduler_owned_queued_command_player(action_queue=Queue())
    observation = ArenaObservation(
        board_text="frame",
        legal_moves=("bridge_input",),
        active_player="player_0",
    )

    assert player.drain_scheduler_owned_actions(observation=observation, max_items=4) == []


def test_local_human_input_driver_continuous_state_async_waits_for_scheduler_deadline(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    action_queue = LatestActionMailbox()
    action_queue.put(
        dump_action_payload(
            build_action_payload(
                action="right",
                player_id="player_0",
                sample_id="retro-smoke",
                metadata={"hold_ticks": 1, "input_mode": "stateful", "input_seq": 1},
            )
        )
    )
    driver = LocalHumanInputDriver(
        driver_id="player_driver/human_local_input",
        family="human",
    )
    player = driver.bind(
        PlayerBindingSpec(
            seat="player_0",
            player_id="player_0",
            player_kind="human",
            driver_id="player_driver/human_local_input",
            driver_params={"action_queue": action_queue, "stateful_actions": True, "timeout_ms": 16},
        )
    )
    observation = ArenaObservation(
        board_text="retro frame",
        legal_moves=("noop", "right", "right_jump"),
        active_player="player_0",
    )

    current_time = {"value": 100.0}

    def fake_monotonic() -> float:
        return current_time["value"]

    monkeypatch.setattr(human_local_input_module.time, "monotonic", fake_monotonic)

    assert player.start_thinking(observation, deadline_ms=16) is True
    assert player.has_action() is False

    current_time["value"] = 100.017
    assert player.has_action() is True
    first_action = player.pop_action()
    assert first_action.move == "right"

    assert player.start_thinking(observation, deadline_ms=16) is True
    assert player.has_action() is False

    action_queue.put(
        dump_action_payload(
            build_action_payload(
                action="right_jump",
                player_id="player_0",
                sample_id="retro-smoke",
                metadata={"hold_ticks": 1, "input_mode": "stateful", "input_seq": 2},
            )
        )
    )

    assert player.has_action() is False
    current_time["value"] = 100.034
    assert player.has_action() is True
    second_action = player.pop_action()
    assert second_action.move == "right_jump"


def test_local_human_input_driver_continuous_state_async_reuses_last_state_at_deadline(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    action_queue = LatestActionMailbox()
    action_queue.put(
        dump_action_payload(
            build_action_payload(
                action="right",
                player_id="player_0",
                sample_id="retro-smoke",
                metadata={"input_seq": 1},
            )
        )
    )
    player = LocalHumanInputDriver(
        driver_id="player_driver/human_local_input",
        family="human",
    ).bind(
        PlayerBindingSpec(
            seat="player_0",
            player_id="player_0",
            player_kind="human",
            driver_id="player_driver/human_local_input",
            driver_params={
                "action_queue": action_queue,
                "input_semantics": "continuous_state",
                "timeout_ms": 16,
            },
        )
    )
    observation = ArenaObservation(
        board_text="retro frame",
        legal_moves=("noop", "right", "right_jump"),
        active_player="player_0",
    )
    current_time = {"value": 100.0}

    monkeypatch.setattr(
        human_local_input_module.time,
        "monotonic",
        lambda: current_time["value"],
    )

    assert player.start_thinking(observation, deadline_ms=16) is True
    current_time["value"] = 100.017
    assert player.has_action() is True
    first_action = player.pop_action()
    assert first_action.move == "right"

    assert player.start_thinking(observation, deadline_ms=16) is True
    current_time["value"] = 100.034
    assert player.has_action() is True
    second_action = player.pop_action()
    assert second_action.move == "right"


def test_local_human_input_driver_discrete_mode_supports_async_timeout_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    action_queue: Queue[str] = Queue()
    driver = LocalHumanInputDriver(
        driver_id="player_driver/human_local_input",
        family="human",
    )
    player = driver.bind(
        PlayerBindingSpec(
            seat="player_0",
            player_id="player_0",
            player_kind="human",
            driver_id="player_driver/human_local_input",
            driver_params={
                "action_queue": action_queue,
                "timeout_ms": 50,
                "timeout_fallback_move": "noop",
            },
        )
    )
    observation = ArenaObservation(
        board_text="realtime frame",
        legal_moves=("noop", "issue_command"),
        active_player="player_0",
    )

    current_time = {"value": 100.0}

    def fake_monotonic() -> float:
        return current_time["value"]

    monkeypatch.setattr(human_local_input_module.time, "monotonic", fake_monotonic)

    assert player.start_thinking(observation, deadline_ms=50) is True
    assert player.has_action() is False

    action_queue.put(
        dump_action_payload(
            build_action_payload(
                action="issue_command",
                player_id="player_0",
                sample_id="realtime-smoke",
            )
        )
    )

    current_time["value"] = 100.025
    assert player.has_action() is False

    current_time["value"] = 100.051

    assert player.has_action() is True
    action = player.pop_action()
    assert action.move == "issue_command"

    assert player.start_thinking(observation, deadline_ms=50) is True
    assert player.has_action() is False
    current_time["value"] = 100.102
    assert player.has_action() is True
    fallback_action = player.pop_action()
    assert fallback_action.move == "noop"
