from __future__ import annotations

import json
from queue import Queue

from gage_eval.role.arena.core.players import PlayerBindingSpec
from gage_eval.role.arena.human_input_protocol import build_action_payload, dump_action_payload
from gage_eval.role.arena.player_drivers.human_local_input import LocalHumanInputDriver
from gage_eval.role.arena.types import ArenaObservation


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
        "hold_ticks": 6,
        "source": "arena_visual",
    }
