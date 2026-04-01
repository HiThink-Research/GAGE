from __future__ import annotations

import json
from queue import Empty

import pytest

from gage_eval.role.arena.human_input_protocol import (
    ContinuousStateMailbox,
    LatestActionMailbox,
    SampleActionRouter,
    build_action_payload,
    dump_action_payload,
)


def test_sample_action_router_isolates_same_player_id_across_samples() -> None:
    router_a = SampleActionRouter(sample_id="sample_a", player_ids=["Human"])
    router_b = SampleActionRouter(sample_id="sample_b", player_ids=["Human"])

    router_a.put(
        dump_action_payload(
            build_action_payload(
                action="A1",
                player_id="Human",
                sample_id="sample_a",
            )
        )
    )
    router_b.put(
        dump_action_payload(
            build_action_payload(
                action="B2",
                player_id="Human",
                sample_id="sample_b",
            )
        )
    )

    payload_a = json.loads(router_a.queue_for("Human").get_nowait())
    payload_b = json.loads(router_b.queue_for("Human").get_nowait())

    assert payload_a["sample_id"] == "sample_a"
    assert payload_a["action"] == "A1"
    assert payload_b["sample_id"] == "sample_b"
    assert payload_b["action"] == "B2"


def test_sample_action_router_drops_payload_from_other_sample() -> None:
    router = SampleActionRouter(sample_id="sample_a", player_ids=["Human"])

    router.put(
        dump_action_payload(
            build_action_payload(
                action="B2",
                player_id="Human",
                sample_id="sample_b",
            )
        )
    )

    try:
        router.queue_for("Human").get_nowait()
    except Empty:
        return
    raise AssertionError("Expected mismatched sample payload to be dropped")


def test_latest_action_mailbox_keeps_only_most_recent_unread_payload() -> None:
    mailbox = LatestActionMailbox()

    mailbox.put("first")
    mailbox.put("second")

    assert mailbox.get_nowait() == "second"
    with pytest.raises(Empty):
        mailbox.get_nowait()


def test_latest_action_mailbox_ignores_stale_payloads_when_input_seq_regresses() -> None:
    mailbox = LatestActionMailbox()

    mailbox.put(
        dump_action_payload(
            build_action_payload(
                action="right",
                player_id="player_0",
                sample_id="retro-smoke",
                metadata={"input_seq": 10},
            )
        )
    )
    mailbox.put(
        dump_action_payload(
            build_action_payload(
                action="noop",
                player_id="player_0",
                sample_id="retro-smoke",
                metadata={"input_seq": 9},
            )
        )
    )

    payload = json.loads(mailbox.get_nowait())
    assert payload["action"] == "right"
    assert payload["metadata"]["input_seq"] == 10


def test_sample_action_router_can_route_stateful_players_to_latest_mailboxes() -> None:
    router = SampleActionRouter(
        sample_id="sample_a",
        player_ids=["Human", "Bot"],
        realtime_input_semantics_by_player={"Human": "continuous_state"},
    )

    router.put(
        dump_action_payload(
            build_action_payload(
                action="right",
                player_id="Human",
                sample_id="sample_a",
            )
        )
    )
    router.put(
        dump_action_payload(
            build_action_payload(
                action="right_jump",
                player_id="Human",
                sample_id="sample_a",
            )
        )
    )

    human_route = router.queue_for("Human")
    assert isinstance(human_route, LatestActionMailbox)
    assert json.loads(human_route.get_nowait())["action"] == "right_jump"
    with pytest.raises(Empty):
        human_route.get_nowait()


def test_sample_action_router_defaults_to_fifo_for_queued_command_players() -> None:
    router = SampleActionRouter(
        sample_id="sample_a",
        player_ids=["Commander"],
        realtime_input_semantics_by_player={"Commander": "queued_command"},
    )

    router.put(
        dump_action_payload(
            build_action_payload(
                action="build_barracks",
                player_id="Commander",
                sample_id="sample_a",
            )
        )
    )
    router.put(
        dump_action_payload(
            build_action_payload(
                action="train_engineer",
                player_id="Commander",
                sample_id="sample_a",
            )
        )
    )

    queued_route = router.queue_for("Commander")
    assert not isinstance(queued_route, ContinuousStateMailbox)
    assert json.loads(queued_route.get_nowait())["action"] == "build_barracks"
    assert json.loads(queued_route.get_nowait())["action"] == "train_engineer"


def test_sample_action_router_keeps_latest_state_for_queued_command_players() -> None:
    router = SampleActionRouter(
        sample_id="sample_a",
        player_ids=["Commander"],
        realtime_input_semantics_by_player={"Commander": "queued_command"},
    )

    router.put(
        dump_action_payload(
            build_action_payload(
                action="build_barracks",
                player_id="Commander",
                sample_id="sample_a",
                metadata={"input_seq": 1},
            )
        )
    )
    router.put(
        dump_action_payload(
            build_action_payload(
                action="train_engineer",
                player_id="Commander",
                sample_id="sample_a",
                metadata={"input_seq": 2},
            )
        )
    )

    queued_route = router.queue_for("Commander")
    latest_route = router.latest_for("Commander")

    assert json.loads(queued_route.get_nowait())["action"] == "build_barracks"
    assert json.loads(queued_route.get_nowait())["action"] == "train_engineer"
    assert json.loads(latest_route.get_nowait())["action"] == "train_engineer"
