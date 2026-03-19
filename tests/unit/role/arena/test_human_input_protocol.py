from __future__ import annotations

import json
from queue import Empty

from gage_eval.role.arena.human_input_protocol import (
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
