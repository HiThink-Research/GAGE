from __future__ import annotations

import json

from gage_eval.role.arena.games.mahjong.mahjong_input_mapper import MahjongInputMapper
from gage_eval.role.arena.games.retro.retro_input_mapper import RetroInputMapper
from gage_eval.role.arena.input_mapping import BrowserKeyEvent, GameInputMapper, HumanActionEvent


class _EchoMapper(GameInputMapper):
    def _map_event_to_actions(self, *, event: BrowserKeyEvent, context):
        if event.key is None:
            return []
        return [
            HumanActionEvent(
                player_id=str(context.get("human_player_id") or "player_x"),
                move=f"KEY:{event.key}",
                raw=event.key,
                metadata={"event_type": event.event_type},
            )
        ]


def test_input_mapper_normalizes_context_and_defaults() -> None:
    mapper = _EchoMapper()
    actions = mapper.handle_browser_event(
        {"type": "keydown", "key": "x"},
        context={"human_player_id": "human_7"},
    )
    assert len(actions) == 1
    assert actions[0].player_id == "human_7"
    assert actions[0].move == "KEY:x"
    assert actions[0].to_queue_payload() == "x"


def test_input_mapper_ignores_invalid_payload() -> None:
    mapper = _EchoMapper()
    assert mapper.handle_browser_event({"foo": "bar"}, context={"human_player_id": "h"}) == []


def test_retro_input_mapper_emits_action_and_dedups() -> None:
    mapper = RetroInputMapper(default_hold_ticks=3)
    context = {"human_player_id": "p0"}

    first = mapper.handle_browser_event({"type": "keydown", "key": "w"}, context=context)
    assert len(first) == 1
    assert first[0].player_id == "p0"
    assert first[0].move == "UP"
    assert json.loads(first[0].raw)["hold_ticks"] == 3

    second = mapper.handle_browser_event({"type": "keydown", "key": "w"}, context=context)
    assert second == []

    mapper.handle_browser_event({"type": "keyup", "key": "w"}, context=context)
    third = mapper.handle_browser_event(
        {"type": "keydown", "key": "w", "hold_ticks": 6},
        context=context,
    )
    assert len(third) == 1
    assert json.loads(third[0].raw)["hold_ticks"] == 6


def test_mahjong_input_mapper_emits_action_with_chat_payload() -> None:
    mapper = MahjongInputMapper()
    actions = mapper.handle_browser_event(
        {
            "event": "action_submit",
            "action": "B1",
            "chat": "draw luck",
        },
        context={
            "human_player_id": "p1",
            "legal_moves": ["B1", "Stand"],
        },
    )

    assert len(actions) == 1
    assert actions[0].player_id == "p1"
    assert actions[0].move == "B1"
    payload = json.loads(actions[0].raw)
    assert payload["action"] == "B1"
    assert payload["chat"] == "draw luck"


def test_mahjong_input_mapper_resolves_index_and_filters_illegal_action() -> None:
    mapper = MahjongInputMapper()
    context = {"human_player_id": "p0", "legal_moves": ["B1", "Stand", "Pong"]}

    by_index = mapper.handle_browser_event(
        {"event": "action_submit", "action_index": 2},
        context=context,
    )
    assert len(by_index) == 1
    assert by_index[0].move == "Stand"

    illegal = mapper.handle_browser_event(
        {"event": "action_submit", "action": "C9"},
        context=context,
    )
    assert illegal == []
