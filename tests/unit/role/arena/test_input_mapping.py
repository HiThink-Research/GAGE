from __future__ import annotations

import json

from gage_eval.role.arena.games.common.grid_coord_input_mapper import GridCoordInputMapper
from gage_eval.role.arena.games.doudizhu.doudizhu_input_mapper import DoudizhuInputMapper
from gage_eval.role.arena.games.mahjong.mahjong_input_mapper import MahjongInputMapper
from gage_eval.role.arena.games.pettingzoo.pettingzoo_input_mapper import PettingZooDiscreteInputMapper
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
    assert first[0].move == "up"
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


def test_doudizhu_input_mapper_maps_pass_alias_and_chat() -> None:
    mapper = DoudizhuInputMapper()
    actions = mapper.handle_browser_event(
        {
            "event": "action_submit",
            "move": "skip",
            "chat": "let it go",
        },
        context={
            "human_player_id": "p2",
            "legal_moves": ["pass", "333"],
        },
    )

    assert len(actions) == 1
    assert actions[0].player_id == "p2"
    assert actions[0].move == "pass"
    payload = json.loads(actions[0].raw)
    assert payload["action"] == "pass"
    assert payload["chat"] == "let it go"


def test_doudizhu_input_mapper_supports_index_and_filters_illegal() -> None:
    mapper = DoudizhuInputMapper()
    context = {"human_player_id": "p0", "legal_moves": ["pass", "333", "4444"]}

    by_index = mapper.handle_browser_event(
        {"event": "action_submit", "action_index": 3},
        context=context,
    )
    assert len(by_index) == 1
    assert by_index[0].move == "4444"

    illegal = mapper.handle_browser_event(
        {"event": "action_submit", "action": "AAAA"},
        context=context,
    )
    assert illegal == []


def test_grid_coord_input_mapper_maps_coordinate_and_index() -> None:
    mapper = GridCoordInputMapper(coord_scheme="A1")
    context = {"human_player_id": "p0", "legal_moves": ["A1", "B2", "C3"]}

    by_text = mapper.handle_browser_event(
        {"event": "action_submit", "move": "b2"},
        context=context,
    )
    assert len(by_text) == 1
    assert by_text[0].move == "B2"
    assert by_text[0].raw == "B2"

    by_index = mapper.handle_browser_event(
        {"event": "action_submit", "action_index": 3},
        context=context,
    )
    assert len(by_index) == 1
    assert by_index[0].move == "C3"


def test_grid_coord_input_mapper_supports_row_col_and_filters_illegal() -> None:
    mapper = GridCoordInputMapper(coord_scheme="ROW_COL")
    context = {"human_player_id": "p1", "legal_moves": ["1,1", "2,2"]}

    valid = mapper.handle_browser_event(
        {"event": "action_submit", "coord": "2 2"},
        context=context,
    )
    assert len(valid) == 1
    assert valid[0].move == "2,2"

    illegal = mapper.handle_browser_event(
        {"event": "action_submit", "coord": "3,3"},
        context=context,
    )
    assert illegal == []


def test_pettingzoo_discrete_input_mapper_maps_action_and_index() -> None:
    mapper = PettingZooDiscreteInputMapper()
    context = {"human_player_id": "p0", "legal_moves": ["NOOP", "FIRE", "RIGHT"]}

    by_label = mapper.handle_browser_event(
        {"event": "action_submit", "action": "fire"},
        context=context,
    )
    assert len(by_label) == 1
    assert by_label[0].move == "FIRE"
    assert by_label[0].raw == "FIRE"

    by_index = mapper.handle_browser_event(
        {"event": "action_submit", "action_index": 3},
        context=context,
    )
    assert len(by_index) == 1
    assert by_index[0].move == "RIGHT"


def test_pettingzoo_discrete_input_mapper_filters_illegal() -> None:
    mapper = PettingZooDiscreteInputMapper()
    context = {"human_player_id": "p1", "legal_moves": ["0", "1", "2"]}

    illegal = mapper.handle_browser_event(
        {"event": "action_submit", "action": "99"},
        context=context,
    )
    assert illegal == []
