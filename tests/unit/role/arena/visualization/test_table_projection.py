from __future__ import annotations

from gage_eval.game_kits.phase_card_game.doudizhu.visualization import (
    VISUALIZATION_SPEC as DOUDIZHU_VISUALIZATION_SPEC,
)
from gage_eval.game_kits.phase_card_game.mahjong.visualization import (
    VISUALIZATION_SPEC as MAHJONG_VISUALIZATION_SPEC,
)
from gage_eval.role.arena.visualization.assembly import assemble_visual_scene
from gage_eval.role.arena.visualization.contracts import ObserverRef, TimelineEvent, VisualSession


def test_doudizhu_table_projection_masks_non_observer_private_hands() -> None:
    scene = assemble_visual_scene(
        visual_session=VisualSession(
            session_id="doudizhu-sample",
            game_id="doudizhu",
            plugin_id=DOUDIZHU_VISUALIZATION_SPEC.plugin_id,
            observer=ObserverRef(observer_id="player_0", observer_kind="player"),
        ),
        event=TimelineEvent(
            seq=7,
            ts_ms=1007,
            type="snapshot",
            label="snapshot",
        ),
        snapshot_body={
            "active_player_id": "player_0",
            "observer_player_id": "player_0",
            "player_ids": ["player_0", "player_1", "player_2"],
            "player_names": {
                "player_0": "Player 0",
                "player_1": "Player 1",
                "player_2": "Player 2",
            },
            "public_state": {
                "landlord_id": "player_0",
                "num_cards_left": {"player_0": 2, "player_1": 2, "player_2": 2},
                "played_cards": [
                    {"player_id": "player_0", "cards": ["3"]},
                    {"player_id": "player_1", "cards": []},
                    {"player_id": "player_2", "cards": []},
                ],
                "seen_cards": ["3"],
                "trace": [{"player": 0, "action": "3"}],
            },
            "private_state": {
                "self_id": "player_0",
                "current_hand": ["3", "4"],
                "current_hand_text": "3, 4",
            },
            "ui_state": {
                "roles": {
                    "player_0": "landlord",
                    "player_1": "peasant",
                    "player_2": "peasant",
                },
                "seat_order": {
                    "bottom": "player_0",
                    "left": "player_1",
                    "right": "player_2",
                },
                "latest_actions": [["3"], [], []],
                "move_history": [{"player_idx": 0, "move": "3"}],
            },
            "legal_moves": ["pass", "4"],
            "chat_log": [{"player_id": "player_1", "text": "watch this"}],
            "move_count": 1,
            "last_move": "3",
        },
        visualization_spec=DOUDIZHU_VISUALIZATION_SPEC,
    )

    assert scene.kind == "table"
    assert scene.summary["seatCount"] == 3
    assert "snapshot" not in scene.body
    assert "event" not in scene.body

    table = scene.body["table"]
    seats = {seat["playerId"]: seat for seat in table["seats"]}
    assert seats["player_0"]["seatId"] == "bottom"
    assert seats["player_0"]["role"] == "landlord"
    assert seats["player_0"]["hand"] == {
        "isVisible": True,
        "cards": ["3", "4"],
        "maskedCount": 0,
    }
    assert seats["player_1"]["hand"] == {
        "isVisible": False,
        "cards": [],
        "maskedCount": 2,
    }
    assert seats["player_2"]["playedCards"] == []
    assert table["center"] == {
        "label": "Seen cards",
        "cards": ["3"],
        "history": ["player_0: 3"],
    }
    assert scene.body["status"] == {
        "activePlayerId": "player_0",
        "observerPlayerId": "player_0",
        "privateViewPlayerId": "player_0",
        "moveCount": 1,
        "lastMove": "3",
        "landlordId": "player_0",
    }
    assert scene.legal_actions == (
        {"id": "pass", "label": "pass", "text": "pass"},
        {"id": "4", "label": "4", "text": "4"},
    )


def test_doudizhu_table_projection_keeps_spectator_status_distinct_from_private_view() -> None:
    scene = assemble_visual_scene(
        visual_session=VisualSession(
            session_id="doudizhu-spectator",
            game_id="doudizhu",
            plugin_id=DOUDIZHU_VISUALIZATION_SPEC.plugin_id,
            observer=ObserverRef(observer_id="", observer_kind="spectator"),
        ),
        event=TimelineEvent(
            seq=8,
            ts_ms=1008,
            type="snapshot",
            label="snapshot",
        ),
        snapshot_body={
            "active_player_id": "player_1",
            "observer_player_id": "player_0",
            "player_ids": ["player_0", "player_1", "player_2"],
            "player_names": {
                "player_0": "Player 0",
                "player_1": "Player 1",
                "player_2": "Player 2",
            },
            "public_state": {
                "landlord_id": "player_2",
                "num_cards_left": {"player_0": 3, "player_1": 4, "player_2": 5},
                "played_cards": [],
                "seen_cards": ["J", "Q", "K"],
            },
            "private_state": {
                "self_id": "player_0",
                "current_hand": ["3", "4", "5"],
            },
            "ui_state": {
                "roles": {
                    "player_0": "peasant",
                    "player_1": "peasant",
                    "player_2": "landlord",
                },
                "seat_order": {
                    "bottom": "player_0",
                    "left": "player_1",
                    "right": "player_2",
                },
            },
        },
        visualization_spec=DOUDIZHU_VISUALIZATION_SPEC,
    )

    seats = {seat["playerId"]: seat for seat in scene.body["table"]["seats"]}
    assert all(seat["isObserver"] is False for seat in seats.values())
    assert seats["player_0"]["hand"] == {
        "isVisible": False,
        "cards": [],
        "maskedCount": 3,
    }
    assert scene.body["status"]["observerPlayerId"] is None
    assert scene.body["status"]["privateViewPlayerId"] == "player_0"


def test_mahjong_table_projection_masks_private_hand_for_spectator() -> None:
    scene = assemble_visual_scene(
        visual_session=VisualSession(
            session_id="mahjong-sample",
            game_id="mahjong",
            plugin_id=MAHJONG_VISUALIZATION_SPEC.plugin_id,
            observer=ObserverRef(observer_id="", observer_kind="spectator"),
        ),
        event=TimelineEvent(
            seq=9,
            ts_ms=1009,
            type="snapshot",
            label="snapshot",
        ),
        snapshot_body={
            "active_player_id": "east",
            "observer_player_id": "east",
            "player_ids": ["east", "south", "west", "north"],
            "player_names": {
                "east": "East",
                "south": "South",
                "west": "West",
                "north": "North",
            },
            "public_state": {
                "discards": ["B1", "C1", "D1"],
                "melds": {"south": ["Pong C3"]},
            },
            "private_state": {
                "hand": ["B1", "Red"],
                "hand_raw": ["bamboo-1", "dragons-red"],
            },
            "legal_moves": ["B1", "Red"],
            "chat_log": [{"player_id": "south", "text": "pon"}],
            "move_count": 4,
            "last_move": "C1",
        },
        visualization_spec=MAHJONG_VISUALIZATION_SPEC,
    )

    assert scene.kind == "table"
    assert scene.summary["seatCount"] == 4

    table = scene.body["table"]
    seats = {seat["playerId"]: seat for seat in table["seats"]}
    assert seats["east"]["isActive"] is True
    assert all(seat["isObserver"] is False for seat in seats.values())
    assert seats["east"]["hand"] == {
        "isVisible": False,
        "cards": [],
        "maskedCount": 2,
    }
    assert seats["south"]["publicNotes"] == ["Pong C3"]
    assert table["center"] == {
        "label": "Discards",
        "cards": ["B1", "C1", "D1"],
        "history": [],
    }
    assert scene.body["status"] == {
        "activePlayerId": "east",
        "observerPlayerId": None,
        "privateViewPlayerId": "east",
        "moveCount": 4,
        "lastMove": "C1",
        "landlordId": None,
    }
    assert scene.legal_actions == (
        {"id": "B1", "label": "B1", "text": "B1"},
        {"id": "Red", "label": "Red", "text": "Red"},
    )


def test_mahjong_table_projection_keeps_private_hand_for_player_observer() -> None:
    scene = assemble_visual_scene(
        visual_session=VisualSession(
            session_id="mahjong-player-observer",
            game_id="mahjong",
            plugin_id=MAHJONG_VISUALIZATION_SPEC.plugin_id,
            observer=ObserverRef(observer_id="east", observer_kind="player"),
        ),
        event=TimelineEvent(
            seq=10,
            ts_ms=1010,
            type="snapshot",
            label="snapshot",
        ),
        snapshot_body={
            "active_player_id": "east",
            "observer_player_id": "east",
            "player_ids": ["east", "south", "west", "north"],
            "player_names": {
                "east": "East",
                "south": "South",
                "west": "West",
                "north": "North",
            },
            "public_state": {
                "discards": ["B1"],
                "melds": {},
            },
            "private_state": {
                "hand": ["B1", "Red"],
                "hand_raw": ["bamboo-1", "dragons-red"],
            },
            "legal_moves": ["B1"],
        },
        visualization_spec=MAHJONG_VISUALIZATION_SPEC,
    )

    seats = {seat["playerId"]: seat for seat in scene.body["table"]["seats"]}
    assert seats["east"]["hand"] == {
        "isVisible": True,
        "cards": ["B1", "Red"],
        "maskedCount": 0,
    }
    assert scene.body["status"]["privateViewPlayerId"] == "east"
