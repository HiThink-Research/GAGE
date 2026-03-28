from __future__ import annotations

from collections.abc import Callable
from typing import Any

import pytest

from gage_eval.game_kits.contracts import GameVisualizationSpec
from gage_eval.game_kits.aec_env_game.pettingzoo.visualization import (
    VISUALIZATION_SPEC as PETTINGZOO_VISUALIZATION_SPEC,
)
from gage_eval.game_kits.board_game.gomoku.visualization import (
    VISUALIZATION_SPEC as GOMOKU_VISUALIZATION_SPEC,
)
from gage_eval.game_kits.board_game.tictactoe.visualization import (
    VISUALIZATION_SPEC as TICTACTOE_VISUALIZATION_SPEC,
)
from gage_eval.game_kits.phase_card_game.doudizhu.visualization import (
    VISUALIZATION_SPEC as DOUDIZHU_VISUALIZATION_SPEC,
)
from gage_eval.game_kits.phase_card_game.mahjong.visualization import (
    VISUALIZATION_SPEC as MAHJONG_VISUALIZATION_SPEC,
)
from gage_eval.game_kits.real_time_game.retro_platformer.visualization import (
    VISUALIZATION_SPEC as RETRO_VISUALIZATION_SPEC,
)
from gage_eval.game_kits.real_time_game.vizdoom.visualization import (
    VISUALIZATION_SPEC as VIZDOOM_VISUALIZATION_SPEC,
)
from gage_eval.role.arena.visualization.assembly import assemble_visual_scene
from gage_eval.role.arena.visualization.contracts import ObserverRef, TimelineEvent, VisualScene, VisualSession
from gage_eval.role.arena.visualization.recorder import ArenaVisualSessionRecorder


def _event(seq: int) -> TimelineEvent:
    return TimelineEvent(seq=seq, ts_ms=1000 + seq, type="snapshot", label="snapshot")


def _build_gomoku_scene() -> VisualScene:
    return assemble_visual_scene(
        visual_session=VisualSession(
            session_id="gomoku-acceptance",
            game_id="gomoku",
            plugin_id=GOMOKU_VISUALIZATION_SPEC.plugin_id,
            observer=ObserverRef(observer_id="Black", observer_kind="player"),
        ),
        event=_event(5),
        snapshot_body={
            "active_player_id": "White",
            "observer_player_id": "Black",
            "board_text": "   A B C\n 3 . . .\n 2 . W .\n 1 B . .",
            "legal_moves": ["B1", "C1", "A3"],
            "move_count": 2,
            "last_move": "B2",
            "player_ids": ["Black", "White"],
            "player_names": {"Black": "Black", "White": "White"},
            "coord_scheme": "A1",
            "winning_line": ["A1", "B2"],
        },
        visualization_spec=GOMOKU_VISUALIZATION_SPEC,
    )


def _build_tictactoe_scene() -> VisualScene:
    return assemble_visual_scene(
        visual_session=VisualSession(
            session_id="tictactoe-acceptance",
            game_id="tictactoe",
            plugin_id=TICTACTOE_VISUALIZATION_SPEC.plugin_id,
            observer=ObserverRef(observer_id="player_0", observer_kind="player"),
        ),
        event=_event(3),
        snapshot_body={
            "active_player_id": "player_1",
            "observer_player_id": "player_0",
            "board_text": "   1 2 3\n 3 . . .\n 2 . O .\n 1 X . .",
            "legal_moves": ["1,2", "1,3", "3,1"],
            "move_count": 2,
            "last_move": "2,2",
            "player_ids": ["player_0", "player_1"],
            "player_names": {"player_0": "Alpha", "player_1": "Beta"},
            "coord_scheme": "ROW_COL",
            "winning_line": ["1,1", "2,2", "3,3"],
        },
        visualization_spec=TICTACTOE_VISUALIZATION_SPEC,
    )


def _build_doudizhu_scene(observer: ObserverRef) -> VisualScene:
    return assemble_visual_scene(
        visual_session=VisualSession(
            session_id="doudizhu-acceptance",
            game_id="doudizhu",
            plugin_id=DOUDIZHU_VISUALIZATION_SPEC.plugin_id,
            observer=observer,
        ),
        event=_event(7),
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


def _build_mahjong_scene(observer: ObserverRef) -> VisualScene:
    return assemble_visual_scene(
        visual_session=VisualSession(
            session_id="mahjong-acceptance",
            game_id="mahjong",
            plugin_id=MAHJONG_VISUALIZATION_SPEC.plugin_id,
            observer=observer,
        ),
        event=_event(9),
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


def _build_pettingzoo_scene() -> VisualScene:
    return assemble_visual_scene(
        visual_session=VisualSession(
            session_id="pettingzoo-acceptance",
            game_id="pettingzoo",
            plugin_id=PETTINGZOO_VISUALIZATION_SPEC.plugin_id,
            observer=ObserverRef(observer_id="pilot_0", observer_kind="player"),
        ),
        event=_event(11),
        snapshot_body={
            "step": 3,
            "tick": 3,
            "move_count": 3,
            "stream_id": "arena",
            "last_move": "FIRE",
            "observation": {
                "board_text": "Space Invaders wave 3",
                "active_player": "pilot_0",
                "legal_moves": ["NOOP", "LEFT", "FIRE"],
                "metadata": {
                    "env_id": "pettingzoo.atari.space_invaders_v2",
                    "player_id": "pilot_0",
                    "reward": 1.5,
                },
                "view": {
                    "text": "Space Invaders wave 3",
                },
                "context": {
                    "mode": "turn",
                    "step": 3,
                },
            },
            "media": {
                "primary": {
                    "mediaId": "pz-frame-3",
                    "transport": "artifact_ref",
                    "mimeType": "image/png",
                    "url": "frames/pz-frame-3.png",
                }
            },
        },
        visualization_spec=PETTINGZOO_VISUALIZATION_SPEC,
    )


def _build_vizdoom_scene() -> VisualScene:
    return assemble_visual_scene(
        visual_session=VisualSession(
            session_id="vizdoom-acceptance",
            game_id="vizdoom",
            plugin_id=VIZDOOM_VISUALIZATION_SPEC.plugin_id,
            observer=ObserverRef(observer_id="p0", observer_kind="player"),
        ),
        event=_event(17),
        snapshot_body={
            "tick": 17,
            "step": 17,
            "stream_id": "pov",
            "actor": "p0",
            "observation": {
                "board_text": "Tick 17. Legal actions: 0, 1, 2",
                "active_player": "p0",
                "metadata": {
                    "reward": 0.75,
                },
                "view": {
                    "text": "Tick 17. Legal actions: 0, 1, 2",
                },
                "legal_actions": {
                    "items": ["0", "1", "2"],
                },
                "context": {
                    "mode": "tick",
                    "tick": 17,
                    "step": 17,
                },
            },
            "media": {
                "primary": {
                    "mediaId": "vizdoom-frame-17",
                    "transport": "artifact_ref",
                    "mimeType": "image/jpeg",
                    "url": "frames/vizdoom-frame-17.jpg",
                }
            },
            "viewport": {
                "width": 320,
                "height": 180,
            },
        },
        visualization_spec=VIZDOOM_VISUALIZATION_SPEC,
    )


def _build_retro_scene() -> VisualScene:
    return assemble_visual_scene(
        visual_session=VisualSession(
            session_id="retro-acceptance",
            game_id="retro_platformer",
            plugin_id=RETRO_VISUALIZATION_SPEC.plugin_id,
            observer=ObserverRef(observer_id="player_0", observer_kind="player"),
        ),
        event=_event(23),
        snapshot_body={
            "tick": 23,
            "step": 5,
            "move_count": 5,
            "last_move": "right_jump",
            "reward": 2.0,
            "observation": {
                "board_text": "Mario x03 · coins 17",
                "active_player": "player_0",
                "legal_moves": [
                    {"id": "noop", "label": "No-op", "text": "No-op"},
                    {"id": "right", "label": "Move Right", "text": "Move Right", "hold_ticks": 6},
                    {
                        "id": "right_jump",
                        "label": "Right + Jump",
                        "text": "Right + Jump",
                        "hold_ticks": 6,
                    },
                ],
                "metadata": {
                    "env_id": "SuperMarioBros3-Nes-v0",
                },
                "view": {
                    "text": "Mario x03 · coins 17",
                },
                "context": {
                    "mode": "tick",
                    "tick": 23,
                    "step": 5,
                },
            },
            "media": {
                "primary": {
                    "mediaId": "retro-frame-23",
                    "transport": "artifact_ref",
                    "mimeType": "image/png",
                    "url": "frames/retro-frame-23.png",
                }
            },
        },
        visualization_spec=RETRO_VISUALIZATION_SPEC,
    )


def _assert_non_blank_board(scene: VisualScene) -> None:
    board = scene.body["board"]
    cells = board["cells"]
    assert len(cells) >= 9
    assert any(cell["occupant"] for cell in cells)
    assert any(cell["isLegalAction"] for cell in cells)


def _assert_non_blank_table(scene: VisualScene) -> None:
    table = scene.body["table"]
    seats = table["seats"]
    assert len(seats) >= 3
    assert any(seat["hand"]["isVisible"] or seat["hand"]["maskedCount"] > 0 for seat in seats)
    assert table["center"]["label"]


def _assert_non_blank_frame(scene: VisualScene) -> None:
    assert scene.media is not None
    assert scene.media.primary is not None
    assert scene.media.primary.media_id
    assert scene.body["frame"]["title"]
    assert scene.body["view"]["text"] or scene.body["frame"]["viewport"] is not None


def _build_acceptance_session(
    *,
    game_id: str,
    visualization_spec: GameVisualizationSpec,
    scheduling_family: str,
) -> VisualSession:
    recorder = ArenaVisualSessionRecorder(
        plugin_id=visualization_spec.plugin_id,
        game_id=game_id,
        scheduling_family=scheduling_family,
        session_id=f"{game_id}-acceptance",
        observer_modes=tuple(visualization_spec.observer_schema.get("supported_modes", ())),
    )
    recorder.record_decision_window_open(
        ts_ms=1001,
        step=1,
        tick=1,
        player_id="human-0",
        observation={"gameId": game_id},
    )
    return recorder.build_visual_session()


_SCENE_CASES: tuple[
    tuple[
        str,
        GameVisualizationSpec,
        Callable[[], VisualScene],
        Callable[[VisualScene], None],
        tuple[str, ...],
        str,
    ],
    ...,
] = (
    (
        "gomoku",
        GOMOKU_VISUALIZATION_SPEC,
        _build_gomoku_scene,
        _assert_non_blank_board,
        ("player", "global"),
        "turn",
    ),
    (
        "tictactoe",
        TICTACTOE_VISUALIZATION_SPEC,
        _build_tictactoe_scene,
        _assert_non_blank_board,
        ("player", "global"),
        "turn",
    ),
    (
        "doudizhu",
        DOUDIZHU_VISUALIZATION_SPEC,
        lambda: _build_doudizhu_scene(ObserverRef(observer_id="player_0", observer_kind="player")),
        _assert_non_blank_table,
        ("player", "global"),
        "turn",
    ),
    (
        "mahjong",
        MAHJONG_VISUALIZATION_SPEC,
        lambda: _build_mahjong_scene(ObserverRef(observer_id="east", observer_kind="player")),
        _assert_non_blank_table,
        ("player", "global"),
        "turn",
    ),
    (
        "pettingzoo",
        PETTINGZOO_VISUALIZATION_SPEC,
        _build_pettingzoo_scene,
        _assert_non_blank_frame,
        ("player", "global"),
        "agent_cycle",
    ),
    (
        "vizdoom",
        VIZDOOM_VISUALIZATION_SPEC,
        _build_vizdoom_scene,
        _assert_non_blank_frame,
        ("player", "camera"),
        "real_time_tick",
    ),
    (
        "retro_platformer",
        RETRO_VISUALIZATION_SPEC,
        _build_retro_scene,
        _assert_non_blank_frame,
        ("player", "camera"),
        "real_time_tick",
    ),
)


@pytest.mark.parametrize(
    (
        "game_id",
        "visualization_spec",
        "scene_builder",
        "surface_assertion",
        "expected_observer_modes",
        "expected_scheduling_family",
    ),
    _SCENE_CASES,
)
def test_arena_visual_acceptance_matrix_exposes_render_playback_seek_observer_and_human_intent(
    game_id: str,
    visualization_spec: GameVisualizationSpec,
    scene_builder: Callable[[], VisualScene],
    surface_assertion: Callable[[VisualScene], None],
    expected_observer_modes: tuple[str, ...],
    expected_scheduling_family: str,
) -> None:
    scene = scene_builder()
    session = _build_acceptance_session(
        game_id=game_id,
        visualization_spec=visualization_spec,
        scheduling_family=expected_scheduling_family,
    )

    assert scene.game_id == game_id
    assert scene.plugin_id == visualization_spec.plugin_id
    assert scene.kind == visualization_spec.visual_kind
    assert scene.scene_id.endswith(f":seq:{scene.seq}")
    assert scene.legal_actions
    surface_assertion(scene)

    assert session.game_id == game_id
    assert session.plugin_id == visualization_spec.plugin_id
    assert session.playback.mode == "live_tail"
    assert session.playback.can_seek is True
    assert session.scheduling.family == expected_scheduling_family
    assert session.scheduling.phase == "waiting_for_intent"
    assert session.scheduling.accepts_human_intent is True
    assert session.capabilities == {
        "supportsReplay": True,
        "supportsTimeline": True,
        "supportsSeek": True,
        "observerModes": list(expected_observer_modes),
    }


@pytest.mark.parametrize(
    ("scene_builder", "player_id", "expected_visible_cards"),
    [
        (_build_doudizhu_scene, "player_0", ["3", "4"]),
        (_build_mahjong_scene, "east", ["B1", "Red"]),
    ],
)
def test_arena_visual_acceptance_table_observer_switch_reprojects_private_view(
    scene_builder: Callable[[ObserverRef], VisualScene],
    player_id: str,
    expected_visible_cards: list[str],
) -> None:
    spectator_scene = scene_builder(ObserverRef(observer_id="", observer_kind="spectator"))
    player_scene = scene_builder(ObserverRef(observer_id=player_id, observer_kind="player"))

    spectator_seats = {
        seat["playerId"]: seat
        for seat in spectator_scene.body["table"]["seats"]
    }
    player_seats = {
        seat["playerId"]: seat
        for seat in player_scene.body["table"]["seats"]
    }

    assert spectator_scene.body["status"]["observerPlayerId"] is None
    assert spectator_seats[player_id]["hand"]["isVisible"] is False
    assert spectator_seats[player_id]["hand"]["cards"] == []

    assert player_scene.body["status"]["observerPlayerId"] == player_id
    assert player_seats[player_id]["hand"]["isVisible"] is True
    assert player_seats[player_id]["hand"]["cards"] == expected_visible_cards
