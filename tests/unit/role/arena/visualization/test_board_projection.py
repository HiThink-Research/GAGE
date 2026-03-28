from __future__ import annotations

from gage_eval.game_kits.board_game.gomoku.visualization import VISUALIZATION_SPEC as GOMOKU_VISUALIZATION_SPEC
from gage_eval.game_kits.board_game.tictactoe.visualization import (
    VISUALIZATION_SPEC as TICTACTOE_VISUALIZATION_SPEC,
)
from gage_eval.role.arena.visualization.assembly import assemble_visual_scene
from gage_eval.role.arena.visualization.contracts import ObserverRef, TimelineEvent, VisualSession


def test_gomoku_board_projection_structures_cells_and_legal_actions() -> None:
    scene = assemble_visual_scene(
        visual_session=VisualSession(
            session_id="gomoku-sample",
            game_id="gomoku",
            plugin_id=GOMOKU_VISUALIZATION_SPEC.plugin_id,
        ),
        event=TimelineEvent(
            seq=5,
            ts_ms=1005,
            type="snapshot",
            label="snapshot",
        ),
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

    assert scene.kind == "board"
    assert scene.summary["boardSize"] == 3
    assert scene.summary["coordScheme"] == "A1"
    assert "snapshot" not in scene.body
    assert "event" not in scene.body

    board = scene.body["board"]
    assert board["size"] == 3
    assert len(board["cells"]) == 9

    cells = {cell["coord"]: cell for cell in board["cells"]}
    assert cells["A1"]["occupant"] == "B"
    assert cells["A1"]["playerId"] == "Black"
    assert cells["A1"]["isWinningCell"] is True
    assert cells["B2"]["occupant"] == "W"
    assert cells["B2"]["playerName"] == "White"
    assert cells["B2"]["isLastMove"] is True
    assert cells["C1"]["isLegalAction"] is True

    assert scene.body["status"] == {
        "activePlayerId": "White",
        "observerPlayerId": "Black",
        "moveCount": 2,
        "lastMove": "B2",
        "winningLine": ["A1", "B2"],
    }
    assert scene.legal_actions == (
        {"id": "B1", "label": "B1", "coord": "B1", "row": 0, "col": 1},
        {"id": "C1", "label": "C1", "coord": "C1", "row": 0, "col": 2},
        {"id": "A3", "label": "A3", "coord": "A3", "row": 2, "col": 0},
    )


def test_tictactoe_board_projection_supports_row_col_coordinates() -> None:
    scene = assemble_visual_scene(
        visual_session=VisualSession(
            session_id="ttt-sample",
            game_id="tictactoe",
            plugin_id=TICTACTOE_VISUALIZATION_SPEC.plugin_id,
        ),
        event=TimelineEvent(
            seq=3,
            ts_ms=1003,
            type="snapshot",
            label="snapshot",
        ),
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

    assert scene.kind == "board"
    assert scene.summary["boardSize"] == 3
    assert scene.summary["coordScheme"] == "ROW_COL"

    board = scene.body["board"]
    cells = {cell["coord"]: cell for cell in board["cells"]}
    assert cells["1,1"]["occupant"] == "X"
    assert cells["1,1"]["playerId"] == "player_0"
    assert cells["2,2"]["occupant"] == "O"
    assert cells["2,2"]["isLastMove"] is True
    assert cells["3,3"]["isWinningCell"] is True

    assert scene.body["players"] == [
        {"playerId": "player_0", "playerName": "Alpha", "token": "X"},
        {"playerId": "player_1", "playerName": "Beta", "token": "O"},
    ]
    assert scene.legal_actions == (
        {"id": "1,2", "label": "1,2", "coord": "1,2", "row": 0, "col": 1},
        {"id": "1,3", "label": "1,3", "coord": "1,3", "row": 0, "col": 2},
        {"id": "3,1", "label": "3,1", "coord": "3,1", "row": 2, "col": 0},
    )


def test_board_projection_prefers_visual_session_observer_override() -> None:
    scene = assemble_visual_scene(
        visual_session=VisualSession(
            session_id="gomoku-observer-override",
            game_id="gomoku",
            plugin_id=GOMOKU_VISUALIZATION_SPEC.plugin_id,
            observer=ObserverRef(observer_id="White", observer_kind="player"),
        ),
        event=TimelineEvent(
            seq=6,
            ts_ms=1006,
            type="snapshot",
            label="snapshot",
        ),
        snapshot_body={
            "active_player_id": "White",
            "observer_player_id": "Black",
            "board_text": "   A B C\n 3 . . .\n 2 . W .\n 1 B . .",
            "legal_moves": ["B1"],
            "move_count": 2,
            "last_move": "B2",
            "player_ids": ["Black", "White"],
            "player_names": {"Black": "Black", "White": "White"},
            "coord_scheme": "A1",
        },
        visualization_spec=GOMOKU_VISUALIZATION_SPEC,
    )

    assert scene.body["status"]["observerPlayerId"] == "White"
