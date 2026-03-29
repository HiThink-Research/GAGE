from __future__ import annotations

from gage_eval.role.adapters.arena import ArenaRoleAdapter
from gage_eval.role.adapters.base import RoleAdapterState


def test_gomoku_arena_scripted_vertical_win() -> None:
    arena = ArenaRoleAdapter(
        adapter_id="arena",
        game_kit="gomoku",
        env="gomoku_standard",
        runtime_overrides={
            "board_size": 3,
            "win_len": 3,
            "coord_scheme": "A1",
            "rule_profile": "freestyle",
            "win_directions": ["vertical"],
        },
        visualizer={"enabled": False},
        players=[
            {"seat": "black", "player_id": "Black", "player_kind": "dummy", "actions": ["A1", "A2", "A3"]},
            {"seat": "white", "player_id": "White", "player_kind": "dummy", "actions": ["B1", "B2"]},
        ],
    )
    sample = {
        "id": "gomoku_001",
        "game_kit": "gomoku",
        "env": "gomoku_standard",
        "messages": [],
        "runtime_overrides": {
            "board_size": 3,
            "win_len": 3,
            "coord_scheme": "A1",
            "rule_profile": "freestyle",
            "win_directions": ["vertical"],
        },
    }

    output = arena.invoke({"sample": sample}, RoleAdapterState())

    result = output["result"]

    assert result["winner"] == "Black"
    assert result["result"] == "win"
    assert result["win_direction"] == "vertical"
    assert result["line_length"] == 3
    assert result["rule_profile"] == "freestyle"


def test_gomoku_arena_scripted_row_col_win() -> None:
    arena = ArenaRoleAdapter(
        adapter_id="arena",
        game_kit="gomoku",
        env="gomoku_standard",
        runtime_overrides={
            "board_size": 3,
            "win_len": 3,
            "coord_scheme": "ROW_COL",
        },
        visualizer={"enabled": False},
        players=[
            {"seat": "black", "player_id": "Human", "player_kind": "dummy", "actions": ["1,1", "1,2", "1,3"]},
            {"seat": "white", "player_id": "Bot", "player_kind": "dummy", "actions": ["2,1", "2,2"]},
        ],
    )
    sample = {
        "id": "gomoku_002",
        "game_kit": "gomoku",
        "env": "gomoku_standard",
        "messages": [],
        "runtime_overrides": {
            "board_size": 3,
            "win_len": 3,
            "coord_scheme": "ROW_COL",
        },
    }

    output = arena.invoke({"sample": sample}, RoleAdapterState())

    result = output["result"]

    assert result["winner"] == "Human"
    assert result["result"] == "win"
    assert result["rule_profile"] == "freestyle"
