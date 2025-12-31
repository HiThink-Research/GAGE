from __future__ import annotations

from gage_eval.role.adapters.arena import ArenaRoleAdapter
from gage_eval.role.adapters.base import RoleAdapter, RoleAdapterState
from gage_eval.role.resource_profile import NodeResource, ResourceProfile
from gage_eval.role.role_manager import RoleManager


class _StaticMoveAdapter(RoleAdapter):
    def __init__(self, adapter_id: str, moves: list[str], role_type: str = "dut_model") -> None:
        super().__init__(adapter_id=adapter_id, role_type=role_type, capabilities=("text",))
        self._moves = list(moves)
        self._index = 0

    async def ainvoke(self, payload: dict, state: RoleAdapterState) -> dict:
        if self._moves:
            if self._index < len(self._moves):
                move = self._moves[self._index]
            else:
                move = self._moves[-1]
        else:
            move = ""
        self._index += 1
        return {"answer": move}


def _role_manager() -> RoleManager:
    return RoleManager(ResourceProfile([NodeResource(node_id="local", gpus=0, cpus=2)]))


def test_gomoku_arena_backend_vs_backend_vertical_win() -> None:
    rm = _role_manager()
    rm.register_role_adapter("p0_adapter", _StaticMoveAdapter("p0_adapter", ["A1", "A2", "A3"]))
    rm.register_role_adapter("p1_adapter", _StaticMoveAdapter("p1_adapter", ["B1", "B2"]))

    arena = ArenaRoleAdapter(
        adapter_id="arena",
        environment={"impl": "gomoku_local_v1", "board_size": 3, "coord_scheme": "A1"},
        rules={"win_len": 3, "rule_profile": "freestyle", "win_directions": ["vertical"]},
        scheduler={"type": "turn", "max_turns": 9},
        parser={"impl": "grid_parser_v1", "coord_scheme": "A1"},
        visualizer={"enabled": False},
        players=[
            {"name": "p0", "type": "backend", "ref": "p0_adapter"},
            {"name": "p1", "type": "backend", "ref": "p1_adapter"},
        ],
    )
    sample = {
        "id": "gomoku_001",
        "messages": [],
        "metadata": {
            "board_size": 3,
            "win_len": 3,
            "coord_scheme": "A1",
            "rule_profile": "freestyle",
            "win_directions": ["vertical"],
            "player_ids": ["p0", "p1"],
            "player_names": {"p0": "Black", "p1": "White"},
            "start_player_id": "p0",
        },
        "eval_config": {"max_turns": 9, "retry_illegal": 0},
    }

    output = arena.invoke({"sample": sample, "role_manager": rm}, RoleAdapterState())

    assert output["winner"] == "p0"
    assert output["result"] == "win"
    assert output["win_direction"] == "vertical"
    assert output["line_length"] == 3
    assert output["rule_profile"] == "freestyle"


def test_gomoku_arena_human_vs_backend_row_col() -> None:
    rm = _role_manager()
    rm.register_role_adapter("human_adapter", _StaticMoveAdapter("human_adapter", ["1,1", "1,2", "1,3"], "human"))
    rm.register_role_adapter("bot_adapter", _StaticMoveAdapter("bot_adapter", ["2,1", "2,2"]))

    arena = ArenaRoleAdapter(
        adapter_id="arena",
        environment={"impl": "gomoku_local_v1", "board_size": 3, "coord_scheme": "ROW_COL"},
        rules={"win_len": 3},
        scheduler={"type": "turn", "max_turns": 9},
        parser={"impl": "grid_parser_v1", "coord_scheme": "ROW_COL"},
        visualizer={"enabled": False},
        players=[
            {"name": "human", "type": "human", "ref": "human_adapter"},
            {"name": "bot", "type": "backend", "ref": "bot_adapter"},
        ],
    )
    sample = {
        "id": "gomoku_002",
        "messages": [],
        "metadata": {
            "board_size": 3,
            "win_len": 3,
            "coord_scheme": "ROW_COL",
            "player_ids": ["human", "bot"],
            "player_names": {"human": "Human", "bot": "Bot"},
            "start_player_id": "human",
        },
        "eval_config": {"max_turns": 9, "retry_illegal": 0},
    }

    output = arena.invoke({"sample": sample, "role_manager": rm}, RoleAdapterState())

    assert output["winner"] == "human"
    assert output["result"] == "win"
    assert output["rule_profile"] == "freestyle"
