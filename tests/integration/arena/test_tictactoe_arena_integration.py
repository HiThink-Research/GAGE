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


def test_tictactoe_arena_backend_vs_backend_diagonal_win() -> None:
    rm = _role_manager()
    rm.register_role_adapter("x_adapter", _StaticMoveAdapter("x_adapter", ["1,1", "2,2", "3,3"]))
    rm.register_role_adapter("o_adapter", _StaticMoveAdapter("o_adapter", ["1,2", "1,3"]))

    arena = ArenaRoleAdapter(
        adapter_id="arena",
        environment={"impl": "tictactoe_v1", "board_size": 3, "coord_scheme": "ROW_COL"},
        rules={"win_len": 3},
        scheduler={"type": "turn", "max_turns": 9},
        parser={"impl": "grid_parser_v1", "coord_scheme": "ROW_COL", "board_size": 3},
        visualizer={"enabled": False},
        players=[
            {"name": "X", "type": "backend", "ref": "x_adapter"},
            {"name": "O", "type": "backend", "ref": "o_adapter"},
        ],
    )
    sample = {
        "id": "tictactoe_001",
        "messages": [],
        "metadata": {
            "board_size": 3,
            "win_len": 3,
            "coord_scheme": "ROW_COL",
            "player_ids": ["X", "O"],
            "player_names": {"X": "X", "O": "O"},
            "start_player_id": "X",
        },
        "eval_config": {"max_turns": 9, "retry_illegal": 0},
    }

    output = arena.invoke({"sample": sample, "role_manager": rm}, RoleAdapterState())

    assert output["winner"] == "X"
    assert output["result"] == "win"


def test_tictactoe_arena_human_vs_backend_row_win() -> None:
    rm = _role_manager()
    rm.register_role_adapter(
        "human_adapter",
        _StaticMoveAdapter("human_adapter", ["1,1", "1,2", "1,3"], "human"),
    )
    rm.register_role_adapter("bot_adapter", _StaticMoveAdapter("bot_adapter", ["2,1", "2,2"]))

    arena = ArenaRoleAdapter(
        adapter_id="arena",
        environment={"impl": "tictactoe_v1", "board_size": 3, "coord_scheme": "ROW_COL"},
        rules={"win_len": 3},
        scheduler={"type": "turn", "max_turns": 9},
        parser={"impl": "grid_parser_v1", "coord_scheme": "ROW_COL", "board_size": 3},
        visualizer={"enabled": False},
        players=[
            {"name": "Human", "type": "human", "ref": "human_adapter"},
            {"name": "Bot", "type": "backend", "ref": "bot_adapter"},
        ],
    )
    sample = {
        "id": "tictactoe_002",
        "messages": [],
        "metadata": {
            "board_size": 3,
            "win_len": 3,
            "coord_scheme": "ROW_COL",
            "player_ids": ["Human", "Bot"],
            "player_names": {"Human": "Human", "Bot": "Bot"},
            "start_player_id": "Human",
        },
        "eval_config": {"max_turns": 9, "retry_illegal": 0},
    }

    output = arena.invoke({"sample": sample, "role_manager": rm}, RoleAdapterState())

    assert output["winner"] == "Human"
    assert output["result"] == "win"
