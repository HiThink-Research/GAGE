from __future__ import annotations

from queue import Queue

from gage_eval.role.adapters.arena import ArenaRoleAdapter
from gage_eval.role.adapters.base import RoleAdapter, RoleAdapterState
from gage_eval.role.resource_profile import NodeResource, ResourceProfile
from gage_eval.role.role_manager import RoleManager


class _StaticMoveAdapter(RoleAdapter):
    def __init__(self, adapter_id: str, moves: list[str], role_type: str = "dut_model") -> None:
        super().__init__(adapter_id=adapter_id, role_type=role_type, capabilities=("text",))
        self._moves = list(moves)
        self._index = 0

    def invoke(self, payload: dict, state: RoleAdapterState | None = None) -> dict:
        del payload
        return super().invoke({}, state or RoleAdapterState())

    async def ainvoke(self, payload: dict, state: RoleAdapterState) -> dict:
        del payload, state
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
    rm.register_backend("x_adapter", _StaticMoveAdapter("x_adapter", ["1,1", "2,2", "3,3"]))
    rm.register_backend("o_adapter", _StaticMoveAdapter("o_adapter", ["1,2", "1,3"]))

    arena = ArenaRoleAdapter(
        adapter_id="arena",
        game_kit="tictactoe",
        env="tictactoe_standard",
        runtime_overrides={"coord_scheme": "ROW_COL", "scheduler": "turn/default"},
        visualizer={"enabled": False},
        players=[
            {"seat": "x", "player_id": "X", "player_kind": "llm", "backend_id": "x_adapter"},
            {"seat": "o", "player_id": "O", "player_kind": "llm", "backend_id": "o_adapter"},
        ],
    )
    sample = {
        "id": "tictactoe_001",
        "game_kit": "tictactoe",
        "env": "tictactoe_standard",
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
    result = output["result"]

    assert result["winner"] == "X"
    assert result["result"] == "win"


def test_tictactoe_arena_human_vs_backend_row_win() -> None:
    rm = _role_manager()
    rm.register_backend("bot_adapter", _StaticMoveAdapter("bot_adapter", ["2,1", "2,2"]))
    human_actions: Queue[str] = Queue()
    for move in ("1,1", "1,2", "1,3"):
        human_actions.put(move)

    arena = ArenaRoleAdapter(
        adapter_id="arena",
        game_kit="tictactoe",
        env="tictactoe_standard",
        runtime_overrides={"coord_scheme": "ROW_COL", "scheduler": "turn/default"},
        visualizer={"enabled": False},
        players=[
            {
                "seat": "x",
                "player_id": "Human",
                "player_kind": "human",
                "driver_params": {"action_queue": human_actions},
            },
            {"seat": "o", "player_id": "Bot", "player_kind": "llm", "backend_id": "bot_adapter"},
        ],
    )
    sample = {
        "id": "tictactoe_002",
        "game_kit": "tictactoe",
        "env": "tictactoe_standard",
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
    result = output["result"]

    assert result["winner"] == "Human"
    assert result["result"] == "win"
