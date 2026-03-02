from __future__ import annotations

from typing import Any, Sequence

from gage_eval.role.arena.games.doudizhu import env as doudizhu_env_module
from gage_eval.role.arena.games.doudizhu.env import DoudizhuArenaEnvironment


class _StubCore:
    num_players = 3

    def __init__(self) -> None:
        self._active_player = 0

    def reset(self) -> None:
        self._active_player = 0

    def get_active_player_id(self) -> int:
        return self._active_player

    def get_observation(self, player_index: int) -> dict[str, Any]:
        return {"player_index": player_index}

    def get_legal_actions(self, player_index: int) -> list[int]:
        _ = player_index
        return [1, 2]

    def get_perfect_information(self) -> dict[str, Any]:
        return {"hand_cards_with_suit": [["S3"], ["H4"], ["D5"]]}

    def encode_action(self, action_text: str) -> int:
        normalized = str(action_text).strip().lower()
        if normalized == "pass":
            return 1
        if normalized == "333":
            return 2
        raise ValueError(f"Unknown action: {action_text}")

    def step(self, action_id: int) -> None:
        _ = action_id
        self._active_player = (self._active_player + 1) % self.num_players

    def is_terminal(self) -> bool:
        return False

    def get_payoffs(self) -> list[float]:
        return [0.0, 0.0, 0.0]


class _StubFormatter:
    def __init__(self, *, player_id_map: dict[int, str]) -> None:
        self._player_id_map = dict(player_id_map)

    def format_observation(
        self,
        raw_obs: dict[str, Any],
        legal_action_ids: Sequence[int],
    ) -> tuple[dict[str, Any], dict[str, Any], list[str]]:
        public_state = {
            "landlord_id": self._player_id_map.get(0, "player_0"),
            "observer_index": raw_obs.get("player_index"),
        }
        private_state = {"hand": ["S3"]}
        legal_moves = [self.format_action(action_id) for action_id in legal_action_ids]
        return public_state, private_state, legal_moves

    @staticmethod
    def format_action(action_id: int) -> str:
        return "pass" if int(action_id) == 1 else "333"


def test_doudizhu_arena_exposes_get_last_frame(monkeypatch) -> None:
    monkeypatch.setattr(doudizhu_env_module, "make_core", lambda _game_type: _StubCore())
    monkeypatch.setattr(doudizhu_env_module, "DoudizhuFormatter", _StubFormatter)
    env = DoudizhuArenaEnvironment(
        player_ids=["player_0", "player_1", "player_2"],
        replay_live=False,
    )

    initial_frame = env.get_last_frame()
    assert isinstance(initial_frame, dict)
    assert initial_frame["active_player_id"] == "player_0"
    assert initial_frame["legal_moves"] == ["pass", "333"]
    assert initial_frame["ui_state"]["active_player_id"] == "player_0"

    env.observe("player_0")
    latest_frame = env.get_last_frame()
    assert latest_frame["observer_player_id"] == "player_0"
    assert "board_text" in latest_frame
