from __future__ import annotations

from pathlib import Path
from typing import Any, Optional, Sequence

from gage_eval.game_kits.phase_card_game.mahjong import (
    environment as mahjong_env_module,
)
from gage_eval.game_kits.phase_card_game.mahjong.environment import MahjongArena


REPO_ROOT = Path(__file__).resolve().parents[4]
_LEGACY_RENDERER_SUBSTRING = "".join(("show", "down"))
_LEGACY_VIEWER_NAME = "-".join(("rlcard", "".join(("show", "down"))))


def _read_text(relpath: str) -> str:
    return (REPO_ROOT / relpath).read_text(encoding="utf-8")


class _StubCore:
    num_players = 4

    def __init__(self) -> None:
        self._active_player = 0

    def reset(self) -> None:
        self._active_player = 0

    def get_all_hands(self) -> dict[int, list[str]]:
        return {0: ["bamboo-1"], 1: [], 2: [], 3: []}

    def get_active_player_id(self) -> int:
        return self._active_player

    def get_observation(self, player_index: int) -> dict[str, Any]:
        return {"player_index": player_index}

    def get_legal_actions(self, player_index: int) -> list[int]:
        _ = player_index
        return [0, 1]

    def step(self, action_id: int) -> None:
        _ = action_id

    def is_terminal(self) -> bool:
        return False

    def get_payoffs(self) -> list[float]:
        return [0.0, 0.0, 0.0, 0.0]


class _StubFormatter:
    def format_observation(
        self,
        raw_obs: dict[str, Any],
        legal_action_ids: Sequence[int],
    ) -> tuple[dict[str, Any], dict[str, Any], list[str]]:
        public_state = {"round": 1, "seat": raw_obs.get("player_index")}
        private_state = {"hand": ["B1"]}
        legal_moves = [self.format_action(action_id) for action_id in legal_action_ids]
        return public_state, private_state, legal_moves

    @staticmethod
    def format_action(action_id: int) -> str:
        return "B1" if int(action_id) == 0 else "Stand"


class _StubParser:
    @staticmethod
    def parse(
        payload: str | dict[str, Any],
        *,
        legal_action_ids: Optional[Sequence[int]] = None,
        legal_moves: Optional[Sequence[str]] = None,
        **kwargs: Any,
    ) -> Any:
        _ = (payload, legal_action_ids, legal_moves, kwargs)
        raise AssertionError("parse should not be called in frame export test")


class _StubRenderer:
    @staticmethod
    def render_frame(frame: dict[str, Any]) -> dict[str, Any]:
        rendered = dict(frame)
        rendered["renderer"] = "stub"
        return rendered


def test_mahjong_arena_exposes_get_last_frame(monkeypatch) -> None:
    monkeypatch.setattr(
        mahjong_env_module,
        "build_action_maps",
        lambda: ({0: "B1", 1: "Stand"}, {"b1": 0, "stand": 1}, {0: "bamboo-1", 1: "stand"}),
    )
    arena = MahjongArena(
        core=_StubCore(),
        formatter=_StubFormatter(),
        parser=_StubParser(),
        renderer=_StubRenderer(),
        player_ids=["player_0", "player_1", "player_2", "player_3"],
    )

    initial_frame = arena.get_last_frame()
    assert isinstance(initial_frame, dict)
    assert initial_frame["active_player_id"] == "player_0"
    assert initial_frame["legal_moves"] == ["B1", "Stand"]
    assert initial_frame["renderer"] == "stub"

    arena.observe("player_0")
    latest_frame = arena.get_last_frame()
    assert latest_frame["observer_player_id"] == "player_0"
    assert latest_frame["public_state"]["round"] == 1


def test_mahjong_run_scripts_do_not_default_to_legacy_viewer() -> None:
    run_sh = _read_text("scripts/run/arenas/mahjong/run.sh")
    human_vs_ai = _read_text("scripts/run/arenas/mahjong/run_human_vs_ai_legacy.sh")
    human_vs_dummy = _read_text("scripts/run/arenas/mahjong/run_human_vs_dummy_legacy.sh")
    real_ai = _read_text("scripts/run/arenas/mahjong/run_real_ai_legacy.sh")

    assert 'MODE="${MODE:-human-vs-ai}"' in run_sh
    assert _LEGACY_RENDERER_SUBSTRING not in run_sh
    for script in (human_vs_ai, human_vs_dummy, real_ai):
        assert _LEGACY_VIEWER_NAME not in script
        assert "frontend/arena-visual" in script
        assert "VITE_ARENA_GATEWAY_BASE_URL" in script
        assert "/sessions/" in script
