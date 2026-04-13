from __future__ import annotations

import importlib

import pytest

from gage_eval.game_kits.aec_env_game.pettingzoo import replay as pettingzoo_replay


_LEGACY_TOOL_MODULE = ".".join(
    ("gage_eval", "tools", "_".join(("ws", "rgb", "replay")))
)
_LEGACY_REPLAY_FACTORY = "_".join(("build", "ws", "rgb", "replay", "display"))


class _StubPettingZooReplayEnvironment:
    def __init__(self, **_: object) -> None:
        self._frames = [
            {"board_text": "initial", "metadata": {"step": 0}},
            {"board_text": "observed-1", "metadata": {"step": 1}},
            {"board_text": "applied-1", "metadata": {"step": 2}},
        ]
        self._index = 0

    def get_last_frame(self) -> dict[str, object]:
        return dict(self._frames[self._index])

    def get_active_player(self) -> str:
        return "player_0"

    def observe(self, player: str) -> None:
        _ = player
        self._index = min(1, len(self._frames) - 1)

    def apply(self, action) -> None:
        _ = action
        self._index = min(2, len(self._frames) - 1)
        return None


def test_legacy_replay_tool_public_path_is_removed() -> None:
    with pytest.raises(ModuleNotFoundError):
        importlib.import_module(_LEGACY_TOOL_MODULE)


def test_pettingzoo_replay_builds_visualization_artifact_without_legacy_public_api(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        pettingzoo_replay,
        "PettingZooAecArenaEnvironment",
        _StubPettingZooReplayEnvironment,
    )
    monkeypatch.setattr(
        pettingzoo_replay,
        "_can_import_env",
        lambda env_id: env_id == "pettingzoo.atari.space_invaders_v2",
    )

    artifact = pettingzoo_replay.build_replay_artifact(
        {
            "sample": {
                "id": "sample_1",
                "metadata": {"player_ids": ["player_0"]},
                "predict_result": [
                    {
                        "result": {
                            "move_log": [
                                {"index": 1, "player": "player_0", "move": "RIGHT"},
                            ]
                        }
                    }
                ],
            }
        },
        task_id="pettingzoo_space_invaders_dummy",
        fps=10.0,
        max_frames=0,
    )

    assert not hasattr(pettingzoo_replay, _LEGACY_REPLAY_FACTORY)
    assert not hasattr(pettingzoo_replay, "ReplayFrameCursor")
    assert artifact["artifact_type"] == "pettingzoo_replay_frames"
    assert artifact["artifact_id"] == "replay:sample_1:pettingzoo"
    assert artifact["label"] == "pettingzoo_replay:pettingzoo.atari.space_invaders_v2"
    assert artifact["human_player_id"] == "player_0"
    assert artifact["frame_count"]() == 3
    assert artifact["frame_at"](2)["board_text"] == "applied-1"


def test_pettingzoo_replay_infers_env_id_from_current_game_arena_metadata(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        pettingzoo_replay,
        "_can_import_env",
        lambda env_id: env_id == "pettingzoo.atari.space_invaders_v2",
    )

    resolved = pettingzoo_replay._infer_env_id(
        {
            "metadata": {
                "game_id": "pettingzoo_atari_space_invaders_v2",
                "game_arena": {"env": "space_invaders"},
            }
        },
        "",
    )

    assert resolved == "pettingzoo.atari.space_invaders_v2"


def test_pettingzoo_replay_accepts_result_move_log_from_current_sample_envelope() -> None:
    resolved = pettingzoo_replay._resolve_game_log(
        {
            "predict_result": [
                {
                    "result": {
                        "move_log": [
                            {"index": 1, "player": "pilot_alpha", "move": "RIGHTFIRE"},
                            {"index": 2, "player": "pilot_beta", "move": "LEFT"},
                        ]
                    }
                }
            ]
        }
    )

    assert resolved == [
        {"index": 1, "player": "pilot_alpha", "move": "RIGHTFIRE"},
        {"index": 2, "player": "pilot_beta", "move": "LEFT"},
    ]
