from __future__ import annotations

import os

import numpy as np
import pytest

from gage_eval.role.arena.games.retro.action_codec import RetroActionCodec
from gage_eval.role.arena.games.retro.observation import ActionSchema, InfoDeltaFeeder, InfoLastFeeder, ObservationBuilder
from gage_eval.role.arena.games.retro.replay import ReplaySchemaWriter
from gage_eval.role.arena.games.retro.retro_env import StableRetroArenaEnvironment
from gage_eval.role.arena.games.retro.temporal.data_contract_v15 import (
    build_action_dict,
    build_observation_dict,
    build_result_dict,
)
from gage_eval.role.arena.types import ArenaAction, GameResult


def test_action_codec_legal_moves_and_encode() -> None:
    codec = RetroActionCodec(buttons=["LEFT", "RIGHT", "A", "B", "START"])

    legal = codec.legal_moves()
    assert "noop" in legal
    assert "right" in legal

    encoded = codec.encode("right_jump")
    assert encoded.buttons
    assert "RIGHT" in encoded.pressed
    assert "A" in encoded.pressed


def test_action_codec_unknown_move_raises() -> None:
    codec = RetroActionCodec(buttons=["LEFT", "RIGHT", "A"])

    with pytest.raises(ValueError, match="unknown retro move"):
        codec.encode("teleport")


def test_action_codec_legal_moves_override_filters() -> None:
    codec = RetroActionCodec(
        buttons=["LEFT", "RIGHT", "A", "B"],
        legal_moves=["right", "noop", "not_a_move"],
    )
    assert codec.legal_moves() == ["right", "noop"]


def test_action_codec_macro_map_override_and_noop_added() -> None:
    codec = RetroActionCodec(
        buttons=["LEFT", "RIGHT", "A"],
        macro_map={"dash": ["RIGHT", "A"]},
    )
    assert "dash" in codec.legal_moves()
    assert "noop" in codec.legal_moves()


def test_observation_builder_stores_contract_dict_in_metadata() -> None:
    builder = ObservationBuilder(
        info_feeder=InfoDeltaFeeder(window_size=2),
        action_schema=ActionSchema(hold_ticks_min=1, hold_ticks_max=10, default_hold_ticks=3),
        token_budget=120,
    )
    obs = builder.build(
        player_id="player_0",
        active_player="player_0",
        legal_moves=["noop", "right"],
        last_move="noop",
        tick=5,
        decision_count=2,
        info_history=[{"x": 1}, {"x": 2}],
        raw_info={"x": 2},
        reward_total=1.5,
    )

    assert obs.metadata.get("contract_v15")
    assert obs.metadata.get("prompt_text")
    assert obs.metadata.get("info_projection")


def test_observation_builder_truncates_info_text() -> None:
    builder = ObservationBuilder(
        info_feeder=InfoLastFeeder(),
        action_schema=ActionSchema(),
        token_budget=2,
    )
    obs = builder.build(
        player_id="player_0",
        active_player="player_0",
        legal_moves=["noop"],
        last_move=None,
        tick=1,
        decision_count=1,
        info_history=[{"long": "x" * 200}],
        raw_info={"long": "x" * 200},
        reward_total=0.0,
    )
    prompt = obs.metadata.get("prompt_text") or ""
    assert "..." in prompt


def test_temporal_contract_builders() -> None:
    obs = build_observation_dict(
        view_text="hello",
        legal_actions=["noop"],
        active_player="player_0",
        tick=1,
        step=2,
        info={"x": 1},
    )
    assert obs["context"]["mode"] == "tick"
    assert obs["extra"]["info"]["x"] == 1

    action = build_action_dict(player="player_0", move="noop", raw="noop", hold_ticks=3)
    assert action["hold_ticks"] == 3

    result = build_result_dict(status="win", reason="terminated", winner="player_0", replay_path=None)
    assert result["status"] == "win"


def test_replay_schema_writer_records_decision_fields() -> None:
    writer = ReplaySchemaWriter(
        game="test",
        state=None,
        run_id=None,
        sample_id="s0",
        replay_output_dir=None,
        replay_filename=None,
        frame_output_dir=None,
        frame_stride=1,
        snapshot_stride=1,
        rom_path=None,
    )
    action = ArenaAction(player="player_0", move="right", raw="raw", metadata={"hold_ticks": 4})
    writer.append_decision(action, start_tick=3, end_tick=6)
    assert writer._moves, "Expected moves to be recorded"
    move = writer._moves[0]
    assert move["hold_ticks"] == 4
    assert move["start_tick"] == 3


@pytest.mark.io
def test_replay_schema_writer_writes_replay_and_frame(tmp_path) -> None:
    writer = ReplaySchemaWriter(
        game="test",
        state="Start",
        run_id="run_1",
        sample_id="sample_1",
        replay_output_dir=str(tmp_path),
        replay_filename="replay.json",
        frame_output_dir=str(tmp_path),
        frame_stride=1,
        snapshot_stride=1,
        rom_path=None,
    )
    frame = np.zeros((2, 2, 3), dtype=np.uint8)
    writer.append_tick(tick=1, reward=1.0, info={"x": 1}, frame=frame, done=False)
    result = GameResult(
        winner=None,
        result="draw",
        reason="terminated",
        move_count=0,
        illegal_move_count=0,
        final_board="",
        move_log=[],
    )
    replay_path = writer.finalize(result)
    assert replay_path is not None
    assert (tmp_path / "replay.json").exists()
    assert any(path.name.startswith("frame_") for path in tmp_path.iterdir())


def test_retro_env_derive_result_paths() -> None:
    assert StableRetroArenaEnvironment._derive_result(False, True, {}) == "draw"
    assert StableRetroArenaEnvironment._derive_result(True, False, {"win": True}) == "win"
    assert StableRetroArenaEnvironment._derive_result(True, False, {}) == "loss"


def test_retro_env_record_path_resolution_with_run_id(tmp_path) -> None:
    env = StableRetroArenaEnvironment(
        game="TestGame",
        state="Start",
        record_bk2=True,
        run_id="run_123",
        sample_id="sample_1",
    )
    os.environ["GAGE_EVAL_SAVE_DIR"] = str(tmp_path)
    path = env._resolve_record_output_path()
    assert path is not None
    assert "run_123" in str(path)
    assert str(path).endswith(".bk2")
