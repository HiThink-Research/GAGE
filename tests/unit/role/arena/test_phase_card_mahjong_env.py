from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

from gage_eval.game_kits.phase_card_game.mahjong.envs.riichi_4p import (
    Riichi4pEnvironment,
)
from gage_eval.role.arena.games.mahjong import env as mahjong_env_module
from gage_eval.role.arena.games.mahjong.env import MahjongArena
from gage_eval.role.arena.types import ArenaAction


def test_riichi_4p_environment_wraps_mahjong_arena(tmp_path: Path) -> None:
    player_specs = [
        SimpleNamespace(player_id="east", display_name="East"),
        SimpleNamespace(player_id="south", display_name="South"),
        SimpleNamespace(player_id="west", display_name="West"),
        SimpleNamespace(player_id="north", display_name="North"),
    ]

    environment = Riichi4pEnvironment(
        player_specs=player_specs,
        replay_output_dir=str(tmp_path),
        replay_filename="mahjong_replay.json",
        sample_id="mahjong_sample_1",
    )

    assert isinstance(environment._adapter, MahjongArena)

    result = None
    scripted_moves = [
        ("east", "B1"),
        ("south", "C1"),
        ("west", "D1"),
        ("north", "East"),
        ("east", "Red"),
    ]
    for player_id, move in scripted_moves:
        result = environment.apply(
            ArenaAction(player=player_id, move=move, raw=move)
        )

    assert result is not None
    assert result.winner == "east"
    assert result.result == "win"
    assert result.replay_path == str(tmp_path / "mahjong_sample_1" / "replay.json")


def test_riichi_4p_environment_replay_keeps_terminal_metadata(tmp_path: Path) -> None:
    player_specs = [
        SimpleNamespace(player_id="east", display_name="East"),
        SimpleNamespace(player_id="south", display_name="South"),
        SimpleNamespace(player_id="west", display_name="West"),
        SimpleNamespace(player_id="north", display_name="North"),
    ]

    environment = Riichi4pEnvironment(
        player_specs=player_specs,
        replay_output_dir=str(tmp_path),
        replay_filename="mahjong_replay.json",
        sample_id="mahjong_sample_2",
    )

    scripted_moves = [
        ("east", "B1"),
        ("south", "C1"),
        ("west", "D1"),
        ("north", "East"),
        ("east", "Red"),
    ]
    result = None
    for player_id, move in scripted_moves:
        result = environment.apply(
            ArenaAction(player=player_id, move=move, raw=move)
        )

    assert result is not None
    replay_payload = json.loads((tmp_path / "mahjong_sample_2" / "replay.json").read_text(encoding="utf-8"))
    assert replay_payload["winner"] == "east"
    assert replay_payload["result"] == "win"
    assert replay_payload["result_reason"] == "terminal"
    assert replay_payload["end_reason"] == "hu"


def test_riichi_4p_environment_uses_explicit_core_when_make_core_unavailable(
    tmp_path: Path,
    monkeypatch,
) -> None:
    player_specs = [
        SimpleNamespace(player_id="east", display_name="East"),
        SimpleNamespace(player_id="south", display_name="South"),
        SimpleNamespace(player_id="west", display_name="West"),
        SimpleNamespace(player_id="north", display_name="North"),
    ]

    monkeypatch.setattr(
        mahjong_env_module,
        "make_core",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("make_core should not be called")),
    )

    environment = Riichi4pEnvironment(
        player_specs=player_specs,
        replay_output_dir=str(tmp_path),
        replay_filename="mahjong_replay.json",
    )

    assert isinstance(environment._adapter, MahjongArena)
    assert environment.get_active_player() == "east"
    result = environment.apply(ArenaAction(player="east", move="B1", raw="B1"))
    assert result is None
