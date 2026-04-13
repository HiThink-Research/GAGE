from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

from gage_eval.game_kits.aec_env_game.pettingzoo.envs.space_invaders import (
    SpaceInvadersEnvironment,
)
from gage_eval.game_kits.board_game.gomoku.envs.gomoku_standard import (
    GomokuStandardEnvironment,
)
from gage_eval.game_kits.board_game.tictactoe.envs.tictactoe_standard import (
    TicTacToeStandardEnvironment,
)
from gage_eval.role.arena.core.invocation import GameArenaInvocationContext
from gage_eval.role.arena.core.types import ArenaSample
from gage_eval.role.arena.types import ArenaAction


class _Player:
    def __init__(self, player_id: str) -> None:
        self.player_id = player_id
        self.display_name = player_id


def _build_invocation(*, run_id: str, sample_id: str) -> GameArenaInvocationContext:
    return GameArenaInvocationContext(
        adapter_id="gamekit_replay_writer_test",
        sample_id=sample_id,
        trace=SimpleNamespace(run_id=run_id, sample_id=sample_id),
        sample_payload={"id": sample_id},
    )


def _load_replay_payload(result) -> tuple[Path, dict[str, object]]:
    replay_path = Path(str(result.replay_path))
    payload = json.loads(replay_path.read_text(encoding="utf-8"))
    assert isinstance(payload, dict)
    return replay_path, payload


def test_tictactoe_gamekit_writer_materializes_replay_manifest(temp_workspace: Path) -> None:
    sample_id = "tictactoe_writer_sample"
    env = TicTacToeStandardEnvironment.from_runtime(
        sample=ArenaSample(game_kit="tictactoe", env="tictactoe_standard"),
        resolved=SimpleNamespace(
            game_kit=SimpleNamespace(defaults={}),
            env_spec=SimpleNamespace(defaults={}),
        ),
        resources={},
        player_specs=(_Player("X"), _Player("O")),
        invocation_context=_build_invocation(run_id="run-tictactoe-writer", sample_id=sample_id),
    )

    result = None
    for player_id, coord in (("X", "1,1"), ("O", "1,2"), ("X", "2,2"), ("O", "1,3"), ("X", "3,3")):
        result = env.apply(ArenaAction(player=player_id, move=coord, raw=coord))

    assert result is not None
    replay_path, payload = _load_replay_payload(result)

    assert replay_path.exists()
    assert replay_path.name == "replay.json"
    assert replay_path.parent.name == sample_id
    assert payload["schema"] == "gage_replay/v1"
    assert payload["meta"]["game_kit"] == "tictactoe"
    assert payload["meta"]["env"] == "tictactoe_standard"


def test_gomoku_gamekit_writer_materializes_replay_manifest(temp_workspace: Path) -> None:
    sample_id = "gomoku_writer_sample"
    env = GomokuStandardEnvironment.from_runtime(
        sample=ArenaSample(game_kit="gomoku", env="gomoku_standard"),
        resolved=SimpleNamespace(
            game_kit=SimpleNamespace(defaults={}),
            env_spec=SimpleNamespace(defaults={"board_size": 5, "win_len": 4}),
        ),
        resources={},
        player_specs=(_Player("Black"), _Player("White")),
        invocation_context=_build_invocation(run_id="run-gomoku-writer", sample_id=sample_id),
    )

    result = None
    for player_id, coord in (
        ("Black", "A1"),
        ("White", "A2"),
        ("Black", "B1"),
        ("White", "B2"),
        ("Black", "C1"),
        ("White", "C2"),
        ("Black", "D1"),
    ):
        result = env.apply(ArenaAction(player=player_id, move=coord, raw=coord))

    assert result is not None
    replay_path, payload = _load_replay_payload(result)

    assert replay_path.exists()
    assert replay_path.name == "replay.json"
    assert replay_path.parent.name == sample_id
    assert payload["schema"] == "gage_replay/v1"
    assert payload["meta"]["game_kit"] == "gomoku"
    assert payload["meta"]["env"] == "gomoku_standard"


def test_space_invaders_gamekit_writer_materializes_replay_manifest(temp_workspace: Path) -> None:
    sample_id = "space_invaders_writer_sample"
    env = SpaceInvadersEnvironment.from_runtime(
        sample=ArenaSample(game_kit="pettingzoo", env="space_invaders"),
        resolved=SimpleNamespace(
            game_kit=SimpleNamespace(defaults={}),
            env_spec=SimpleNamespace(
                defaults={
                    "env_id": "pettingzoo.atari.space_invaders_v2",
                    "backend_mode": "dummy",
                    "max_cycles": 4,
                }
            ),
        ),
        resources={},
        player_specs=(_Player("pilot_alpha"), _Player("pilot_beta")),
        invocation_context=_build_invocation(run_id="run-space-invaders-writer", sample_id=sample_id),
    )

    result = None
    while result is None:
        active_player = env.get_active_player()
        observation = env.observe(active_player)
        result = env.apply(
            ArenaAction(
                player=active_player,
                move=observation.legal_moves[0],
                raw=observation.legal_moves[0],
            )
        )

    replay_path, payload = _load_replay_payload(result)

    assert replay_path.exists()
    assert replay_path.name == "replay.json"
    assert replay_path.parent.name == sample_id
    assert payload["schema"] == "gage_replay/v1"
    assert payload["meta"]["game_kit"] == "pettingzoo"
    assert payload["meta"]["env"] == "space_invaders"
