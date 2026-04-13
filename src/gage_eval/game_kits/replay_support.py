"""Shared helpers for GameKit-owned replay artifact materialization."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Any, Mapping, Sequence

from gage_eval.role.arena.replay_paths import resolve_replay_manifest_path
from gage_eval.role.arena.replay_schema_writer import ReplaySchemaWriter
from gage_eval.role.arena.types import GameResult


def materialize_gamekit_replay(
    *,
    result: GameResult,
    game_kit: str,
    env: str,
    run_id: str | None,
    sample_id: str | None,
    replay_output_dir: str | None = None,
    arena_trace: Sequence[Mapping[str, Any]] | None = None,
    scheduler_type: str = "turn",
    extra_meta: Mapping[str, Any] | None = None,
) -> GameResult:
    """Write a canonical replay manifest for one GameKit result when possible."""

    if result.replay_path not in (None, ""):
        return result
    replay_path = resolve_replay_manifest_path(
        run_id=run_id,
        sample_id=sample_id,
        output_dir=replay_output_dir,
    )
    if replay_path is None:
        return result
    replay_file = replay_path.expanduser()
    materialized_result = replace(result, replay_path=str(replay_file))
    ReplaySchemaWriter(
        run_dir=_resolve_replay_writer_run_dir(replay_file),
        sample_id=replay_file.parent.name,
        output_dir=str(replay_file.parent.parent),
    ).write(
        scheduler_type=str(scheduler_type or "turn"),
        result=materialized_result,
        move_log=[dict(entry) for entry in materialized_result.move_log],
        arena_trace=arena_trace,
        extra_meta={
            "game_kit": str(game_kit),
            "env": str(env),
            **dict(extra_meta or {}),
        },
    )
    return materialized_result


def _resolve_replay_writer_run_dir(replay_file: Path) -> Path:
    parents = replay_file.parents
    if len(parents) >= 3:
        return parents[2]
    return replay_file.parent
