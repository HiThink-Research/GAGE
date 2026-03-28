from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import fields, is_dataclass
from types import MappingProxyType
from typing import Any

from gage_eval.role.arena.output.models import ArenaOutput


def _freeze_value(value):
    if is_dataclass(value):
        frozen_fields = {
            field.name: _freeze_value(getattr(value, field.name))
            for field in fields(value)
        }
        return type(value)(**frozen_fields)
    if isinstance(value, Mapping):
        frozen_mapping = {key: _freeze_value(item) for key, item in value.items()}
        return MappingProxyType(frozen_mapping)
    if isinstance(value, list):
        return tuple(_freeze_value(item) for item in value)
    if isinstance(value, tuple):
        return tuple(_freeze_value(item) for item in value)
    if isinstance(value, set):
        return frozenset(_freeze_value(item) for item in value)
    if isinstance(value, frozenset):
        return frozenset(_freeze_value(item) for item in value)
    return value


def _read_value(source: Any, key: str) -> Any:
    if isinstance(source, Mapping):
        return source.get(key)
    return getattr(source, key, None)


def _coerce_optional_str(value: Any) -> str | None:
    if value in (None, ""):
        return None
    return str(value)


def _coerce_int(value: Any, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _coerce_float_map(value: Any) -> dict[str, float] | None:
    if not isinstance(value, Mapping):
        return None
    normalized: dict[str, float] = {}
    for key, item in value.items():
        try:
            normalized[str(key)] = float(item)
        except (TypeError, ValueError):
            continue
    return normalized or None


def _coerce_sequence_list(value: Any) -> list[Any] | None:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        return None
    return list(value)


def _build_header_contract(sample: Any) -> dict[str, object]:
    runtime_overrides = _read_value(sample, "runtime_overrides")
    if not isinstance(runtime_overrides, Mapping):
        runtime_overrides = {}

    header: dict[str, object] = {
        "game_kit": str(_read_value(sample, "game_kit") or "unknown"),
        "env": _read_value(sample, "env"),
    }
    scheduler = _read_value(sample, "scheduler")
    if scheduler is None:
        scheduler = runtime_overrides.get("scheduler") or runtime_overrides.get("scheduler_binding")
    if scheduler not in (None, ""):
        header["scheduler"] = str(scheduler)

    players = _read_value(sample, "players")
    if isinstance(players, Sequence) and not isinstance(players, (str, bytes)):
        player_ids = []
        for player in players:
            if not isinstance(player, Mapping):
                continue
            player_id = (
                player.get("player_id")
                or player.get("id")
                or player.get("name")
                or player.get("seat")
            )
            if player_id not in (None, ""):
                player_ids.append(str(player_id))
        if player_ids:
            header["player_ids"] = player_ids
    return header


def _build_footer_contract(
    result: Any,
    arena_trace: Sequence[Mapping[str, object]],
) -> dict[str, object] | None:
    if result is None:
        if not arena_trace:
            return None
        return {
            "total_steps": len(arena_trace),
            "termination_reason": "unknown",
        }

    footer: dict[str, object] = {
        "winner_player_id": _coerce_optional_str(_read_value(result, "winner")),
        "termination_reason": str(
            _read_value(result, "reason")
            or _read_value(result, "result")
            or "unknown"
        ),
        "total_steps": max(
            0,
            _coerce_int(_read_value(result, "move_count"), len(arena_trace)),
        ),
    }
    ranks = _coerce_sequence_list(_read_value(result, "ranks"))
    if ranks is not None:
        footer["ranks"] = ranks
    final_scores = _coerce_float_map(
        _read_value(result, "final_scores") or _read_value(result, "scores")
    )
    if final_scores is not None:
        footer["final_scores"] = final_scores
    episode_returns = _coerce_float_map(
        _read_value(result, "episode_returns") or _read_value(result, "returns")
    )
    if episode_returns is not None:
        footer["episode_returns"] = episode_returns
    return footer


def _build_artifacts_contract(session: Any, result: Any) -> dict[str, object] | None:
    artifacts: dict[str, object] = {}
    replay_path = _coerce_optional_str(_read_value(result, "replay_path"))
    if replay_path is not None:
        artifacts["replay_ref"] = replay_path

    visual_session_ref = None
    recorder = _read_value(session, "visual_recorder")
    recorder_artifacts = _read_value(recorder, "artifacts") if recorder is not None else None
    if recorder_artifacts is not None:
        visual_session_ref = _coerce_optional_str(
            _read_value(recorder_artifacts, "visual_session_ref")
        )
    if visual_session_ref is not None:
        artifacts["visual_session_ref"] = visual_session_ref
    return artifacts or None


class ArenaOutputWriter:
    def finalize(self, session) -> ArenaOutput:
        frozen_sample = _freeze_value(session.sample)
        result = session.ensure_result()
        frozen_result = _freeze_value(result)
        frozen_trace = tuple(_freeze_value(entry) for entry in session.arena_trace)
        return ArenaOutput(
            sample=frozen_sample,
            tick=session.tick,
            step=session.step,
            result=frozen_result,
            arena_trace=frozen_trace,
            header=_freeze_value(_build_header_contract(session.sample)),
            trace=frozen_trace,
            footer=_freeze_value(_build_footer_contract(result, frozen_trace)),
            artifacts=_freeze_value(_build_artifacts_contract(session, result)),
        )
