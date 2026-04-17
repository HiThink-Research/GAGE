from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import fields, is_dataclass, replace
import json
from types import MappingProxyType
from typing import Any

from gage_eval.role.arena.output.models import ArenaOutput
from gage_eval.role.arena.types import GameResult


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


def _coerce_error_list(value: Any) -> list[dict[str, object]] | None:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        return None
    normalized: list[dict[str, object]] = []
    for item in value:
        if not isinstance(item, Mapping):
            continue
        normalized.append({str(key): element for key, element in item.items()})
    return normalized or None


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

    total_steps = _coerce_int(_read_value(result, "move_count"), len(arena_trace))
    if arena_trace:
        total_steps = max(total_steps, len(arena_trace))
    footer: dict[str, object] = {
        "winner_player_id": _coerce_optional_str(_read_value(result, "winner")),
        "termination_reason": str(
            _read_value(result, "reason")
            or _read_value(result, "result")
            or "unknown"
        ),
        "total_steps": max(0, total_steps),
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
    resources = _read_value(session, "resources")
    resource_artifacts = _read_value(resources, "resource_artifacts")
    if isinstance(resource_artifacts, Mapping):
        artifacts.update({str(key): value for key, value in resource_artifacts.items()})
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
    visual_artifacts_error = _coerce_optional_str(_read_value(session, "_visual_artifacts_error"))
    if visual_artifacts_error is not None:
        artifacts["artifact_error"] = {
            "error_code": "visualization_failure",
            "message": visual_artifacts_error,
        }
    return artifacts or None


def _build_game_context_contract(session: Any) -> dict[str, object]:
    sample = _read_value(session, "sample")
    resources = _read_value(session, "resources")
    payload: dict[str, object] = {
        "game_kit": str(_read_value(sample, "game_kit") or "unknown"),
        "env": _read_value(sample, "env"),
    }
    support_workflow = _coerce_optional_str(
        _read_value(_read_value(session, "support_workflow"), "workflow_id")
    )
    if support_workflow is not None:
        payload["support_workflow"] = support_workflow
    visualization_spec = _coerce_optional_str(
        _read_value(_read_value(session, "visualization_spec"), "spec_id")
    )
    if visualization_spec is not None:
        payload["visualization_spec"] = visualization_spec
    resource_categories = _read_value(resources, "resource_categories")
    if isinstance(resource_categories, Sequence) and not isinstance(resource_categories, (str, bytes)):
        payload["resource_categories"] = [str(item) for item in resource_categories]
    lifecycle_phase = _coerce_optional_str(_read_value(resources, "lifecycle_phase"))
    if lifecycle_phase is not None:
        payload["resource_lifecycle_phase"] = lifecycle_phase
    support_errors = _coerce_error_list(_read_value(session, "support_errors"))
    if support_errors is not None:
        payload["support_errors"] = support_errors
    resource_errors = _coerce_error_list(_read_value(resources, "errors"))
    if resource_errors is not None:
        payload["resource_errors"] = resource_errors
    return payload


def _attach_arena_trace_to_result(
    result: Any,
    arena_trace: Sequence[Mapping[str, object]],
) -> Any:
    if result is None or not arena_trace:
        return result

    frozen_trace = tuple(dict(entry) for entry in arena_trace)
    if isinstance(result, GameResult):
        if tuple(result.arena_trace) == frozen_trace:
            return result
        return replace(result, arena_trace=frozen_trace)

    if isinstance(result, Mapping):
        existing_trace = result.get("arena_trace")
        if isinstance(existing_trace, Sequence) and not isinstance(existing_trace, (str, bytes)):
            if list(existing_trace) == list(frozen_trace):
                return result
        enriched = dict(result)
        enriched["arena_trace"] = list(frozen_trace)
        return enriched

    if is_dataclass(result) and hasattr(result, "arena_trace"):
        existing_trace = getattr(result, "arena_trace", ())
        if isinstance(existing_trace, Sequence) and not isinstance(existing_trace, (str, bytes)):
            if list(existing_trace) == list(frozen_trace):
                return result
        try:
            return replace(result, arena_trace=frozen_trace)
        except TypeError:
            return result

    return result


def _result_to_mapping(result: Any) -> dict[str, Any] | None:
    if isinstance(result, Mapping):
        return dict(result)
    if is_dataclass(result):
        return {
            field.name: getattr(result, field.name)
            for field in fields(result)
        }
    return None


def _parse_sectioned_final_board(final_board: str) -> dict[str, Any] | None:
    section_headers = {
        "Public State:": "public_state",
        "Private State:": "private_state",
        "Chat Log:": "chat_log",
        "UI_STATE_JSON:": "ui_state",
    }
    structured: dict[str, Any] = {}
    current_key: str | None = None
    current_lines: list[str] = []

    def _flush_current() -> None:
        nonlocal current_key, current_lines
        if current_key is None:
            current_lines = []
            return
        raw_value = "\n".join(current_lines).strip()
        if not raw_value:
            current_key = None
            current_lines = []
            return
        try:
            structured[current_key] = json.loads(raw_value)
        except json.JSONDecodeError:
            structured[current_key] = raw_value
        current_key = None
        current_lines = []

    for line in final_board.splitlines():
        stripped = line.strip()
        if stripped in section_headers:
            _flush_current()
            current_key = section_headers[stripped]
            continue
        if stripped.startswith("Legal Moves (preview):"):
            _flush_current()
            preview_text = stripped.split(":", 1)[1].strip()
            structured["legal_moves_preview"] = (
                []
                if preview_text in {"", "none"}
                else [item.strip() for item in preview_text.split(",") if item.strip()]
            )
            continue
        if current_key is None:
            continue
        if not stripped and not current_lines:
            continue
        current_lines.append(line)

    _flush_current()
    return structured or None


def _parse_structured_final_board(final_board: Any) -> Any:
    if not isinstance(final_board, str):
        return None
    stripped = final_board.strip()
    if not stripped:
        return None
    if stripped.startswith("{") or stripped.startswith("["):
        try:
            return json.loads(stripped)
        except json.JSONDecodeError:
            pass
    return _parse_sectioned_final_board(stripped)


def _enrich_result_for_output(
    result: Any,
    arena_trace: Sequence[Mapping[str, object]],
) -> Any:
    with_trace = _attach_arena_trace_to_result(result, arena_trace)
    payload = _result_to_mapping(with_trace)
    if payload is None:
        return with_trace
    structured_final_board = _parse_structured_final_board(payload.get("final_board"))
    if structured_final_board is not None:
        payload["final_board_structured"] = structured_final_board
        return payload
    return with_trace


class ArenaOutputWriter:
    def finalize(self, session) -> ArenaOutput:
        frozen_sample = _freeze_value(session.sample)
        result = _enrich_result_for_output(
            session.ensure_result(),
            session.arena_trace,
        )
        frozen_result = _freeze_value(result)
        frozen_trace = tuple(_freeze_value(entry) for entry in session.arena_trace)
        resource_artifacts = _freeze_value(_build_artifacts_contract(session, result))
        return ArenaOutput(
            sample=frozen_sample,
            tick=session.tick,
            step=session.step,
            result=frozen_result,
            arena_trace=frozen_trace,
            header=_freeze_value(_build_header_contract(session.sample)),
            trace=frozen_trace,
            footer=_freeze_value(_build_footer_contract(result, frozen_trace)),
            resource_artifacts=resource_artifacts,
            game_context=_freeze_value(_build_game_context_contract(session)),
            artifacts=resource_artifacts,
        )
