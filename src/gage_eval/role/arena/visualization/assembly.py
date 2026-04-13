from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from typing import Any

from gage_eval.game_kits.contracts import GameVisualizationSpec
from gage_eval.game_kits.board_game.gomoku.coord_scheme import GomokuCoordCodec, normalize_coord_scheme
from gage_eval.role.arena.visualization.contracts import (
    MediaSourceRef,
    TimelineEvent,
    VisualScene,
    VisualSceneMedia,
    VisualSession,
)


def assemble_visual_scene(
    *,
    visual_session: VisualSession,
    event: TimelineEvent,
    snapshot_anchor: Mapping[str, Any] | None = None,
    snapshot_body: Any = None,
    visualization_spec: GameVisualizationSpec | None = None,
) -> VisualScene:
    plugin_id = (
        visualization_spec.plugin_id
        if visualization_spec is not None
        else visual_session.plugin_id
    )
    kind = (
        visualization_spec.visual_kind
        if visualization_spec is not None
        else _infer_visual_kind(plugin_id)
    )
    media = _extract_scene_media(snapshot_body, event.payload, kind=kind)
    overlays: tuple[dict[str, Any], ...] = ()
    if kind == "board":
        body, legal_actions, active_player_id = _project_board_scene(
            visual_session=visual_session,
            snapshot_body=snapshot_body,
            event_payload=event.payload,
            visualization_spec=visualization_spec,
        )
    elif kind == "table":
        body, legal_actions, active_player_id = _project_table_scene(
            visual_session=visual_session,
            snapshot_body=snapshot_body,
            event_payload=event.payload,
            visualization_spec=visualization_spec,
        )
    elif kind == "frame":
        body, legal_actions, active_player_id, overlays = _project_frame_scene(
            visual_session=visual_session,
            snapshot_body=snapshot_body,
            event_payload=event.payload,
            visualization_spec=visualization_spec,
        )
    elif kind == "rts":
        body, legal_actions, active_player_id, overlays = _project_rts_scene(
            visual_session=visual_session,
            snapshot_body=snapshot_body,
            event_payload=event.payload,
            visualization_spec=visualization_spec,
        )
    else:
        active_player_id = _extract_active_player(snapshot_body, event.payload)
        legal_actions = _extract_legal_actions(snapshot_body, event.payload)
        body = _build_default_scene_body(
            event=event,
            snapshot_anchor=snapshot_anchor,
            snapshot_body=snapshot_body,
        )

    summary: dict[str, Any] = {
        "eventType": event.type,
        "eventLabel": event.label,
    }
    if snapshot_anchor is not None:
        summary["snapshotSeq"] = int(snapshot_anchor["seq"])
        if snapshot_anchor.get("label") is not None:
            summary["snapshotLabel"] = snapshot_anchor["label"]
    if kind == "board":
        board_payload = body.get("board") if isinstance(body, Mapping) else None
        if isinstance(board_payload, Mapping):
            board_size = board_payload.get("size")
            coord_scheme = board_payload.get("coordScheme")
            if isinstance(board_size, int):
                summary["boardSize"] = board_size
            if isinstance(coord_scheme, str):
                summary["coordScheme"] = coord_scheme
    elif kind == "table":
        table_payload = body.get("table") if isinstance(body, Mapping) else None
        if isinstance(table_payload, Mapping):
            seats_payload = table_payload.get("seats")
            layout = table_payload.get("layout")
            if isinstance(seats_payload, Sequence) and not isinstance(seats_payload, (str, bytes)):
                summary["seatCount"] = len(seats_payload)
            if isinstance(layout, str):
                summary["tableLayout"] = layout
    elif kind == "frame":
        frame_payload = body.get("frame") if isinstance(body, Mapping) else None
        if isinstance(frame_payload, Mapping):
            title = frame_payload.get("title")
            stream_id = frame_payload.get("streamId")
            viewport = frame_payload.get("viewport")
            if isinstance(title, str):
                summary["frameTitle"] = title
            if isinstance(stream_id, str):
                summary["streamId"] = stream_id
            if isinstance(viewport, Mapping):
                width = viewport.get("width")
                height = viewport.get("height")
                if isinstance(width, int):
                    summary["frameWidth"] = width
                if isinstance(height, int):
                    summary["frameHeight"] = height
    elif kind == "rts":
        frame_payload = body.get("frame") if isinstance(body, Mapping) else None
        if isinstance(frame_payload, Mapping):
            title = frame_payload.get("title")
            stream_id = frame_payload.get("streamId")
            viewport = frame_payload.get("viewport")
            if isinstance(title, str):
                summary["frameTitle"] = title
            if isinstance(stream_id, str):
                summary["streamId"] = stream_id
            if isinstance(viewport, Mapping):
                width = viewport.get("width")
                height = viewport.get("height")
                if isinstance(width, int):
                    summary["frameWidth"] = width
                if isinstance(height, int):
                    summary["frameHeight"] = height

    return VisualScene(
        scene_id=f"{visual_session.session_id}:seq:{event.seq}",
        game_id=visual_session.game_id,
        plugin_id=plugin_id,
        kind=kind,
        ts_ms=event.ts_ms,
        seq=event.seq,
        phase="replay",
        active_player_id=active_player_id,
        legal_actions=legal_actions,
        summary=summary,
        body=body,
        media=media,
        overlays=overlays,
    )


def collect_scene_media_refs(scene: VisualScene) -> dict[str, MediaSourceRef]:
    refs: dict[str, MediaSourceRef] = {}
    if scene.media is None:
        return refs
    if scene.media.primary is not None:
        refs[scene.media.primary.media_id] = scene.media.primary
    for item in scene.media.auxiliary:
        refs[item.media_id] = item
    return refs


def _build_default_scene_body(
    *,
    event: TimelineEvent,
    snapshot_anchor: Mapping[str, Any] | None,
    snapshot_body: Any,
) -> dict[str, Any]:
    body: dict[str, Any] = {
        "event": {
            "seq": event.seq,
            "tsMs": event.ts_ms,
            "type": event.type,
            "label": event.label,
            "payload": event.payload,
        }
    }
    if snapshot_body is not None:
        body["snapshot"] = snapshot_body
    if snapshot_anchor is not None:
        body["snapshotAnchor"] = {
            key: value
            for key, value in snapshot_anchor.items()
            if key != "snapshotRef"
        }
    return body


def _project_board_scene(
    *,
    visual_session: VisualSession,
    snapshot_body: Any,
    event_payload: Any,
    visualization_spec: GameVisualizationSpec | None,
) -> tuple[dict[str, Any], tuple[dict[str, Any], ...], str | None]:
    source = _build_projection_source(snapshot_body, event_payload)
    if not source:
        source = {}
    _apply_board_result_overrides(
        source,
        result_payloads=_collect_result_payloads(event_payload, snapshot_body),
        snapshot_body=snapshot_body,
    )

    coord_scheme = normalize_coord_scheme(
        str(
            source.get("coord_scheme")
            or _scene_projection_rule_value(
                visualization_spec,
                "default_coord_scheme",
            )
            or "A1"
        )
    )
    board_rows = _parse_square_board_text(source.get("board_text"))
    board_size = (
        len(board_rows)
        if board_rows
        else _infer_board_size_from_coords(
            source.get("legal_moves"),
            source.get("winning_line"),
            source.get("last_move"),
            coord_scheme=coord_scheme,
        )
    )
    board_size = int(max(board_size, 0))

    codec = _build_coord_codec(board_size=board_size, coord_scheme=coord_scheme)
    legal_actions = _extract_board_legal_actions(source, codec=codec)
    legal_coords = {str(action["coord"]) for action in legal_actions if action.get("coord")}
    last_move = _normalize_coord(source.get("last_move"))
    winning_line = _normalize_coord_sequence(source.get("winning_line"))
    winning_cells = set(winning_line)

    player_ids = _normalize_player_ids(source.get("player_ids"), source.get("player_names"))
    player_names = _normalize_player_names(source.get("player_names"), player_ids=player_ids)
    ordered_tokens = _normalize_tokens(
        _scene_projection_rule_value(visualization_spec, "ordered_player_tokens")
    )

    players: list[dict[str, str]] = []
    token_lookup: dict[str, dict[str, str]] = {}
    for index, player_id in enumerate(player_ids):
        token = (
            ordered_tokens[index]
            if index < len(ordered_tokens)
            else f"P{index + 1}"
        )
        player_name = player_names.get(player_id, player_id)
        players.append(
            {
                "playerId": player_id,
                "playerName": player_name,
                "token": token,
            }
        )
        token_lookup[token.upper()] = {
            "playerId": player_id,
            "playerName": player_name,
        }

    grid = _build_projected_grid(board_rows=board_rows, board_size=board_size)
    cells: list[dict[str, Any]] = []
    if codec is not None:
        for row_idx in range(board_size):
            row_number = board_size - row_idx
            logical_row = row_number - 1
            for col_idx in range(board_size):
                coord = codec.index_to_coord(logical_row, col_idx)
                occupant = _normalize_board_occupant(grid[row_idx][col_idx])
                player = token_lookup.get(occupant.upper(), {}) if occupant else {}
                cells.append(
                    {
                        "coord": coord,
                        "row": logical_row,
                        "col": col_idx,
                        "occupant": occupant,
                        "playerId": player.get("playerId"),
                        "playerName": player.get("playerName"),
                        "isLastMove": bool(last_move and coord == last_move),
                        "isWinningCell": coord in winning_cells,
                        "isLegalAction": coord in legal_coords,
                    }
                )

    active_player_id = _extract_active_player(source)
    observer_player_id = _resolve_board_observer_player(
        visual_session=visual_session,
        source=source,
    )
    move_count = _coerce_int(source.get("move_count"), default=0)

    body: dict[str, Any] = {
        "board": {
            "size": board_size,
            "coordScheme": coord_scheme,
            "cells": cells,
        },
        "players": players,
        "status": {
            "activePlayerId": active_player_id,
            "observerPlayerId": observer_player_id,
            "moveCount": move_count,
            "lastMove": last_move,
            "winningLine": winning_line,
        },
    }
    return body, legal_actions, active_player_id


def _collect_result_payloads(*payloads: Any) -> tuple[Mapping[str, Any], ...]:
    collected: list[Mapping[str, Any]] = []
    for payload in payloads:
        if not isinstance(payload, Mapping):
            continue
        result = payload.get("result")
        if isinstance(result, Mapping):
            collected.append(dict(result))
    return tuple(collected)


def _apply_board_result_overrides(
    source: dict[str, Any],
    *,
    result_payloads: Sequence[Mapping[str, Any]],
    snapshot_body: Any,
) -> None:
    if not result_payloads:
        return

    final_board = _first_result_string(result_payloads, "final_board", "finalBoard")
    if final_board is not None:
        source["board_text"] = final_board

    move_count = None
    raw_move_count = _first_result_value(result_payloads, "move_count", "moveCount")
    if raw_move_count is not None:
        move_count = _coerce_int(raw_move_count, default=0)
    if move_count is not None:
        source["move_count"] = move_count

    last_move = None
    for result_payload in result_payloads:
        last_move = _extract_result_last_move(result_payload)
        if last_move is not None:
            break
    if last_move is None:
        last_move = _extract_snapshot_trace_last_move(snapshot_body)
    if last_move is not None:
        source["last_move"] = last_move

    winning_line = _first_result_value(result_payloads, "winning_line", "winningLine")
    if winning_line is not None:
        source["winning_line"] = winning_line


def _extract_result_last_move(result_payload: Mapping[str, Any]) -> str | None:
    for key in ("last_move", "lastMove"):
        coord = _normalize_coord(result_payload.get(key))
        if coord is not None:
            return coord

    move_log = result_payload.get("move_log")
    if move_log is None:
        move_log = result_payload.get("moveLog")
    if isinstance(move_log, Sequence) and not isinstance(move_log, (str, bytes)):
        for entry in reversed(move_log):
            if not isinstance(entry, Mapping):
                continue
            if "__truncated__" in entry:
                return None
            coord = _normalize_coord(entry.get("coord") or entry.get("move"))
            if coord is not None:
                return coord
    return None


def _extract_snapshot_trace_last_move(snapshot_body: Any) -> str | None:
    if not isinstance(snapshot_body, Mapping):
        return None
    arena_trace = snapshot_body.get("arenaTrace")
    if not isinstance(arena_trace, Mapping):
        return None
    for key in ("action_applied", "action_raw"):
        coord = _normalize_coord(arena_trace.get(key))
        if coord is not None:
            return coord
    return None


def _first_result_string(
    result_payloads: Sequence[Mapping[str, Any]],
    *keys: str,
) -> str | None:
    value = _first_result_value(result_payloads, *keys)
    return _string_or_none(value)


def _first_result_value(
    result_payloads: Sequence[Mapping[str, Any]],
    *keys: str,
) -> Any:
    for result_payload in result_payloads:
        for key in keys:
            if key in result_payload and result_payload.get(key) is not None:
                return result_payload.get(key)
    return None


def _merge_mapping_payloads(*payloads: Any) -> dict[str, Any]:
    merged: dict[str, Any] = {}
    for payload in payloads:
        if isinstance(payload, Mapping):
            merged.update(dict(payload))
    return merged


def _build_projection_source(*payloads: Any) -> dict[str, Any]:
    source = _merge_mapping_payloads(*payloads)
    observation_layers = _flatten_observation_layers(source.get("observation"))
    for observation_layer in observation_layers:
        source.update(dict(observation_layer))
    observation = observation_layers[-1] if observation_layers else {}
    metadata = _mapping_or_empty(source.get("metadata"))
    if not metadata:
        for observation_layer in reversed(observation_layers):
            metadata = _mapping_or_empty(observation_layer.get("metadata"))
            if metadata:
                break
    context = _mapping_or_empty(source.get("context"))
    if not context:
        for observation_layer in reversed(observation_layers):
            context = _mapping_or_empty(observation_layer.get("context"))
            if context:
                break
    view = _mapping_or_empty(source.get("view"))
    if not view:
        for observation_layer in reversed(observation_layers):
            view = _mapping_or_empty(observation_layer.get("view"))
            if view:
                break

    _merge_defaults(source, observation)
    _merge_defaults(source, metadata)
    _merge_defaults(source, context)
    if observation:
        source["observation"] = observation
    if metadata:
        source["metadata"] = metadata
    if context:
        source["context"] = context
    if view:
        source["view"] = view
    return source


def _flatten_observation_layers(value: Any) -> tuple[dict[str, Any], ...]:
    layers: list[dict[str, Any]] = []
    current = _mapping_or_empty(value)
    while current:
        layers.append(current)
        nested = _mapping_or_empty(current.get("observation"))
        if not nested:
            break
        current = nested
    return tuple(layers)


def _merge_defaults(target: dict[str, Any], payload: Mapping[str, Any]) -> None:
    for key, value in payload.items():
        target.setdefault(str(key), value)


def _scene_projection_rule_value(
    visualization_spec: GameVisualizationSpec | None,
    key: str,
) -> Any:
    if visualization_spec is None:
        return None
    rules = visualization_spec.scene_projection_rules
    if not isinstance(rules, Mapping):
        return None
    return rules.get(key)


def _parse_square_board_text(board_text: Any) -> list[list[str]]:
    if not isinstance(board_text, str) or not board_text.strip():
        return []
    lines = [line.strip() for line in board_text.splitlines() if line.strip()]
    if not lines:
        return []
    header_tokens = lines[0].split()
    has_header = bool(header_tokens) and all(
        token.isalpha() or token.isdigit()
        for token in header_tokens
    )
    start_idx = 1 if has_header else 0
    rows: list[list[str]] = []
    for line in lines[start_idx:]:
        tokens = [token for token in line.split() if token]
        if not tokens:
            continue
        if tokens[0].isdigit():
            tokens = tokens[1:]
        rows.append(tokens)
    if not rows:
        return []
    row_len = len(rows[0])
    if row_len == 0:
        return []
    if any(len(row) != row_len for row in rows):
        return []
    if has_header and len(header_tokens) != row_len:
        return []
    if len(rows) != row_len:
        return []
    return rows


def _build_projected_grid(
    *,
    board_rows: Sequence[Sequence[str]],
    board_size: int,
) -> list[list[str]]:
    empty_row = ["." for _ in range(board_size)]
    if not board_rows:
        return [list(empty_row) for _ in range(board_size)]
    return [
        [str(item) for item in board_rows[row_idx]]
        if row_idx < len(board_rows)
        else list(empty_row)
        for row_idx in range(board_size)
    ]


def _build_coord_codec(*, board_size: int, coord_scheme: str) -> GomokuCoordCodec | None:
    if board_size <= 0:
        return None
    try:
        return GomokuCoordCodec(board_size=board_size, coord_scheme=coord_scheme)
    except Exception:
        return None


def _extract_board_legal_actions(
    payload: Mapping[str, Any],
    *,
    codec: GomokuCoordCodec | None,
) -> tuple[dict[str, Any], ...]:
    actions: list[dict[str, Any]] = []
    legal_moves = payload.get("legal_moves")
    if isinstance(legal_moves, Sequence) and not isinstance(legal_moves, (str, bytes)):
        for item in legal_moves:
            normalized = _normalize_board_legal_action(item, codec=codec)
            if normalized is not None:
                actions.append(normalized)
    if actions:
        return tuple(actions)

    for key in ("legalActions", "legal_actions"):
        value = payload.get(key)
        if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
            continue
        for item in value:
            normalized = _normalize_board_legal_action(item, codec=codec)
            if normalized is not None:
                actions.append(normalized)
        if actions:
            return tuple(actions)
    return ()


def _normalize_board_legal_action(
    item: Any,
    *,
    codec: GomokuCoordCodec | None,
) -> dict[str, Any] | None:
    if codec is None:
        return None

    if isinstance(item, str):
        coord = _normalize_coord(item)
        if coord is None:
            return None
        row_col = _coord_to_index(coord, codec=codec)
        if row_col is None:
            return None
        row, col = row_col
        return {
            "id": coord,
            "label": coord,
            "coord": coord,
            "row": row,
            "col": col,
        }

    if not isinstance(item, Mapping):
        return None

    coord = _normalize_coord(item.get("coord"))
    if coord is None:
        raw_id = item.get("id")
        if isinstance(raw_id, str):
            coord = _normalize_coord(raw_id)

    row = item.get("row")
    col = item.get("col")
    if coord is not None:
        row_col = _coord_to_index(coord, codec=codec)
        if row_col is None:
            return None
        row, col = row_col
    elif isinstance(row, int) and isinstance(col, int):
        try:
            coord = codec.index_to_coord(int(row), int(col))
        except Exception:
            return None
    else:
        return None

    action_id = item.get("id")
    if not isinstance(action_id, str) or not action_id.strip():
        action_id = coord
    label = item.get("label")
    if not isinstance(label, str) or not label.strip():
        label = coord
    return {
        "id": str(action_id),
        "label": str(label),
        "coord": coord,
        "row": int(row),
        "col": int(col),
    }


def _coord_to_index(coord: str, *, codec: GomokuCoordCodec) -> tuple[int, int] | None:
    try:
        return codec.coord_to_index(coord)
    except Exception:
        return None


def _extract_observer_player(payload: Mapping[str, Any]) -> str | None:
    for key in ("observerPlayerId", "observer_player_id"):
        value = _string_or_none(payload.get(key))
        if value is not None:
            return value
    return None


def _coerce_int(value: Any, *, default: int) -> int:
    value = _unwrap_scalar(value)
    if value is None:
        return default
    try:
        return int(value)
    except Exception:
        return default


def _coerce_optional_int(value: Any) -> int | None:
    value = _unwrap_scalar(value)
    if value is None:
        return None
    try:
        return int(value)
    except Exception:
        return None


def _normalize_coord(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    return text.upper()


def _normalize_coord_sequence(value: Any) -> list[str]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        return []
    coords: list[str] = []
    for item in value:
        coord = _normalize_coord(item)
        if coord is not None:
            coords.append(coord)
    return coords


def _normalize_player_ids(player_ids: Any, player_names: Any) -> list[str]:
    if isinstance(player_ids, Sequence) and not isinstance(player_ids, (str, bytes)):
        normalized = [text for item in player_ids if (text := _string_or_none(item)) is not None]
        if normalized:
            return normalized
    if isinstance(player_names, Mapping):
        return [str(key) for key in player_names.keys()]
    return []


def _normalize_player_names(player_names: Any, *, player_ids: Sequence[str]) -> dict[str, str]:
    if isinstance(player_names, Mapping):
        return {
            str(player_id): _string_or_none(player_names.get(player_id, player_id)) or str(player_id)
            for player_id in player_ids
        }
    if isinstance(player_names, Sequence) and not isinstance(player_names, (str, bytes)):
        names_by_id: dict[str, str] = {}
        for idx, player_id in enumerate(player_ids):
            if idx < len(player_names):
                names_by_id[player_id] = _string_or_none(player_names[idx]) or player_id
            else:
                names_by_id[player_id] = player_id
        return names_by_id
    return {str(player_id): str(player_id) for player_id in player_ids}


def _infer_table_player_ids(
    *,
    public_state: Mapping[str, Any],
    private_state: Mapping[str, Any],
    ui_state: Mapping[str, Any],
) -> list[str]:
    inferred: list[str] = []

    def _append(value: Any) -> None:
        text = _string_or_none(value)
        if text is None or text in inferred:
            return
        inferred.append(text)

    player_id_candidates = ui_state.get("player_ids")
    if isinstance(player_id_candidates, Sequence) and not isinstance(player_id_candidates, (str, bytes)):
        for item in player_id_candidates:
            _append(item)
        if inferred:
            return inferred

    for mapping in (
        ui_state.get("player_names"),
        ui_state.get("roles"),
        public_state.get("num_cards_left"),
        public_state.get("melds"),
    ):
        for key in _mapping_or_empty(mapping).keys():
            _append(key)

    for value in _mapping_or_empty(ui_state.get("seat_order")).values():
        _append(value)

    played_cards = public_state.get("played_cards")
    if isinstance(played_cards, Sequence) and not isinstance(played_cards, (str, bytes)):
        for item in played_cards:
            if isinstance(item, Mapping):
                _append(item.get("player_id") or item.get("playerId"))

    for item in _normalize_chat_log(public_state.get("chat_log")):
        _append(item.get("playerId"))

    _append(private_state.get("self_id"))
    return inferred


def _normalize_tokens(value: Any) -> list[str]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        return []
    return [str(item) for item in value if str(item).strip()]


def _normalize_board_occupant(value: Any) -> str | None:
    text = str(value or "").strip()
    if text in {"", ".", "-", "_"}:
        return None
    return text


def _infer_board_size_from_coords(
    legal_moves: Any,
    winning_line: Any,
    last_move: Any,
    *,
    coord_scheme: str,
) -> int:
    candidates: list[str] = []
    if isinstance(legal_moves, Sequence) and not isinstance(legal_moves, (str, bytes)):
        for item in legal_moves:
            if isinstance(item, str):
                candidates.append(item)
            elif isinstance(item, Mapping):
                if isinstance(item.get("coord"), str):
                    candidates.append(str(item["coord"]))
                elif isinstance(item.get("id"), str):
                    candidates.append(str(item["id"]))
    if isinstance(winning_line, Sequence) and not isinstance(winning_line, (str, bytes)):
        candidates.extend(str(item) for item in winning_line if item is not None)
    if isinstance(last_move, str):
        candidates.append(last_move)
    if not candidates:
        return 0

    for size in range(1, 26 + 1):
        codec = _build_coord_codec(board_size=size, coord_scheme=coord_scheme)
        if codec is None:
            continue
        if all(_coord_to_index(coord.strip().upper(), codec=codec) is not None for coord in candidates):
            return size
    return 0


def _project_table_scene(
    *,
    visual_session: VisualSession,
    snapshot_body: Any,
    event_payload: Any,
    visualization_spec: GameVisualizationSpec | None,
) -> tuple[dict[str, Any], tuple[dict[str, Any], ...], str | None]:
    source = _build_projection_source(snapshot_body, event_payload)
    result_payloads = _collect_result_payloads(event_payload, snapshot_body)
    _apply_table_result_overrides(
        source,
        result_payloads=result_payloads,
    )
    public_state = _mapping_or_empty(source.get("public_state"))
    private_state = _mapping_or_empty(source.get("private_state"))
    ui_state = _mapping_or_empty(source.get("ui_state"))
    player_ids = _normalize_player_ids(source.get("player_ids"), source.get("player_names"))
    if not player_ids:
        player_ids = _infer_table_player_ids(
            public_state=public_state,
            private_state=private_state,
            ui_state=ui_state,
        )
    player_names_source = source.get("player_names")
    if player_names_source is None:
        player_names_source = ui_state.get("player_names")
    player_names = _normalize_player_names(player_names_source, player_ids=player_ids)
    active_player_id = _extract_active_player(source, public_state, ui_state)
    requested_observer_player_id = _resolve_table_observer_player(
        visual_session=visual_session,
    )
    private_view_player_id = _resolve_table_private_view_player(
        source=source,
        private_state=private_state,
    )
    observer_player_id = _resolve_projected_table_observer_player(
        requested_observer_player_id=requested_observer_player_id,
        private_view_player_id=private_view_player_id,
    )
    observer_kind = visual_session.observer.observer_kind
    chat_log_source = source.get("chat_log")
    if chat_log_source is None:
        chat_log_source = ui_state.get("chat_log")
    if chat_log_source is None:
        chat_log_source = public_state.get("chat_log")
    chat_log = _normalize_chat_log(chat_log_source)
    raw_move_count = source.get("move_count")
    if raw_move_count is None:
        raw_move_count = ui_state.get("move_count")
    move_count = _coerce_int(raw_move_count, default=0)
    last_move = source.get("last_move")
    landlord_id = public_state.get("landlord_id")
    if landlord_id is None:
        landlord_id = ui_state.get("landlord_id")
    legal_actions = _extract_table_legal_actions(source)
    table_game = str(
        _scene_projection_rule_value(visualization_spec, "table_game")
        or source.get("game_id")
        or visual_session.game_id
    )
    if table_game == "doudizhu":
        _apply_doudizhu_result_move_overrides(
            source,
            player_ids=player_ids,
            result_payloads=result_payloads,
        )
        public_state = _mapping_or_empty(source.get("public_state"))
        private_state = _mapping_or_empty(source.get("private_state"))

    if table_game == "doudizhu":
        table_payload = _project_doudizhu_table(
            player_ids=player_ids,
            player_names=player_names,
            public_state=public_state,
            private_state=private_state,
            ui_state=ui_state,
            active_player_id=active_player_id,
            observer_player_id=observer_player_id,
            observer_kind=observer_kind,
            visualization_spec=visualization_spec,
        )
    else:
        landlord_id = None
        table_payload = _project_mahjong_table(
            player_ids=player_ids,
            player_names=player_names,
            public_state=public_state,
            private_state=private_state,
            active_player_id=active_player_id,
            observer_player_id=observer_player_id,
            private_view_player_id=private_view_player_id,
            observer_kind=observer_kind,
            visualization_spec=visualization_spec,
        )

    panels: dict[str, Any] = {
        "chatLog": chat_log,
    }
    if table_game == "doudizhu":
        move_history = ui_state.get("move_history")
        if not isinstance(move_history, Sequence) or isinstance(move_history, (str, bytes)):
            move_history = public_state.get("trace")
        panels.update(
            _build_doudizhu_panel_payload(
                player_ids=player_ids,
                player_names=player_names,
                move_history=move_history,
                active_player_id=active_player_id,
                landlord_id=_string_or_none(landlord_id),
                move_count=move_count,
                last_move=_string_or_none(last_move),
            )
        )
    else:
        mahjong_result = _extract_mahjong_result_summary(
            source=source,
            public_state=public_state,
            result_payloads=result_payloads,
        )
        panels.update(
            _build_mahjong_panel_payload(
                player_ids=player_ids,
                player_names=player_names,
                public_state=public_state,
                active_player_id=active_player_id,
                move_count=move_count,
                mahjong_result=mahjong_result,
            )
        )

    body: dict[str, Any] = {
        "table": table_payload,
        "status": {
            "activePlayerId": active_player_id,
            "observerPlayerId": observer_player_id,
            "privateViewPlayerId": private_view_player_id,
            "moveCount": move_count,
            "lastMove": None if last_move is None else str(last_move),
            "landlordId": None if landlord_id is None else str(landlord_id),
        },
        "panels": panels,
    }
    if table_game == "mahjong":
        last_discard = _normalize_mahjong_last_discard(public_state.get("last_discard"))
        if last_discard is not None:
            body["status"]["lastDiscard"] = last_discard
        mahjong_result = _extract_mahjong_result_summary(
            source=source,
            public_state=public_state,
            result_payloads=result_payloads,
        )
        body["status"]["winner"] = mahjong_result["winner"]
        body["status"]["result"] = mahjong_result["result"]
        body["status"]["resultReason"] = mahjong_result["resultReason"]
        if mahjong_result["remainingTiles"] is not None:
            body["status"]["remainingTiles"] = mahjong_result["remainingTiles"]
    return body, legal_actions, active_player_id


def _apply_table_result_overrides(
    source: dict[str, Any],
    *,
    result_payloads: Sequence[Mapping[str, Any]],
) -> None:
    if not result_payloads:
        return

    final_board = _first_result_string(result_payloads, "final_board", "finalBoard")
    if final_board is not None:
        source["board_text"] = final_board
        parsed_state = _parse_structured_table_board_text(final_board)
        for key in ("public_state", "private_state", "ui_state", "chat_log", "legal_moves"):
            if key in parsed_state:
                source[key] = parsed_state[key]

    raw_move_count = _first_result_value(result_payloads, "move_count", "moveCount")
    if raw_move_count is not None:
        source["move_count"] = _coerce_int(raw_move_count, default=0)

    last_move = None
    for result_payload in result_payloads:
        last_move = _extract_result_table_last_move(result_payload)
        if last_move is not None:
            break
    if last_move is not None:
        source["last_move"] = last_move


def _parse_structured_table_board_text(board_text: str) -> dict[str, Any]:
    parsed: dict[str, Any] = {}
    lines = [line.rstrip() for line in board_text.splitlines()]
    index = 0
    section_keys = {
        "Public State:": "public_state",
        "Private State:": "private_state",
        "Chat Log:": "chat_log",
        "UI_STATE_JSON:": "ui_state",
    }

    while index < len(lines):
        line = lines[index].strip()
        target_key = section_keys.get(line)
        if target_key is not None:
            index += 1
            while index < len(lines) and not lines[index].strip():
                index += 1
            if index < len(lines):
                try:
                    parsed[target_key] = json.loads(lines[index].strip())
                except Exception:
                    pass
        elif line.startswith("Legal Moves (preview):"):
            preview = line.split(":", 1)[1].strip()
            if preview.lower() == "none" or preview == "":
                parsed["legal_moves"] = []
            else:
                parsed["legal_moves"] = [
                    item.strip()
                    for item in preview.split(",")
                    if item.strip()
                ]
        index += 1

    return parsed


def _extract_result_table_last_move(result_payload: Mapping[str, Any]) -> str | None:
    for key in ("last_move", "lastMove"):
        value = _string_or_none(result_payload.get(key))
        if value is not None:
            return value

    move_log = result_payload.get("move_log")
    if move_log is None:
        move_log = result_payload.get("moveLog")
    if isinstance(move_log, Sequence) and not isinstance(move_log, (str, bytes)):
        for entry in reversed(move_log):
            if not isinstance(entry, Mapping):
                continue
            if "__truncated__" in entry:
                return None
            for key in ("action_card", "action_text", "move", "action"):
                value = _string_or_none(entry.get(key))
                if value is not None:
                    return value
    return None


def _apply_doudizhu_result_move_overrides(
    source: dict[str, Any],
    *,
    player_ids: Sequence[str],
    result_payloads: Sequence[Mapping[str, Any]],
) -> None:
    public_state = _mapping_or_empty(source.get("public_state"))
    private_state = _mapping_or_empty(source.get("private_state"))
    move_log = _extract_result_move_log(result_payloads)
    if not move_log or not public_state:
        return

    existing_trace = public_state.get("trace")
    applied_count = len(existing_trace) if isinstance(existing_trace, Sequence) and not isinstance(existing_trace, (str, bytes)) else 0
    pending_entries = move_log[applied_count:]
    if not pending_entries:
        return

    normalized_player_ids = list(player_ids)
    if not normalized_player_ids:
        normalized_player_ids = _normalize_player_ids(source.get("player_ids"), source.get("player_names"))
    player_index_lookup = {
        player_id: index
        for index, player_id in enumerate(normalized_player_ids)
    }
    card_counts = {
        str(player_id): _coerce_int(count, default=0)
        for player_id, count in _mapping_or_empty(public_state.get("num_cards_left")).items()
    }
    played_cards = _normalize_played_cards(public_state.get("played_cards"))
    seen_cards = _normalize_string_list(public_state.get("seen_cards"))
    trace_entries = list(existing_trace) if isinstance(existing_trace, Sequence) and not isinstance(existing_trace, (str, bytes)) else []
    private_self_id = _string_or_none(private_state.get("self_id"))
    private_hand = _normalize_string_list(private_state.get("current_hand"))

    for entry in pending_entries:
        player_id = _string_or_none(entry.get("player_id") or entry.get("playerId"))
        action_text = _string_or_none(
            entry.get("action_text")
            or entry.get("actionText")
            or entry.get("move")
            or entry.get("action")
        )
        if player_id is None or action_text is None:
            continue

        trace_entry: dict[str, Any] = {"player_id": player_id, "move": action_text}
        player_index = player_index_lookup.get(player_id)
        if player_index is not None:
            trace_entry["player_idx"] = player_index
        trace_entries.append(trace_entry)

        if action_text.lower() == "pass":
            continue

        cards = _split_doudizhu_action_cards(action_text)
        played_cards[player_id] = cards
        seen_cards.extend(cards)
        if player_id in card_counts:
            card_counts[player_id] = max(0, card_counts[player_id] - len(cards))
        if private_self_id == player_id and private_hand:
            private_hand = _remove_doudizhu_cards_from_hand(private_hand, cards)

    public_state["played_cards"] = [
        {
            "player_id": player_id,
            "cards": played_cards.get(player_id, []),
        }
        for player_id in normalized_player_ids
    ]
    public_state["seen_cards"] = seen_cards
    public_state["trace"] = trace_entries
    if card_counts:
        public_state["num_cards_left"] = card_counts
    if private_self_id is not None:
        private_state["current_hand"] = private_hand
        private_state["current_hand_text"] = ", ".join(private_hand)

    source["public_state"] = public_state
    source["private_state"] = private_state


def _extract_result_move_log(
    result_payloads: Sequence[Mapping[str, Any]],
) -> list[Mapping[str, Any]]:
    for result_payload in result_payloads:
        move_log = result_payload.get("move_log")
        if move_log is None:
            move_log = result_payload.get("moveLog")
        if isinstance(move_log, Sequence) and not isinstance(move_log, (str, bytes)):
            normalized = [
                entry
                for entry in move_log
                if isinstance(entry, Mapping) and "__truncated__" not in entry
            ]
            if normalized:
                return normalized
    return []


def _split_doudizhu_action_cards(action_text: str) -> list[str]:
    normalized = str(action_text).strip()
    if not normalized or normalized.lower() == "pass":
        return []
    cards: list[str] = []
    index = 0
    while index < len(normalized):
        token = normalized[index]
        if token == "1" and index + 1 < len(normalized) and normalized[index + 1] == "0":
            cards.append("10")
            index += 2
            continue
        cards.append(token.upper())
        index += 1
    return cards


def _remove_doudizhu_cards_from_hand(hand: Sequence[str], cards: Sequence[str]) -> list[str]:
    remaining = [str(card) for card in hand]
    for card in cards:
        target = str(card).upper()
        match_index = next(
            (
                index
                for index, hand_card in enumerate(remaining)
                if hand_card.upper() == target or hand_card.upper().endswith(target)
            ),
            None,
        )
        if match_index is not None:
            remaining.pop(match_index)
    return remaining


def _project_frame_scene(
    *,
    visual_session: VisualSession,
    snapshot_body: Any,
    event_payload: Any,
    visualization_spec: GameVisualizationSpec | None,
) -> tuple[dict[str, Any], tuple[dict[str, Any], ...], str | None, tuple[dict[str, Any], ...]]:
    source = _build_projection_source(snapshot_body, event_payload)
    observation = _mapping_or_empty(source.get("observation"))
    view = _mapping_or_empty(source.get("view"))
    if not view:
        view = _mapping_or_empty(observation.get("view"))
    context = _mapping_or_empty(source.get("context"))
    if not context:
        context = _mapping_or_empty(observation.get("context"))
    metadata = _mapping_or_empty(source.get("metadata"))
    if not metadata:
        metadata = _mapping_or_empty(observation.get("metadata"))

    active_player_id = _extract_active_player(source, observation, context, metadata)
    observer_player_id = _resolve_frame_observer_player(
        visual_session=visual_session,
        source=source,
    )
    tick = _coerce_int(source.get("tick") or context.get("tick"), default=0)
    step = _coerce_int(source.get("step") or context.get("step"), default=tick)
    move_count = _coerce_int(source.get("move_count"), default=step)
    last_move = _string_or_none(source.get("last_move") or observation.get("last_move") or metadata.get("last_move"))
    reward = _coerce_float_or_none(source.get("reward"))
    if reward is None:
        reward = _coerce_float_or_none(metadata.get("reward"))
    stream_id = _string_or_none(source.get("stream_id") or metadata.get("stream_id"))
    if stream_id is None:
        stream_id = _string_or_none(_scene_projection_rule_value(visualization_spec, "default_stream_id"))
    frame_title = (
        _string_or_none(_scene_projection_rule_value(visualization_spec, "frame_title"))
        or f"{visual_session.game_id} Frame"
    )
    frame_fit = (
        _string_or_none(_scene_projection_rule_value(visualization_spec, "default_fit"))
        or "contain"
    )
    viewport = _extract_frame_viewport(source, observation, view=view)
    view_text = _string_or_none(view.get("text") or observation.get("board_text") or source.get("board_text"))
    legal_actions = _extract_frame_legal_actions(source, observation, metadata)
    subtitle = _resolve_frame_subtitle(
        game_id=visual_session.game_id,
        tick=tick,
        stream_id=stream_id,
    )

    body: dict[str, Any] = {
        "frame": {
            "title": frame_title,
            "subtitle": subtitle,
            "altText": frame_title.replace(" Frame", " frame"),
            "streamId": stream_id,
            "fit": frame_fit,
            "viewport": viewport,
        },
        "status": {
            "activePlayerId": active_player_id,
            "observerPlayerId": observer_player_id,
            "tick": tick,
            "step": step,
            "moveCount": move_count,
            "lastMove": last_move,
            "reward": reward,
        },
        "view": {
            "text": view_text,
        },
        "snapshot": _sanitize_frame_snapshot(snapshot_body),
    }
    overlays = _build_frame_overlays(
        tick=tick,
        reward=reward,
        last_move=last_move,
        stream_id=stream_id,
        game_id=visual_session.game_id,
    )
    return body, legal_actions, active_player_id, overlays


def _project_rts_scene(
    *,
    visual_session: VisualSession,
    snapshot_body: Any,
    event_payload: Any,
    visualization_spec: GameVisualizationSpec | None,
) -> tuple[dict[str, Any], tuple[dict[str, Any], ...], str | None, tuple[dict[str, Any], ...]]:
    source = _build_projection_source(snapshot_body, event_payload)
    observation = _mapping_or_empty(source.get("observation"))
    view = _mapping_or_empty(source.get("view"))
    if not view:
        view = _mapping_or_empty(observation.get("view"))
    context = _mapping_or_empty(source.get("context"))
    if not context:
        context = _mapping_or_empty(observation.get("context"))
    metadata = _mapping_or_empty(source.get("metadata"))
    if not metadata:
        metadata = _mapping_or_empty(observation.get("metadata"))

    active_player_id = _extract_active_player(source, observation, context, metadata)
    observer_player_id = _resolve_frame_observer_player(
        visual_session=visual_session,
        source=source,
    )
    tick = _coerce_int(source.get("tick") or context.get("tick"), default=0)
    step = _coerce_int(source.get("step") or context.get("step"), default=tick)
    move_count = _coerce_int(source.get("move_count"), default=step)
    last_move = _string_or_none(
        source.get("last_move") or observation.get("last_move") or metadata.get("last_move")
    )
    reward = _coerce_float_or_none(source.get("reward"))
    if reward is None:
        reward = _coerce_float_or_none(metadata.get("reward"))
    stream_id = _string_or_none(source.get("stream_id") or metadata.get("stream_id"))
    if stream_id is None:
        stream_id = _string_or_none(
            _scene_projection_rule_value(visualization_spec, "default_stream_id")
        )
    frame_title = (
        _string_or_none(_scene_projection_rule_value(visualization_spec, "frame_title"))
        or f"{visual_session.game_id} RTS"
    )
    frame_fit = (
        _string_or_none(_scene_projection_rule_value(visualization_spec, "default_fit"))
        or "contain"
    )
    viewport = _extract_frame_viewport(source, observation, view=view)
    view_text = _string_or_none(
        view.get("text") or observation.get("board_text") or source.get("board_text")
    )
    legal_actions = _extract_frame_legal_actions(source, observation, metadata)
    map_id = _string_or_none(
        metadata.get("map_id") or metadata.get("env_id") or source.get("env_id")
    )
    map_payload = _project_rts_map(metadata.get("map"))
    selection_payload = _project_rts_selection(metadata.get("selection"))
    economy_payload = _project_rts_economy(metadata.get("economy"))
    objectives_payload = _project_rts_objectives(metadata.get("objectives"))
    units_payload = _project_rts_units(metadata.get("units"))
    production_payload = _project_rts_production(metadata.get("production"))
    map_title = _string_or_none(map_payload.get("title")) or map_id

    body: dict[str, Any] = {
        "frame": {
            "title": frame_title,
            "subtitle": f"Map {map_title}" if map_title is not None else None,
            "altText": frame_title,
            "streamId": stream_id,
            "fit": frame_fit,
            "viewport": viewport,
        },
        "status": {
            "activePlayerId": active_player_id,
            "observerPlayerId": observer_player_id,
            "tick": tick,
            "step": step,
            "moveCount": move_count,
            "lastMove": last_move,
            "reward": reward,
        },
        "view": {
            "text": view_text,
        },
        "rts": {
            "map": map_payload,
            "selection": selection_payload,
            "economy": economy_payload,
            "objectives": objectives_payload,
            "units": units_payload,
            "production": production_payload,
        },
        "snapshot": _sanitize_frame_snapshot(snapshot_body),
    }
    overlays = _build_rts_overlays(
        tick=tick,
        last_move=last_move,
        economy=economy_payload,
    )
    return body, legal_actions, active_player_id, overlays


def _project_doudizhu_table(
    *,
    player_ids: Sequence[str],
    player_names: Mapping[str, str],
    public_state: Mapping[str, Any],
    private_state: Mapping[str, Any],
    ui_state: Mapping[str, Any],
    active_player_id: str | None,
    observer_player_id: str | None,
    observer_kind: str,
    visualization_spec: GameVisualizationSpec | None,
) -> dict[str, Any]:
    roles = _mapping_or_empty(ui_state.get("roles"))
    if not roles:
        landlord_id = _string_or_none(public_state.get("landlord_id"))
        roles = {
            player_id: "landlord" if landlord_id and player_id == landlord_id else "peasant"
            for player_id in player_ids
        }
    seat_order = _mapping_or_empty(ui_state.get("seat_order"))
    seat_ids = _invert_seat_order(seat_order, player_ids=player_ids)
    card_counts = {
        str(player_id): _coerce_int(count, default=0)
        for player_id, count in _mapping_or_empty(public_state.get("num_cards_left")).items()
    }
    played_cards = _normalize_played_cards(public_state.get("played_cards"))
    suitful_hands = _normalize_doudizhu_hands(ui_state.get("hands"), player_ids=player_ids)
    latest_actions = _normalize_doudizhu_latest_actions(
        ui_state.get("latest_actions"),
        player_ids=player_ids,
    )
    private_self_id = _string_or_none(private_state.get("self_id"))
    private_hand = _normalize_string_list(private_state.get("current_hand"))
    seats: list[dict[str, Any]] = []
    for player_id in player_ids:
        is_visible = _table_hand_visible(
            observer_kind=observer_kind,
            observer_player_id=observer_player_id,
            player_id=player_id,
        )
        visible_hand = suitful_hands.get(player_id, [])
        if player_id == private_self_id and private_hand:
            resolved_hand = visible_hand or private_hand
            cards = resolved_hand if is_visible else []
            masked_count = 0 if is_visible else len(resolved_hand)
        else:
            cards = visible_hand if is_visible else []
            masked_count = 0 if is_visible else card_counts.get(player_id, 0)
        seats.append(
            {
                "seatId": seat_ids.get(player_id, player_id),
                "playerId": player_id,
                "playerName": player_names.get(player_id, player_id),
                "role": _string_or_none(roles.get(player_id)),
                "isActive": bool(active_player_id and player_id == active_player_id),
                "isObserver": bool(observer_player_id and player_id == observer_player_id),
                "playedCards": latest_actions.get(player_id, played_cards.get(player_id, [])),
                "publicNotes": [],
                "hand": {
                    "isVisible": is_visible,
                    "cards": cards,
                    "maskedCount": masked_count,
                },
            }
        )

    center_cards = _normalize_doudizhu_card_collection(ui_state.get("seen_cards"))
    if not center_cards:
        center_cards = _normalize_string_list(public_state.get("seen_cards"))
    move_history = ui_state.get("move_history")
    if not isinstance(move_history, Sequence) or isinstance(move_history, (str, bytes)):
        move_history = public_state.get("trace")
    center_history = _format_doudizhu_history(move_history, player_ids=player_ids)
    layout = _string_or_none(_scene_projection_rule_value(visualization_spec, "default_layout")) or "three-seat"
    return {
        "layout": layout,
        "seats": seats,
        "center": {
            "label": "Seen cards",
            "cards": center_cards,
            "history": center_history,
        },
    }


def _project_mahjong_table(
    *,
    player_ids: Sequence[str],
    player_names: Mapping[str, str],
    public_state: Mapping[str, Any],
    private_state: Mapping[str, Any],
    active_player_id: str | None,
    observer_player_id: str | None,
    private_view_player_id: str | None,
    observer_kind: str,
    visualization_spec: GameVisualizationSpec | None,
) -> dict[str, Any]:
    private_hand = _normalize_string_list(private_state.get("hand"))
    private_draw_tile = _string_or_none(private_state.get("draw_tile"))
    card_counts = {
        str(player_id): _coerce_int(count, default=0)
        for player_id, count in _mapping_or_empty(public_state.get("num_cards_left")).items()
    }
    melds = _mapping_or_empty(public_state.get("melds"))
    meld_groups = _normalize_mahjong_meld_groups(public_state.get("meld_groups"))
    discard_lanes = _normalize_mahjong_discard_lanes(public_state.get("discard_lanes"), player_ids=player_ids)
    center_history = _build_mahjong_history(
        player_ids=player_ids,
        player_names=player_names,
        discard_lanes=discard_lanes,
        meld_groups=meld_groups,
        melds=melds,
    )
    seats: list[dict[str, Any]] = []
    for player_id in player_ids:
        is_visible = _table_hand_visible(
            observer_kind=observer_kind,
            observer_player_id=observer_player_id,
            player_id=player_id,
        )
        masked_count = card_counts.get(player_id)
        if masked_count is None and player_id == private_view_player_id and private_hand:
            masked_count = len(private_hand)
        if masked_count is None:
            masked_count = 0
        public_notes = _normalize_string_list(melds.get(player_id))
        seats.append(
            {
                "seatId": player_id,
                "playerId": player_id,
                "playerName": player_names.get(player_id, player_id),
                "role": None,
                "isActive": bool(active_player_id and player_id == active_player_id),
                "isObserver": bool(observer_player_id and player_id == observer_player_id),
                "playedCards": [],
                "publicNotes": public_notes,
                "meldGroups": meld_groups.get(player_id, []),
                "drawTile": private_draw_tile if is_visible else None,
                "hand": {
                    "isVisible": is_visible,
                    "cards": private_hand if is_visible else [],
                    "maskedCount": 0 if is_visible else masked_count,
                    "drawTile": private_draw_tile if is_visible else None,
                },
            }
        )

    layout = _string_or_none(_scene_projection_rule_value(visualization_spec, "default_layout")) or "four-seat"
    center_payload: dict[str, Any] = {
        "label": "Discards",
        "cards": _normalize_string_list(public_state.get("discards")),
        "history": center_history,
    }
    if discard_lanes:
        center_payload["discardLanes"] = discard_lanes

    return {
        "layout": layout,
        "seats": seats,
        "center": center_payload,
    }


def _normalize_mahjong_meld_groups(value: Any) -> dict[str, list[dict[str, Any]]]:
    if not isinstance(value, Mapping):
        return {}

    normalized: dict[str, list[dict[str, Any]]] = {}
    for raw_player_id, raw_groups in value.items():
        player_id = str(raw_player_id)
        groups: list[dict[str, Any]] = []
        if isinstance(raw_groups, Sequence) and not isinstance(raw_groups, (str, bytes)):
            for raw_group in raw_groups:
                if not isinstance(raw_group, Mapping):
                    continue
                tiles = _normalize_string_list(raw_group.get("tiles"))
                label = _string_or_none(raw_group.get("label")) or ("-".join(tiles) if tiles else "")
                if not label and not tiles:
                    continue
                groups.append(
                    {
                        "type": _string_or_none(raw_group.get("type")),
                        "label": label,
                        "tiles": tiles,
                    }
                )
        if groups:
            normalized[player_id] = groups
    return normalized


def _normalize_mahjong_discard_lanes(
    value: Any,
    *,
    player_ids: Sequence[str],
) -> list[dict[str, Any]]:
    if not isinstance(value, Mapping):
        return []

    lanes: list[dict[str, Any]] = []
    for player_id in player_ids:
        cards = _normalize_string_list(value.get(player_id))
        lanes.append(
            {
                "seatId": player_id,
                "playerId": player_id,
                "cards": cards,
            }
        )
    return lanes


def _normalize_mahjong_last_discard(value: Any) -> dict[str, Any] | None:
    if not isinstance(value, Mapping):
        return None

    tile = _string_or_none(value.get("tile"))
    player_id = _string_or_none(value.get("player_id")) or _string_or_none(value.get("playerId"))
    if tile is None and player_id is None:
        return None

    return {
        "playerId": player_id,
        "tile": tile,
        "isTsumogiri": bool(value.get("is_tsumogiri") or value.get("isTsumogiri")),
    }


def _build_mahjong_history(
    *,
    player_ids: Sequence[str],
    player_names: Mapping[str, str],
    discard_lanes: Sequence[Mapping[str, Any]],
    meld_groups: Mapping[str, Sequence[Mapping[str, Any]]],
    melds: Mapping[str, Any],
) -> list[str]:
    history: list[str] = []
    if discard_lanes:
        ordered_lanes: list[tuple[str, list[str]]] = []
        for lane in discard_lanes:
            if not isinstance(lane, Mapping):
                continue
            player_id = _string_or_none(lane.get("playerId")) or _string_or_none(lane.get("seatId"))
            if player_id is None:
                continue
            ordered_lanes.append((player_id, _normalize_string_list(lane.get("cards"))))
        max_discards = max((len(cards) for _, cards in ordered_lanes), default=0)
        for discard_index in range(max_discards):
            for player_id, cards in ordered_lanes:
                if discard_index >= len(cards):
                    continue
                player_name = _resolve_table_player_label(player_id, player_names=player_names)
                history.append(f"{player_name} discarded {cards[discard_index]}")

    appended_meld_players: set[str] = set()
    for player_id in player_ids:
        groups = meld_groups.get(player_id)
        if isinstance(groups, Sequence) and not isinstance(groups, (str, bytes)):
            player_name = _resolve_table_player_label(player_id, player_names=player_names)
            for group in groups:
                if not isinstance(group, Mapping):
                    continue
                label = _string_or_none(group.get("label"))
                if label is None:
                    continue
                history.append(f"{player_name} melded {label}")
            appended_meld_players.add(player_id)

    for player_id in player_ids:
        if player_id in appended_meld_players:
            continue
        player_name = _resolve_table_player_label(player_id, player_names=player_names)
        for label in _normalize_string_list(melds.get(player_id)):
            history.append(f"{player_name} melded {label}")
    return history


def _extract_mahjong_result_summary(
    *,
    source: Mapping[str, Any],
    public_state: Mapping[str, Any],
    result_payloads: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    source_result = source.get("result")
    source_result_text = (
        None
        if isinstance(source_result, Mapping)
        else _string_or_none(source_result)
    )
    remaining_tiles = _coerce_optional_int(
        source.get("remaining_tiles")
        if "remaining_tiles" in source
        else source.get("remainingTiles")
    )
    if remaining_tiles is None:
        remaining_tiles = _coerce_optional_int(
            public_state.get("remaining_tiles")
            if "remaining_tiles" in public_state
            else public_state.get("remainingTiles")
        )
    if remaining_tiles is None:
        remaining_tiles = _coerce_optional_int(
            _first_result_value(
                result_payloads,
                "remaining_tiles",
                "remainingTiles",
            )
        )

    return {
        "winner": (
            _string_or_none(source.get("winner"))
            or _first_result_string(
                result_payloads,
                "winner",
                "winner_player_id",
                "winnerPlayerId",
            )
        ),
        "result": (
            source_result_text
            or _first_result_string(result_payloads, "result")
        ),
        "resultReason": (
            _string_or_none(source.get("result_reason"))
            or _string_or_none(source.get("resultReason"))
            or _first_result_string(result_payloads, "result_reason", "resultReason")
        ),
        "remainingTiles": remaining_tiles,
    }


def _build_mahjong_panel_payload(
    *,
    player_ids: Sequence[str],
    player_names: Mapping[str, str],
    public_state: Mapping[str, Any],
    active_player_id: str | None,
    move_count: int,
    mahjong_result: Mapping[str, Any],
) -> dict[str, Any]:
    discard_lanes = _normalize_mahjong_discard_lanes(public_state.get("discard_lanes"), player_ids=player_ids)
    meld_groups = _normalize_mahjong_meld_groups(public_state.get("meld_groups"))
    melds = _mapping_or_empty(public_state.get("melds"))
    trace = _build_mahjong_history(
        player_ids=player_ids,
        player_names=player_names,
        discard_lanes=discard_lanes,
        meld_groups=meld_groups,
        melds=melds,
    )
    last_discard = _normalize_mahjong_last_discard(public_state.get("last_discard"))

    events: list[dict[str, str]] = []
    if active_player_id is not None:
        active_player = _resolve_table_player_label(active_player_id, player_names=player_names)
        events.append({"label": "Turn", "detail": f"{active_player} to act"})
    if last_discard is not None:
        discard_player = _resolve_table_player_label(
            _string_or_none(last_discard.get("playerId")),
            player_names=player_names,
        )
        discard_tile = _string_or_none(last_discard.get("tile"))
        if discard_tile is not None:
            events.append(
                {
                    "label": "Last discard",
                    "detail": f"{discard_player} discarded {discard_tile}",
                }
            )

    first_open_meld: tuple[str, str] | None = None
    for player_id in player_ids:
        groups = meld_groups.get(player_id)
        labels = [
            _string_or_none(group.get("label"))
            for group in groups
            if isinstance(group, Mapping)
        ] if isinstance(groups, Sequence) and not isinstance(groups, (str, bytes)) else []
        labels = [label for label in labels if label is not None]
        if not labels:
            labels = _normalize_string_list(melds.get(player_id))
        if labels:
            first_open_meld = (
                _resolve_table_player_label(player_id, player_names=player_names),
                labels[0],
            )
            break
    if first_open_meld is not None:
        player_name, meld_label = first_open_meld
        events.append({"label": "Open meld", "detail": f"{player_name}: {meld_label}"})

    winner = _string_or_none(mahjong_result.get("winner"))
    if winner is not None:
        winner_label = _resolve_table_player_label(winner, player_names=player_names)
        events.append({"label": "Winner", "detail": winner_label})
    result = _string_or_none(mahjong_result.get("result"))
    if result is not None:
        events.append({"label": "Result", "detail": result})
    result_reason = _string_or_none(mahjong_result.get("resultReason"))
    if result_reason is not None:
        events.append({"label": "Result reason", "detail": result_reason})
    remaining_tiles = _coerce_optional_int(mahjong_result.get("remainingTiles"))
    if remaining_tiles is not None:
        events.append(
            {
                "label": "Remaining tiles",
                "detail": f"{remaining_tiles} tiles in wall",
            }
        )
    if move_count > 0:
        events.append({"label": "Move count", "detail": f"{move_count} turns recorded"})

    return {
        "events": events,
        "trace": trace,
    }


def _mapping_or_empty(value: Any) -> dict[str, Any]:
    if isinstance(value, Mapping):
        return dict(value)
    return {}


def _string_or_none(value: Any) -> str | None:
    value = _unwrap_scalar(value)
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _unwrap_scalar(value: Any) -> Any:
    if isinstance(value, Mapping):
        kind = value.get("kind")
        if isinstance(kind, str) and "value" in value:
            return value.get("value")
    return value


def _normalize_string_list(value: Any) -> list[str]:
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        normalized: list[str] = []
        for item in value:
            text = _string_or_none(item)
            if text is not None:
                normalized.append(text)
        return normalized
    text = _string_or_none(value)
    return [text] if text is not None else []


def _normalize_chat_log(value: Any) -> list[dict[str, str]]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        return []
    chat_log: list[dict[str, str]] = []
    for item in value:
        if not isinstance(item, Mapping):
            continue
        player_id = _string_or_none(item.get("player_id") or item.get("playerId"))
        text = _string_or_none(item.get("text"))
        if player_id is None or text is None:
            continue
        chat_log.append(
            {
                "playerId": player_id,
                "text": text,
            }
        )
    return chat_log


def _coerce_float_or_none(value: Any) -> float | None:
    value = _unwrap_scalar(value)
    if value is None:
        return None
    try:
        return float(value)
    except Exception:
        return None


def _extract_frame_legal_actions(*payloads: Any) -> tuple[dict[str, Any], ...]:
    candidates: Any = None
    action_labels = _extract_frame_action_labels(*payloads)
    for payload in payloads:
        if not isinstance(payload, Mapping):
            continue
        legal_actions = payload.get("legal_actions")
        if isinstance(legal_actions, Mapping):
            items = legal_actions.get("items")
            if isinstance(items, Sequence) and not isinstance(items, (str, bytes)):
                candidates = items
                break
        legal_moves = payload.get("legal_moves")
        if isinstance(legal_moves, Sequence) and not isinstance(legal_moves, (str, bytes)):
            candidates = legal_moves
            break
    if candidates is None:
        return _extract_legal_actions(*payloads)

    actions: list[dict[str, Any]] = []
    for item in candidates:
        normalized = _normalize_frame_legal_action(item, action_labels=action_labels)
        if normalized is not None:
            actions.append(normalized)
    return tuple(actions)


def _extract_frame_action_labels(*payloads: Any) -> dict[str, str]:
    labels: dict[str, str] = {}
    for payload in payloads:
        if not isinstance(payload, Mapping):
            continue
        raw_mapping = payload.get("action_mapping")
        if raw_mapping is None:
            raw_mapping = payload.get("actionMapping")
        if raw_mapping is None:
            metadata = payload.get("metadata")
            if isinstance(metadata, Mapping):
                raw_mapping = metadata.get("action_mapping")
                if raw_mapping is None:
                    raw_mapping = metadata.get("actionMapping")
        if not isinstance(raw_mapping, Mapping):
            continue
        for raw_action_id, raw_label in raw_mapping.items():
            action_id = _string_or_none(raw_action_id)
            label = _string_or_none(raw_label)
            if action_id is None or label is None:
                continue
            labels[action_id] = _format_frame_action_label(label)
    return labels


def _normalize_frame_legal_action(
    item: Any,
    *,
    action_labels: Mapping[str, str] | None = None,
) -> dict[str, Any] | None:
    label_map = action_labels or {}
    if not isinstance(item, Mapping):
        text = _string_or_none(item)
        if text is None:
            return None
        label = label_map.get(text) or text
        return {
            "id": text,
            "label": label,
            "text": label,
        }
    action_id = _string_or_none(item.get("id")) or _string_or_none(item.get("text")) or _string_or_none(item.get("label"))
    if action_id is None:
        return None
    label = _string_or_none(item.get("label")) or label_map.get(action_id) or action_id
    text = _string_or_none(item.get("text")) or label
    normalized = {
        "id": action_id,
        "label": label,
        "text": text,
    }
    for key, value in item.items():
        if key in {"id", "label", "text"}:
            continue
        normalized[str(key)] = value
    return normalized


def _format_frame_action_label(value: str) -> str:
    text = str(value).strip()
    if not text:
        return ""
    aliases = {
        "ATTACK": "Fire",
        "NOOP": "No-op",
    }
    canonical = text.replace("-", "_").replace(" ", "_").upper()
    if canonical in aliases:
        return aliases[canonical]
    return text.replace("_", " ").replace("-", " ").strip().title()


def _resolve_frame_observer_player(
    *,
    visual_session: VisualSession,
    source: Mapping[str, Any],
) -> str | None:
    if (
        visual_session.observer.observer_kind == "player"
        and visual_session.observer.observer_id.strip()
    ):
        return visual_session.observer.observer_id
    return _extract_observer_player(source)


def _resolve_board_observer_player(
    *,
    visual_session: VisualSession,
    source: Mapping[str, Any],
) -> str | None:
    if (
        visual_session.observer.observer_kind == "player"
        and visual_session.observer.observer_id.strip()
    ):
        return visual_session.observer.observer_id
    return _extract_observer_player(source)


def _extract_frame_viewport(
    source: Mapping[str, Any],
    observation: Mapping[str, Any],
    *,
    view: Mapping[str, Any],
) -> dict[str, int] | None:
    for candidate in (
        source.get("viewport"),
        source.get("frame"),
        view.get("image"),
        observation.get("image"),
    ):
        viewport = _normalize_viewport(candidate)
        if viewport is not None:
            return viewport
    return None


def _project_rts_selection(value: Any) -> dict[str, Any]:
    payload = _mapping_or_empty(value)
    unit_ids = payload.get("unit_ids")
    if not isinstance(unit_ids, Sequence) or isinstance(unit_ids, (str, bytes)):
        unit_ids = payload.get("unitIds")
    resolved_unit_ids = (
        [str(item) for item in unit_ids if item is not None and str(item).strip()]
        if isinstance(unit_ids, Sequence) and not isinstance(unit_ids, (str, bytes))
        else []
    )
    primary_unit_id = _string_or_none(
        payload.get("primary_unit_id") or payload.get("primaryUnitId")
    )
    return {
        "unitIds": resolved_unit_ids,
        "primaryUnitId": primary_unit_id,
    }


def _project_rts_map(value: Any) -> dict[str, Any]:
    payload = _mapping_or_empty(value)
    grid_size = _mapping_or_empty(payload.get("map_size") or payload.get("gridSize"))
    bounds = _mapping_or_empty(payload.get("bounds"))
    image_size = _mapping_or_empty(payload.get("image_size") or payload.get("imageSize"))
    return {
        "id": _string_or_none(payload.get("id")),
        "modId": _string_or_none(payload.get("mod_id") or payload.get("modId")),
        "title": _string_or_none(payload.get("title")),
        "gridSize": {
            "width": _coerce_optional_int(grid_size.get("width")),
            "height": _coerce_optional_int(grid_size.get("height")),
        },
        "bounds": {
            "x": _coerce_optional_int(bounds.get("x")),
            "y": _coerce_optional_int(bounds.get("y")),
            "width": _coerce_optional_int(bounds.get("width")),
            "height": _coerce_optional_int(bounds.get("height")),
        },
        "imageSize": {
            "width": _coerce_optional_int(image_size.get("width")),
            "height": _coerce_optional_int(image_size.get("height")),
        },
        "previewSource": _string_or_none(
            payload.get("preview_source") or payload.get("previewSource")
        ),
    }


def _project_rts_economy(value: Any) -> dict[str, Any]:
    payload = _mapping_or_empty(value)
    power = _mapping_or_empty(payload.get("power"))
    return {
        "credits": _coerce_optional_int(payload.get("credits")),
        "incomePerMinute": _coerce_optional_int(
            payload.get("income_per_minute") or payload.get("incomePerMinute")
        ),
        "power": {
            "produced": _coerce_optional_int(power.get("produced")),
            "used": _coerce_optional_int(power.get("used")),
        },
    }


def _project_rts_objectives(value: Any) -> tuple[dict[str, Any], ...]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        return ()
    projected: list[dict[str, Any]] = []
    for index, item in enumerate(value):
        payload = _mapping_or_empty(item)
        projected.append(
            {
                "id": _string_or_none(payload.get("id")) or f"objective_{index + 1}",
                "label": _string_or_none(payload.get("label")) or f"Objective {index + 1}",
                "status": _string_or_none(payload.get("status")) or "unknown",
            }
        )
    return tuple(projected)


def _project_rts_units(value: Any) -> tuple[dict[str, Any], ...]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        return ()
    projected: list[dict[str, Any]] = []
    for index, item in enumerate(value):
        payload = _mapping_or_empty(item)
        position = _mapping_or_empty(payload.get("position"))
        projected.append(
            {
                "id": _string_or_none(payload.get("id")) or f"unit_{index + 1}",
                "owner": _string_or_none(payload.get("owner")),
                "label": _string_or_none(payload.get("label")) or f"Unit {index + 1}",
                "kind": _string_or_none(payload.get("kind")),
                "hp": _coerce_optional_int(payload.get("hp")),
                "status": _string_or_none(payload.get("status")),
                "position": {
                    "x": _coerce_optional_int(position.get("x")),
                    "y": _coerce_optional_int(position.get("y")),
                },
                "selected": bool(payload.get("selected")),
            }
        )
    return tuple(projected)


def _project_rts_production(value: Any) -> tuple[dict[str, Any], ...]:
    payload = _mapping_or_empty(value)
    queues = payload.get("queues")
    if not isinstance(queues, Sequence) or isinstance(queues, (str, bytes)):
        queues = value
    if not isinstance(queues, Sequence) or isinstance(queues, (str, bytes)):
        return ()
    projected: list[dict[str, Any]] = []
    for index, item in enumerate(queues):
        queue_payload = _mapping_or_empty(item)
        items_payload = queue_payload.get("items")
        if not isinstance(items_payload, Sequence) or isinstance(items_payload, (str, bytes)):
            items_payload = ()
        projected_items: list[dict[str, Any]] = []
        for item_index, raw_item in enumerate(items_payload):
            entry_payload = _mapping_or_empty(raw_item)
            projected_items.append(
                {
                    "id": _string_or_none(entry_payload.get("id")) or f"item_{item_index + 1}",
                    "label": _string_or_none(entry_payload.get("label"))
                    or f"Item {item_index + 1}",
                    "progress": _coerce_float_or_none(entry_payload.get("progress")),
                }
            )
        projected.append(
            {
                "buildingId": _string_or_none(
                    queue_payload.get("building_id") or queue_payload.get("buildingId")
                )
                or f"queue_{index + 1}",
                "label": _string_or_none(queue_payload.get("label")) or f"Queue {index + 1}",
                "items": tuple(projected_items),
            }
        )
    return tuple(projected)


def _normalize_viewport(value: Any) -> dict[str, int] | None:
    if isinstance(value, Mapping):
        width = _coerce_int(value.get("width"), default=0)
        height = _coerce_int(value.get("height"), default=0)
        if width > 0 and height > 0:
            return {
                "width": width,
                "height": height,
            }
        shape = value.get("shape")
        if isinstance(shape, Sequence) and not isinstance(shape, (str, bytes)) and len(shape) >= 2:
            try:
                height = int(_unwrap_scalar(shape[0]))
                width = int(_unwrap_scalar(shape[1]))
            except Exception:
                return None
            return {
                "width": width,
                "height": height,
            }
    return None


def _resolve_frame_subtitle(
    *,
    game_id: str,
    tick: int,
    stream_id: str | None,
) -> str | None:
    if game_id == "retro_platformer" and tick > 0:
        return f"Tick {tick}"
    if stream_id is not None:
        return f"Stream {stream_id}"
    if tick > 0:
        return f"Tick {tick}"
    return None


def _build_frame_overlays(
    *,
    tick: int,
    reward: float | None,
    last_move: str | None,
    stream_id: str | None,
    game_id: str,
) -> tuple[dict[str, Any], ...]:
    overlays: list[dict[str, Any]] = []
    if tick > 0:
        overlays.append({"kind": "badge", "label": "Tick", "value": str(tick)})
    if reward is not None:
        overlays.append({"kind": "badge", "label": "Reward", "value": _format_overlay_float(reward)})
    if last_move is not None:
        overlays.append({"kind": "badge", "label": "Last move", "value": last_move})
    if game_id == "vizdoom" and stream_id is not None:
        overlays.append({"kind": "badge", "label": "Stream", "value": stream_id})
    return tuple(overlays)


def _build_rts_overlays(
    *,
    tick: int,
    last_move: str | None,
    economy: Mapping[str, Any],
) -> tuple[dict[str, Any], ...]:
    overlays: list[dict[str, Any]] = []
    credits = _coerce_optional_int(economy.get("credits"))
    if credits is not None:
        overlays.append({"kind": "badge", "label": "Credits", "value": str(credits)})
    power = _mapping_or_empty(economy.get("power"))
    produced = _coerce_optional_int(power.get("produced"))
    used = _coerce_optional_int(power.get("used"))
    if produced is not None and used is not None:
        overlays.append(
            {
                "kind": "badge",
                "label": "Power",
                "value": f"{produced - used:+d}",
            }
        )
    if tick > 0:
        overlays.append({"kind": "badge", "label": "Tick", "value": str(tick)})
    if last_move is not None:
        overlays.append({"kind": "badge", "label": "Last move", "value": last_move})
    return tuple(overlays)


def _format_overlay_float(value: float) -> str:
    if float(value).is_integer():
        return f"{float(value):.1f}"
    return f"{float(value):.2f}".rstrip("0").rstrip(".")


def _sanitize_frame_snapshot(value: Any) -> Any:
    if not isinstance(value, Mapping):
        return value
    snapshot: dict[str, Any] = {}
    for key, item in value.items():
        if str(key) in {"_rgb", "rgb", "rgb_array", "frame_rgb"}:
            continue
        if isinstance(item, Mapping):
            snapshot[str(key)] = _sanitize_frame_snapshot(item)
            continue
        if isinstance(item, Sequence) and not isinstance(item, (str, bytes)):
            snapshot[str(key)] = [
                _sanitize_frame_snapshot(child) if isinstance(child, Mapping) else child
                for child in item
            ]
            continue
        snapshot[str(key)] = item
    return snapshot


def _extract_table_legal_actions(payload: Mapping[str, Any]) -> tuple[dict[str, Any], ...]:
    actions: list[dict[str, Any]] = []
    legal_moves = payload.get("legal_moves")
    if isinstance(legal_moves, Sequence) and not isinstance(legal_moves, (str, bytes)):
        for item in legal_moves:
            normalized = _normalize_table_legal_action(item)
            if normalized is not None:
                actions.append(normalized)
    if actions:
        return tuple(actions)
    return _extract_legal_actions(payload)


def _normalize_table_legal_action(item: Any) -> dict[str, Any] | None:
    if isinstance(item, str):
        text = item.strip()
        if not text:
            return None
        return {
            "id": text,
            "label": text,
            "text": text,
        }
    if not isinstance(item, Mapping):
        return None
    action_id = _string_or_none(item.get("id")) or _string_or_none(item.get("text"))
    label = _string_or_none(item.get("label")) or action_id
    text = _string_or_none(item.get("text")) or action_id
    if action_id is None or label is None or text is None:
        return None
    return {
        "id": action_id,
        "label": label,
        "text": text,
    }


def _resolve_table_observer_player(
    *,
    visual_session: VisualSession,
 ) -> str | None:
    if (
        visual_session.observer.observer_kind == "player"
        and visual_session.observer.observer_id.strip()
    ):
        return visual_session.observer.observer_id
    return None


def _resolve_table_private_view_player(
    *,
    source: Mapping[str, Any],
    private_state: Mapping[str, Any],
) -> str | None:
    return (
        _string_or_none(private_state.get("self_id"))
        or _string_or_none(source.get("player_id"))
        or _string_or_none(source.get("playerId"))
        or _extract_observer_player(source)
    )


def _resolve_projected_table_observer_player(
    *,
    requested_observer_player_id: str | None,
    private_view_player_id: str | None,
) -> str | None:
    if requested_observer_player_id is None:
        return None
    if private_view_player_id is None:
        return None
    if requested_observer_player_id != private_view_player_id:
        return None
    return requested_observer_player_id


def _normalize_played_cards(value: Any) -> dict[str, list[str]]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        return {}
    played_cards: dict[str, list[str]] = {}
    for item in value:
        if not isinstance(item, Mapping):
            continue
        player_id = _string_or_none(item.get("player_id") or item.get("playerId"))
        if player_id is None:
            continue
        played_cards[player_id] = _normalize_string_list(item.get("cards"))
    return played_cards


def _normalize_doudizhu_hands(
    value: Any,
    *,
    player_ids: Sequence[str],
) -> dict[str, list[str]]:
    return _normalize_doudizhu_player_card_map(value, player_ids=player_ids)


def _normalize_doudizhu_latest_actions(
    value: Any,
    *,
    player_ids: Sequence[str],
) -> dict[str, list[str]]:
    return _normalize_doudizhu_player_card_map(value, player_ids=player_ids)


def _normalize_doudizhu_player_card_map(
    value: Any,
    *,
    player_ids: Sequence[str],
) -> dict[str, list[str]]:
    normalized: dict[str, list[str]] = {}
    if isinstance(value, Mapping):
        for raw_player_id, raw_cards in value.items():
            player_id = _string_or_none(raw_player_id)
            if player_id is None:
                continue
            normalized[player_id] = _normalize_doudizhu_card_collection(raw_cards)
        return normalized

    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        return normalized

    for index, raw_cards in enumerate(value):
        if index >= len(player_ids):
            break
        normalized[player_ids[index]] = _normalize_doudizhu_card_collection(raw_cards)
    return normalized


def _normalize_doudizhu_card_collection(value: Any) -> list[str]:
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        cards: list[str] = []
        for item in value:
            card = _string_or_none(item)
            if card is not None and card.lower() != "pass":
                cards.append(card)
        return cards

    text = _string_or_none(value)
    if text is None or text.lower() == "pass":
        return []
    if " " in text or "," in text:
        return [item for item in text.replace(",", " ").split() if item and item.lower() != "pass"]
    return [text]


def _invert_seat_order(
    seat_order: Mapping[str, Any],
    *,
    player_ids: Sequence[str],
) -> dict[str, str]:
    seat_ids: dict[str, str] = {}
    for seat_id, player_id in seat_order.items():
        resolved_player_id = _string_or_none(player_id)
        if resolved_player_id is None:
            continue
        seat_ids[resolved_player_id] = str(seat_id)
    if seat_ids:
        return seat_ids
    fallback = ("bottom", "left", "right", "top")
    return {
        player_id: fallback[index] if index < len(fallback) else f"seat-{index + 1}"
        for index, player_id in enumerate(player_ids)
    }


def _format_doudizhu_history(value: Any, *, player_ids: Sequence[str]) -> list[str]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        return []
    history: list[str] = []
    for item in value:
        if not isinstance(item, Mapping):
            continue
        move = _string_or_none(item.get("move"))
        if move is None:
            continue
        player_ref = item.get("player_id")
        if player_ref is None and isinstance(item.get("player_idx"), int):
            player_idx = int(item["player_idx"])
            if 0 <= player_idx < len(player_ids):
                player_ref = player_ids[player_idx]
        player_id = _string_or_none(player_ref)
        history.append(f"{player_id or 'unknown'}: {move}")
    return history


def _build_doudizhu_panel_payload(
    *,
    player_ids: Sequence[str],
    player_names: Mapping[str, str],
    move_history: Any,
    active_player_id: str | None,
    landlord_id: str | None,
    move_count: int,
    last_move: str | None,
) -> dict[str, Any]:
    trace_entries = _normalize_doudizhu_trace_entries(
        move_history,
        player_ids=player_ids,
        player_names=player_names,
    )
    normalized_last_move = _string_or_none(last_move)
    events: list[dict[str, str]] = []
    if landlord_id is not None:
        events.append(
            {
                "label": "Landlord",
                "detail": _resolve_table_player_label(landlord_id, player_names=player_names),
            }
        )
    if active_player_id is not None:
        events.append(
            {
                "label": "Turn",
                "detail": f"{_resolve_table_player_label(active_player_id, player_names=player_names)} to act",
            }
        )
    if trace_entries:
        last_entry = trace_entries[-1]
        if normalized_last_move is not None and normalized_last_move.strip().lower() != last_entry["move"].strip().lower():
            events.append({"label": "Last move", "detail": normalized_last_move})
        else:
            events.append(
                {
                    "label": "Last move",
                    "detail": _format_doudizhu_last_move_event(
                        player_name=last_entry["playerName"],
                        move=last_entry["move"],
                    ),
                }
            )
    elif normalized_last_move is not None:
        events.append({"label": "Last move", "detail": normalized_last_move})
    if move_count > 0:
        events.append({"label": "Move count", "detail": f"{move_count} moves recorded"})
    return {
        "events": events,
        "trace": [entry["trace"] for entry in trace_entries],
    }


def _format_doudizhu_last_move_event(*, player_name: str, move: str) -> str:
    if move.strip().lower() == "pass":
        return f"{player_name} passed"
    return f"{player_name} played {move}"


def _normalize_doudizhu_trace_entries(
    value: Any,
    *,
    player_ids: Sequence[str],
    player_names: Mapping[str, str],
) -> list[dict[str, str]]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        return []
    entries: list[dict[str, str]] = []
    for item in value:
        if not isinstance(item, Mapping):
            continue
        move = _string_or_none(item.get("move") or item.get("action") or item.get("action_text"))
        if move is None:
            continue
        player_ref = item.get("player_id") or item.get("playerId")
        if player_ref is None and isinstance(item.get("player_idx"), int):
            player_idx = int(item["player_idx"])
            if 0 <= player_idx < len(player_ids):
                player_ref = player_ids[player_idx]
        if player_ref is None and isinstance(item.get("player"), int):
            player_idx = int(item["player"])
            if 0 <= player_idx < len(player_ids):
                player_ref = player_ids[player_idx]
        player_id = _string_or_none(player_ref)
        player_name = _resolve_table_player_label(player_id, player_names=player_names)
        entries.append(
            {
                "playerId": player_id or "unknown",
                "playerName": player_name,
                "move": move,
                "trace": f"{player_name}: {move}",
            }
        )
    return entries


def _resolve_table_player_label(
    player_id: str | None,
    *,
    player_names: Mapping[str, str],
) -> str:
    if player_id is None:
        return "Unknown"
    return player_names.get(player_id, player_id)


def _table_hand_visible(
    *,
    observer_kind: str,
    observer_player_id: str | None,
    player_id: str,
) -> bool:
    if observer_kind != "player" or observer_player_id is None:
        return False
    return observer_player_id == player_id


def _infer_visual_kind(plugin_id: str) -> str:
    text = str(plugin_id or "")
    for kind in ("board", "table", "frame", "rts"):
        marker = f".{kind}_"
        if marker in text or text.endswith(f".{kind}"):
            return kind
    return "frame"


def _extract_active_player(*payloads: Any) -> str | None:
    for payload in payloads:
        if not isinstance(payload, Mapping):
            continue
        for key in ("activePlayerId", "active_player_id", "active_player", "playerId", "player_id"):
            value = _string_or_none(payload.get(key))
            if value is not None:
                return value
    return None


def _extract_legal_actions(*payloads: Any) -> tuple[dict[str, Any], ...]:
    for payload in payloads:
        if not isinstance(payload, Mapping):
            continue
        for key in ("legalActions", "legal_actions"):
            value = payload.get(key)
            if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
                continue
            actions: list[dict[str, Any]] = []
            for item in value:
                if isinstance(item, Mapping):
                    actions.append(dict(item))
            if actions:
                return tuple(actions)
    return ()


def _extract_scene_media(*payloads: Any, kind: str) -> VisualSceneMedia | None:
    explicit = _extract_explicit_media(*payloads)
    if explicit is not None:
        return explicit

    refs = _collect_media_refs(*payloads)
    if not refs:
        return None
    ordered = tuple(refs.values())
    if kind in {"frame", "rts"}:
        return VisualSceneMedia(primary=ordered[0], auxiliary=ordered[1:])
    return VisualSceneMedia(auxiliary=ordered)


def _extract_explicit_media(*payloads: Any) -> VisualSceneMedia | None:
    for payload in payloads:
        if not isinstance(payload, Mapping):
            continue
        media_payload = payload.get("media")
        if isinstance(media_payload, Mapping):
            primary = None
            primary_payload = media_payload.get("primary")
            primary_candidate = _normalize_media_payload(primary_payload)
            if primary_candidate is not None:
                primary = MediaSourceRef.from_dict(primary_candidate)

            auxiliary_items: list[MediaSourceRef] = []
            auxiliary_payload = media_payload.get("auxiliary")
            if isinstance(auxiliary_payload, Sequence) and not isinstance(auxiliary_payload, (str, bytes)):
                for item in auxiliary_payload:
                    candidate = _normalize_media_payload(item)
                    if candidate is not None:
                        auxiliary_items.append(MediaSourceRef.from_dict(candidate))
            if primary is not None or auxiliary_items:
                return VisualSceneMedia(primary=primary, auxiliary=tuple(auxiliary_items))
    return None


def _collect_media_refs(*payloads: Any) -> dict[str, MediaSourceRef]:
    refs: dict[str, MediaSourceRef] = {}
    for payload in payloads:
        _visit_media_refs(payload, refs)
    return refs


def _visit_media_refs(value: Any, refs: dict[str, MediaSourceRef]) -> None:
    if isinstance(value, Mapping):
        candidate = _normalize_media_payload(value)
        if candidate is not None:
            ref = MediaSourceRef.from_dict(candidate)
            refs[ref.media_id] = ref
            return
        for item in value.values():
            _visit_media_refs(item, refs)
        return

    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        for item in value:
            _visit_media_refs(item, refs)


def _looks_like_media_ref(value: Mapping[str, Any]) -> bool:
    return "mediaId" in value and "transport" in value


def _normalize_media_payload(value: Any) -> dict[str, Any] | None:
    if not isinstance(value, Mapping):
        return None
    inline_candidate = _normalize_inline_media_payload(value)
    if inline_candidate is not None:
        return inline_candidate
    if not _looks_like_media_ref(value):
        return None

    normalized: dict[str, Any] = {}
    for key in ("mediaId", "transport", "mimeType", "url", "previewRef"):
        if key not in value:
            continue
        item = value[key]
        if isinstance(item, Mapping) and item.get("kind") == "string":
            normalized[key] = item.get("value")
            continue
        normalized[key] = item

    if not isinstance(normalized.get("mediaId"), str):
        return None
    if not isinstance(normalized.get("transport"), str):
        return None
    return normalized


def _normalize_inline_media_payload(value: Mapping[str, Any]) -> dict[str, Any] | None:
    raw_url = value.get("data_url")
    if raw_url is None:
        raw_url = value.get("dataUrl")
    if raw_url is None:
        raw_url = value.get("url")
    data_url = _string_or_none(raw_url)
    if data_url is None or not data_url.startswith("data:"):
        return None

    mime_type = _infer_inline_media_mime_type(data_url) or _string_or_none(value.get("mimeType")) or "image/png"
    digest = hashlib.sha1(data_url.encode("utf-8")).hexdigest()[:16]
    return {
        "mediaId": f"inline-media-{digest}",
        "transport": "http_pull",
        "mimeType": mime_type,
        "url": data_url,
    }


def _infer_inline_media_mime_type(data_url: str) -> str | None:
    prefix = "data:"
    if not data_url.startswith(prefix):
        return None
    mime_section = data_url[len(prefix) :].split(";", 1)[0].strip()
    return mime_section or None
