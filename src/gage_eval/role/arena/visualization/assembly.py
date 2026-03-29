from __future__ import annotations

import hashlib
from collections.abc import Mapping, Sequence
from typing import Any

from gage_eval.game_kits.contracts import GameVisualizationSpec
from gage_eval.role.arena.games.gomoku.coord_scheme import GomokuCoordCodec, normalize_coord_scheme
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
    observation = _mapping_or_empty(source.get("observation"))
    metadata = _mapping_or_empty(source.get("metadata"))
    if not metadata:
        metadata = _mapping_or_empty(observation.get("metadata"))
    context = _mapping_or_empty(source.get("context"))
    if not context:
        context = _mapping_or_empty(observation.get("context"))
    view = _mapping_or_empty(source.get("view"))
    if not view:
        view = _mapping_or_empty(observation.get("view"))

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
    source = _build_projection_source(event_payload, snapshot_body)
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
    chat_log = _normalize_chat_log(source.get("chat_log"))
    move_count = _coerce_int(source.get("move_count"), default=0)
    last_move = source.get("last_move")
    landlord_id = public_state.get("landlord_id")
    legal_actions = _extract_table_legal_actions(source)
    table_game = str(
        _scene_projection_rule_value(visualization_spec, "table_game")
        or source.get("game_id")
        or visual_session.game_id
    )

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
        "panels": {
            "chatLog": chat_log,
        },
    }
    return body, legal_actions, active_player_id


def _project_frame_scene(
    *,
    visual_session: VisualSession,
    snapshot_body: Any,
    event_payload: Any,
    visualization_spec: GameVisualizationSpec | None,
) -> tuple[dict[str, Any], tuple[dict[str, Any], ...], str | None, tuple[dict[str, Any], ...]]:
    source = _build_projection_source(event_payload, snapshot_body)
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
    legal_actions = _extract_frame_legal_actions(source, observation)
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
    private_self_id = _string_or_none(private_state.get("self_id"))
    private_hand = _normalize_string_list(private_state.get("current_hand"))
    seats: list[dict[str, Any]] = []
    for player_id in player_ids:
        is_visible = _table_hand_visible(
            observer_kind=observer_kind,
            observer_player_id=observer_player_id,
            player_id=player_id,
        )
        if player_id == private_self_id and private_hand:
            cards = private_hand if is_visible else []
            masked_count = 0 if is_visible else len(private_hand)
        else:
            cards = []
            masked_count = 0 if is_visible else card_counts.get(player_id, 0)
        seats.append(
            {
                "seatId": seat_ids.get(player_id, player_id),
                "playerId": player_id,
                "playerName": player_names.get(player_id, player_id),
                "role": _string_or_none(roles.get(player_id)),
                "isActive": bool(active_player_id and player_id == active_player_id),
                "isObserver": bool(observer_player_id and player_id == observer_player_id),
                "playedCards": played_cards.get(player_id, []),
                "publicNotes": [],
                "hand": {
                    "isVisible": is_visible,
                    "cards": cards,
                    "maskedCount": masked_count,
                },
            }
        )

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
    melds = _mapping_or_empty(public_state.get("melds"))
    seats: list[dict[str, Any]] = []
    for player_id in player_ids:
        is_visible = _table_hand_visible(
            observer_kind=observer_kind,
            observer_player_id=observer_player_id,
            player_id=player_id,
        )
        masked_count = len(private_hand) if player_id == private_view_player_id and private_hand else 0
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
                "hand": {
                    "isVisible": is_visible,
                    "cards": private_hand if is_visible else [],
                    "maskedCount": 0 if is_visible else masked_count,
                },
            }
        )

    layout = _string_or_none(_scene_projection_rule_value(visualization_spec, "default_layout")) or "four-seat"
    return {
        "layout": layout,
        "seats": seats,
        "center": {
            "label": "Discards",
            "cards": _normalize_string_list(public_state.get("discards")),
            "history": [],
        },
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
        normalized = _normalize_frame_legal_action(item)
        if normalized is not None:
            actions.append(normalized)
    return tuple(actions)


def _normalize_frame_legal_action(item: Any) -> dict[str, Any] | None:
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
    action_id = _string_or_none(item.get("id")) or _string_or_none(item.get("text")) or _string_or_none(item.get("label"))
    if action_id is None:
        return None
    label = _string_or_none(item.get("label")) or action_id
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
    if kind == "frame":
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
