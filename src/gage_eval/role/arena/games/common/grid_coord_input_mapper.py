"""Grid coordinate input mapper for websocket-driven human actions."""

from __future__ import annotations

from typing import Any, Mapping, Optional, Sequence

from gage_eval.role.arena.input_mapping import BrowserKeyEvent, GameInputMapper, HumanActionEvent

_DIRECT_ACTION_KEYS = (
    "action",
    "move",
    "coord",
    "selected_action",
    "selected_move",
    "selected_coord",
    "value",
    "text",
)
_INDEX_ACTION_KEYS = ("action_index", "move_index", "index")
_KEYDOWN_TYPES = {"keydown", "key_down"}


class GridCoordInputMapper(GameInputMapper):
    """Maps websocket payloads into grid-game coordinate actions."""

    def __init__(
        self,
        *,
        key_map: Optional[Mapping[str, str]] = None,
        coord_scheme: Optional[str] = None,
        enforce_legal_moves: bool = True,
    ) -> None:
        """Initializes mapper settings.

        Args:
            key_map: Optional shortcut mapping from keyboard key to coordinate text.
            coord_scheme: Optional coordinate scheme hint (for example: A1, ROW_COL).
            enforce_legal_moves: Whether to reject actions missing from legal moves context.
        """

        resolved_key_map: dict[str, str] = {}
        if isinstance(key_map, Mapping):
            for raw_key, raw_coord in key_map.items():
                key_name = _normalize_key(raw_key)
                coord_text = str(raw_coord or "").strip()
                if not key_name or not coord_text:
                    continue
                resolved_key_map[key_name] = coord_text
        self._key_map = resolved_key_map
        self._coord_scheme = str(coord_scheme).strip().upper() if coord_scheme else None
        self._enforce_legal_moves = bool(enforce_legal_moves)

    def _map_event_to_actions(
        self,
        *,
        event: BrowserKeyEvent,
        context: Mapping[str, Any],
    ) -> Sequence[HumanActionEvent]:
        # STEP 1: Resolve one candidate coordinate from payload or keyboard shortcuts.
        legal_moves = _extract_legal_moves(context)
        coord_scheme = _resolve_coord_scheme(context, default=self._coord_scheme)
        candidate = self._resolve_action_text(event=event, legal_moves=legal_moves, coord_scheme=coord_scheme)
        if not candidate:
            return []

        # STEP 2: Align candidate with legal move text when available.
        resolved_move = self._resolve_legal_action(
            candidate=candidate,
            legal_moves=legal_moves,
            coord_scheme=coord_scheme,
        )
        if resolved_move is None:
            return []

        # STEP 3: Emit queue-ready action payload for the current human player.
        player_id = str(context.get("human_player_id") or "player_0")
        return [
            HumanActionEvent(
                player_id=player_id,
                move=resolved_move,
                raw=resolved_move,
                metadata={
                    "source": "grid_ws",
                    "event_type": event.event_type,
                    "candidate_action": candidate,
                    "coord_scheme": coord_scheme,
                },
            )
        ]

    def _resolve_action_text(
        self,
        *,
        event: BrowserKeyEvent,
        legal_moves: Sequence[str],
        coord_scheme: str,
    ) -> Optional[str]:
        payload = event.payload

        # STEP 1: Prefer explicit move fields.
        for key in _DIRECT_ACTION_KEYS:
            value = payload.get(key)
            if value is None:
                continue
            normalized = _normalize_coord_text(str(value), coord_scheme=coord_scheme)
            if normalized:
                return normalized

        # STEP 2: Resolve indexed selection when frontend sends action index.
        indexed = _resolve_index_action(payload, legal_moves)
        if indexed:
            return _normalize_coord_text(indexed, coord_scheme=coord_scheme)

        # STEP 3: Fallback to keyboard shortcuts on keydown events.
        if event.event_type in _KEYDOWN_TYPES and event.key:
            mapped = self._key_map.get(_normalize_key(event.key))
            if mapped:
                return _normalize_coord_text(mapped, coord_scheme=coord_scheme)

        return None

    def _resolve_legal_action(
        self,
        *,
        candidate: str,
        legal_moves: Sequence[str],
        coord_scheme: str,
    ) -> Optional[str]:
        normalized_candidate = _normalize_coord_text(candidate, coord_scheme=coord_scheme)
        if not normalized_candidate:
            return None
        if not legal_moves:
            return normalized_candidate
        lookup = {
            _normalize_coord_text(str(move), coord_scheme=coord_scheme): str(move)
            for move in legal_moves
            if str(move).strip()
        }
        if normalized_candidate in lookup:
            return str(lookup[normalized_candidate])
        if self._enforce_legal_moves:
            return None
        return normalized_candidate


def _resolve_index_action(payload: Mapping[str, Any], legal_moves: Sequence[str]) -> Optional[str]:
    if not legal_moves:
        return None
    for key in _INDEX_ACTION_KEYS:
        value = payload.get(key)
        if value is None:
            continue
        try:
            parsed = int(value)
        except (TypeError, ValueError):
            continue
        one_based = parsed - 1
        zero_based = parsed
        if 0 <= one_based < len(legal_moves):
            return str(legal_moves[one_based])
        if 0 <= zero_based < len(legal_moves):
            return str(legal_moves[zero_based])
    return None


def _extract_legal_moves(context: Mapping[str, Any]) -> list[str]:
    direct = context.get("legal_moves")
    if isinstance(direct, Sequence) and not isinstance(direct, (str, bytes)):
        return [str(item) for item in direct]
    legal_actions = context.get("legal_actions")
    if isinstance(legal_actions, Mapping):
        items = legal_actions.get("items")
        if isinstance(items, Sequence) and not isinstance(items, (str, bytes)):
            return [str(item) for item in items]
    return []


def _resolve_coord_scheme(context: Mapping[str, Any], *, default: Optional[str]) -> str:
    explicit = context.get("coord_scheme")
    if explicit:
        return str(explicit).strip().upper()
    metadata = context.get("metadata")
    if isinstance(metadata, Mapping):
        scheme = metadata.get("coord_scheme")
        if scheme:
            return str(scheme).strip().upper()
    if default:
        return str(default).strip().upper()
    return "A1"


def _normalize_coord_text(value: str, *, coord_scheme: str) -> str:
    cleaned = str(value or "").strip()
    if not cleaned:
        return ""
    lowered = cleaned.lower()
    for prefix in ("action:", "move:", "coord:"):
        if lowered.startswith(prefix):
            cleaned = cleaned[len(prefix) :].strip()
            lowered = cleaned.lower()
            break
    if str(coord_scheme).upper() == "ROW_COL":
        normalized = cleaned.replace(" ", ",").replace(":", ",")
        while ",," in normalized:
            normalized = normalized.replace(",,", ",")
        return normalized.strip(",")
    return cleaned.upper()


def _normalize_key(value: Any) -> str:
    return str(value or "").strip().lower()
