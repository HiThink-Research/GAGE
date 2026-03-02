"""PettingZoo discrete-action mapper for websocket human controls."""

from __future__ import annotations

from typing import Any, Mapping, Optional, Sequence

from gage_eval.role.arena.input_mapping import BrowserKeyEvent, GameInputMapper, HumanActionEvent

_DIRECT_ACTION_KEYS = (
    "action",
    "move",
    "selected_action",
    "selected_move",
    "value",
    "text",
)
_INDEX_ACTION_KEYS = ("action_index", "move_index", "index", "action_id")
_KEYDOWN_TYPES = {"keydown", "key_down"}


class PettingZooDiscreteInputMapper(GameInputMapper):
    """Maps websocket payloads into PettingZoo discrete action events."""

    def __init__(
        self,
        *,
        key_map: Optional[Mapping[str, Any]] = None,
        enforce_legal_moves: bool = True,
    ) -> None:
        """Initializes mapper settings.

        Args:
            key_map: Optional shortcut mapping from keyboard key to action text.
                Each value can be:
                - A plain action string (for example: ``"1"``).
                - A dict with ``move``/``action`` and optional ``player_id``.
            enforce_legal_moves: Whether to reject actions missing from legal moves context.
        """

        resolved_key_map: dict[str, tuple[str, Optional[str]]] = {}
        if isinstance(key_map, Mapping):
            for raw_key, raw_value in key_map.items():
                key_name = _normalize_key(raw_key)
                action_text, target_player_id = _parse_key_binding(raw_value)
                if not key_name or not action_text:
                    continue
                resolved_key_map[key_name] = (action_text, target_player_id)
        self._key_map = resolved_key_map
        self._enforce_legal_moves = bool(enforce_legal_moves)

    def _map_event_to_actions(
        self,
        *,
        event: BrowserKeyEvent,
        context: Mapping[str, Any],
    ) -> Sequence[HumanActionEvent]:
        # STEP 1: Resolve one candidate action from payload fields or shortcuts.
        legal_moves = _extract_legal_moves(context)
        candidate, mapped_player_id = self._resolve_action_text(event=event, legal_moves=legal_moves)
        if not candidate:
            return []

        # STEP 2: Align the candidate with legal move text when available.
        resolved_move = self._resolve_legal_action(candidate=candidate, legal_moves=legal_moves)
        if resolved_move is None:
            return []

        # STEP 3: Emit queue-ready action payload for the current human player.
        player_id = str(mapped_player_id or context.get("human_player_id") or "player_0")
        return [
            HumanActionEvent(
                player_id=player_id,
                move=resolved_move,
                raw=resolved_move,
                metadata={
                    "source": "pettingzoo_ws",
                    "event_type": event.event_type,
                    "candidate_action": candidate,
                },
            )
        ]

    def _resolve_action_text(
        self,
        *,
        event: BrowserKeyEvent,
        legal_moves: Sequence[str],
    ) -> tuple[Optional[str], Optional[str]]:
        payload = event.payload

        # STEP 1: Prefer explicit action fields from browser payload.
        for key in _DIRECT_ACTION_KEYS:
            value = payload.get(key)
            if value is None:
                continue
            action_text = str(value).strip()
            if action_text:
                return action_text, None

        # STEP 2: Resolve indexed action when frontend sends selected index.
        indexed = _resolve_index_action(payload, legal_moves)
        if indexed:
            return indexed, None

        # STEP 3: Fallback to keyboard shortcuts for keydown events.
        if event.event_type in _KEYDOWN_TYPES and event.key:
            mapped = self._key_map.get(_normalize_key(event.key))
            if mapped:
                return mapped

        return None, None

    def _resolve_legal_action(
        self,
        *,
        candidate: str,
        legal_moves: Sequence[str],
    ) -> Optional[str]:
        normalized_candidate = str(candidate).strip()
        if not normalized_candidate:
            return None
        if not legal_moves:
            return normalized_candidate
        lookup = {str(move).strip().lower(): str(move) for move in legal_moves if str(move).strip()}
        normalized_key = normalized_candidate.lower()
        if normalized_key in lookup:
            return lookup[normalized_key]
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


def _normalize_key(value: Any) -> str:
    return str(value or "").strip().lower()


def _parse_key_binding(value: Any) -> tuple[Optional[str], Optional[str]]:
    if isinstance(value, Mapping):
        action_value = (
            value.get("move")
            or value.get("action")
            or value.get("value")
            or value.get("text")
        )
        action_text = str(action_value or "").strip()
        player_value = value.get("player_id") or value.get("playerId") or value.get("player")
        player_id = str(player_value).strip() if player_value is not None else None
        if player_id == "":
            player_id = None
        return (action_text or None), player_id
    action_text = str(value or "").strip()
    return (action_text or None), None
