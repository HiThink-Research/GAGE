"""Lightweight Doudizhu replay projection helpers."""

from __future__ import annotations

from typing import Any, Mapping, Optional, Sequence


def build_doudizhu_replay_projection(
    *,
    player_ids: Sequence[str],
    player_names: Mapping[str, str],
    landlord_id: Optional[str],
    active_player_index: int,
    legal_moves: Sequence[str],
    initial_hands: Sequence[Sequence[str]],
    move_history: Sequence[Mapping[str, Any]],
    chat_log: Sequence[Mapping[str, str]],
    start_time_ms: Optional[int],
) -> dict[str, Any]:
    """Build a compact JSON replay payload for the Doudizhu arena."""

    roles = _resolve_roles(player_ids, landlord_id)
    player_info = [
        {
            "id": idx,
            "index": idx,
            "role": roles.get(player_id, "peasant"),
            "agentInfo": {"name": player_names.get(player_id, player_id)},
        }
        for idx, player_id in enumerate(player_ids)
    ]
    init_hands = [" ".join(cards) for cards in initial_hands]
    if len(init_hands) < len(player_ids):
        init_hands.extend(["" for _ in range(len(player_ids) - len(init_hands))])
    else:
        init_hands = init_hands[: len(player_ids)]

    move_payload: list[dict[str, Any]] = []
    for entry in move_history:
        player_idx = entry.get("player_idx")
        action_cards = entry.get("action_cards")
        action_text = entry.get("action_text")
        move = action_cards if action_cards not in (None, "") else action_text
        if isinstance(move, list):
            move_text = " ".join(str(card) for card in move)
        elif move is None:
            move_text = ""
        else:
            move_text = str(move)
        move_payload.append(
            {
                "playerIdx": int(player_idx) if player_idx is not None else 0,
                "move": move_text,
                "info": {},
                "chat": entry.get("chat"),
                "timestamp_ms": entry.get("timestamp_ms"),
            }
        )

    return {
        "playerInfo": player_info,
        "initHands": init_hands,
        "moveHistory": move_payload,
        "chatLog": list(chat_log),
        "currentPlayer": int(active_player_index),
        "active_player_id": _resolve_active_player_id(player_ids, active_player_index),
        "legalMoves": list(dict.fromkeys(str(move) for move in legal_moves)),
        "start_time_ms": start_time_ms,
    }


def _resolve_roles(player_ids: Sequence[str], landlord_id: Optional[str]) -> dict[str, str]:
    roles: dict[str, str] = {}
    landlord = str(landlord_id) if landlord_id else None
    for player_id in player_ids:
        roles[player_id] = "landlord" if landlord and player_id == landlord else "peasant"
    return roles


def _resolve_active_player_id(player_ids: Sequence[str], active_player_index: int) -> Optional[str]:
    if active_player_index < 0 or active_player_index >= len(player_ids):
        return None
    return str(player_ids[active_player_index])

