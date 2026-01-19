"""Mahjong action mapping helpers aligned with RLCard."""

from __future__ import annotations

from functools import lru_cache
from typing import Any


def rlcard_card_to_code(card: Any) -> str:
    """Convert an RLCard Mahjong card or string into a display code."""

    raw = card.get_str() if hasattr(card, "get_str") else str(card)
    return _format_tile_code(raw)


def rlcard_action_to_display(action_text: str) -> str:
    """Normalize an RLCard action string into a display-friendly code."""

    normalized = str(action_text or "").strip()
    if not normalized:
        return ""
    lowered = normalized.lower()
    special = {"pong", "chow", "gong", "stand"}
    if lowered in special:
        return lowered.capitalize()
    return _format_tile_code(normalized)


@lru_cache(maxsize=1)
def build_action_maps() -> tuple[dict[int, str], dict[str, int], dict[int, str]]:
    """Build action mappings for Mahjong using RLCard's action ids.

    Returns:
        Tuple containing (action_id_to_text, action_text_to_id, action_id_to_raw).
    """

    try:
        from rlcard.games.mahjong.utils import card_encoding_dict
    except Exception as exc:
        raise RuntimeError("rlcard is required to build Mahjong action mappings") from exc

    action_id_to_text: dict[int, str] = {}
    action_text_to_id: dict[str, int] = {}
    action_id_to_raw: dict[int, str] = {}
    for raw_text, action_id in card_encoding_dict.items():
        action_id = int(action_id)
        raw_str = str(raw_text)
        action_id_to_raw[action_id] = raw_str
        display_text = rlcard_action_to_display(raw_str)
        action_id_to_text[action_id] = display_text
        action_text_to_id[display_text.lower()] = action_id
        action_text_to_id[raw_str.lower()] = action_id

    stand_id = card_encoding_dict.get("stand")
    if stand_id is not None:
        stand_id_int = int(stand_id)
        action_text_to_id.setdefault("hu", stand_id_int)
        action_text_to_id.setdefault("pass", stand_id_int)
        action_text_to_id.setdefault("skip", stand_id_int)

    return action_id_to_text, action_text_to_id, action_id_to_raw


def is_tile_action(raw_action: str) -> bool:
    """Return True when the raw RLCard action represents a tile."""

    raw = str(raw_action or "").strip().lower()
    if "-" not in raw:
        return False
    suit = raw.split("-", 1)[0]
    return suit in {"bamboo", "characters", "dots", "dragons", "winds"}


def _format_tile_code(raw_text: str) -> str:
    normalized = str(raw_text or "").strip()
    if not normalized:
        return ""

    upper = normalized.upper()
    if len(upper) == 2 and upper[0] in {"B", "C", "D"} and upper[1].isdigit():
        return upper

    titled = normalized.capitalize()
    if titled in {"East", "South", "West", "North", "Green", "Red", "White"}:
        return titled

    if "-" not in normalized:
        return normalized

    suit, rank = normalized.split("-", 1)
    suit = suit.lower()
    rank = rank.lower()
    if suit == "bamboo":
        return f"B{rank}"
    if suit == "characters":
        return f"C{rank}"
    if suit == "dots":
        return f"D{rank}"
    if suit == "winds":
        return rank.capitalize()
    if suit == "dragons":
        return rank.capitalize()
    return normalized

