"""RLCard core implementation for Doudizhu."""

from __future__ import annotations

import functools
from typing import Optional, Sequence

from gage_eval.game_kits.phase_card_game.shared.rlcard_core import RLCardCore as SharedRLCardCore


def _normalize_rlcard_card(card: object) -> str:
    text = str(card).strip().upper()
    if text in {"BJ", "BLACKJOKER"}:
        return "BJ"
    if text in {"RJ", "REDJOKER"}:
        return "RJ"
    if len(text) >= 2 and text[-1] in {"S", "H", "D", "C"}:
        rank = text[:-1]
        suit = text[-1]
        return f"{suit}{'10' if rank == '10' else rank}"
    return text


class RLCardCore(SharedRLCardCore):
    """Doudizhu-specific RLCard helpers."""

    def get_public_cards(self) -> Optional[Sequence[str]]:
        game = getattr(self._env, "game", None)
        round_state = getattr(game, "round", None)
        dealer = getattr(round_state, "dealer", None)
        deck = getattr(dealer, "deck", None)
        if not deck:
            return super().get_public_cards()
        try:
            from rlcard.games.doudizhu.utils import doudizhu_sort_card
        except Exception:
            doudizhu_sort_card = None
        public_cards = list(deck[-3:])
        if callable(doudizhu_sort_card):
            public_cards.sort(key=functools.cmp_to_key(doudizhu_sort_card))
        return [_normalize_rlcard_card(card) for card in public_cards]

__all__ = ["RLCardCore"]
