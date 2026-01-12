"""Showdown-inspired Doudizhu renderer for the Gradio visualizer."""

from __future__ import annotations

import html
import json
from typing import Any, Optional, Sequence

from gage_eval.registry import registry

_DOUDIZHU_DECK_ORDER = [
    "RJ",
    "BJ",
    "S2",
    "C2",
    "H2",
    "D2",
    "SA",
    "CA",
    "HA",
    "DA",
    "SK",
    "CK",
    "HK",
    "DK",
    "SQ",
    "CQ",
    "HQ",
    "DQ",
    "SJ",
    "CJ",
    "HJ",
    "DJ",
    "ST",
    "CT",
    "HT",
    "DT",
    "S9",
    "C9",
    "H9",
    "D9",
    "S8",
    "C8",
    "H8",
    "D8",
    "S7",
    "C7",
    "H7",
    "D7",
    "S6",
    "C6",
    "H6",
    "D6",
    "S5",
    "C5",
    "H5",
    "D5",
    "S4",
    "C4",
    "H4",
    "D4",
    "S3",
    "C3",
    "H3",
    "D3",
]
_DOUDIZHU_DECK_INDEX = {card: idx for idx, card in enumerate(_DOUDIZHU_DECK_ORDER)}

_SUIT_CLASS = {"H": "hearts", "D": "diams", "S": "spades", "C": "clubs"}
_SUIT_HTML = {"H": "&hearts;", "D": "&diams;", "S": "&spades;", "C": "&clubs;"}

DOUDIZHU_SHOWDOWN_CSS = """
:root {
  --doudizhu-green: #0f5132;
  --doudizhu-green-dark: #0a2f20;
  --doudizhu-gold: #f6c66a;
  --doudizhu-panel: rgba(5, 12, 8, 0.68);
  --doudizhu-panel-border: rgba(255, 255, 255, 0.12);
  --doudizhu-text: #f8f4ed;
  --doudizhu-muted: rgba(248, 244, 237, 0.65);
  --doudizhu-shadow: rgba(0, 0, 0, 0.4);
  --card-bg: #f7f3ec;
  --card-border: #151515;
}

.doudizhu-shell {
  font-family: "Space Grotesk", "IBM Plex Sans", "Source Sans 3", "Segoe UI", sans-serif;
  color: var(--doudizhu-text);
  width: 100%;
}

.doudizhu-wrapper {
  position: relative;
  width: 100%;
  min-height: 520px;
  border-radius: 18px;
  padding: 16px;
  background: radial-gradient(circle at 30% 0%, #1b6d4f 0%, var(--doudizhu-green) 35%, var(--doudizhu-green-dark) 100%);
  box-shadow: 0 18px 40px var(--doudizhu-shadow);
  overflow: hidden;
}

.doudizhu-wrapper::before {
  content: "";
  position: absolute;
  inset: 0;
  background-image: linear-gradient(120deg, rgba(255, 255, 255, 0.08) 0%, rgba(255, 255, 255, 0) 40%),
    repeating-linear-gradient(45deg, rgba(255, 255, 255, 0.04) 0, rgba(255, 255, 255, 0.04) 1px, transparent 1px, transparent 6px);
  opacity: 0.45;
  pointer-events: none;
}

#gameboard-background {
  position: relative;
  width: 100%;
  min-height: 480px;
  height: 100%;
  z-index: 1;
}

.doudizhu-player {
  position: absolute;
  display: flex;
  flex-direction: column;
  gap: 8px;
  z-index: 2;
}

#doudizhu-left {
  top: 18px;
  left: 18px;
  width: 300px;
  align-items: flex-start;
}

#doudizhu-right {
  top: 18px;
  right: 18px;
  width: 300px;
  align-items: flex-end;
  text-align: right;
}

#doudizhu-bottom {
  bottom: 14px;
  left: 50%;
  transform: translateX(-50%);
  width: min(760px, 96%);
  align-items: center;
}

.player-panel {
  width: 100%;
  background: var(--doudizhu-panel);
  border: 1px solid var(--doudizhu-panel-border);
  border-radius: 16px;
  padding: 12px 14px;
  backdrop-filter: blur(6px);
  box-shadow: 0 12px 30px var(--doudizhu-shadow);
}

.player-panel.active {
  border-color: var(--doudizhu-gold);
  box-shadow: 0 0 0 2px rgba(246, 198, 106, 0.6), 0 0 18px rgba(246, 198, 106, 0.55);
  animation: pulseGlow 1.8s ease-in-out infinite;
}

@keyframes pulseGlow {
  0% { box-shadow: 0 0 0 1px rgba(246, 198, 106, 0.4), 0 0 12px rgba(246, 198, 106, 0.35); }
  50% { box-shadow: 0 0 0 2px rgba(246, 198, 106, 0.8), 0 0 22px rgba(246, 198, 106, 0.6); }
  100% { box-shadow: 0 0 0 1px rgba(246, 198, 106, 0.4), 0 0 12px rgba(246, 198, 106, 0.35); }
}

.player-header {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 12px;
  margin-bottom: 10px;
  font-size: 12px;
  text-transform: uppercase;
  letter-spacing: 0.06em;
}

.player-name {
  font-size: 16px;
  font-weight: 600;
  text-transform: none;
  letter-spacing: normal;
}

.player-role {
  padding: 4px 10px;
  border-radius: 999px;
  font-weight: 600;
  font-size: 11px;
  color: #101010;
  background: var(--doudizhu-gold);
}

.player-role.role-peasant {
  background: #d9ed92;
  color: #1f2d16;
}

.player-count {
  font-size: 11px;
  color: var(--doudizhu-muted);
}

.player-hand {
  display: flex;
  flex-direction: column;
  gap: 8px;
}

.played-card-area {
  min-height: 72px;
  display: flex;
  align-items: center;
  justify-content: center;
  width: 100%;
}

.non-card {
  width: 100%;
  padding: 12px;
  border-radius: 12px;
  background: rgba(0, 0, 0, 0.25);
  text-align: center;
  font-weight: 600;
  letter-spacing: 0.08em;
  text-transform: uppercase;
}

.doudizhu-board {
  white-space: pre-wrap;
  font-size: 12px;
  color: var(--doudizhu-text);
  margin: 0;
}

.playingCards {
  display: flex;
  flex-direction: column;
}

.playingCards ul.hand {
  list-style: none;
  padding: 0;
  margin: 0;
  display: flex;
  align-items: flex-end;
  flex-wrap: nowrap;
}

.playingCards ul.hand li {
  margin-left: -18px;
}

.playingCards ul.hand li:first-child {
  margin-left: 0;
}

.card {
  width: 44px;
  height: 62px;
  background: var(--card-bg);
  border: 1px solid var(--card-border);
  border-radius: 8px;
  box-shadow: 0 4px 12px rgba(0, 0, 0, 0.35);
  position: relative;
  display: flex;
  align-items: center;
  justify-content: center;
  font-weight: 700;
  color: #111;
}

.card .rank {
  position: absolute;
  top: 4px;
  left: 5px;
  font-size: 12px;
}

.card .suit {
  position: absolute;
  bottom: 4px;
  right: 6px;
  font-size: 12px;
}

.card.hearts,
.card.diams {
  color: #d7263d;
}

.card.joker {
  background: linear-gradient(135deg, #1f1f1f, #0e0e0e);
  color: #f6c66a;
  border-color: #f6c66a;
}

.card.joker.big {
  color: #f94144;
}

.card.joker.little {
  color: #577590;
}

.player-hand-side .card {
  width: 34px;
  height: 50px;
  font-size: 12px;
}

.player-hand-side .card .rank,
.player-hand-side .card .suit {
  font-size: 10px;
}

.player-hand-bottom .card {
  width: 48px;
  height: 70px;
}

.played-card-area .card {
  width: 36px;
  height: 52px;
}

.played-card-area .card .rank,
.played-card-area .card .suit {
  font-size: 10px;
}

.played-card-area .playingCards ul.hand li {
  margin-left: -14px;
}

.player-hand-placeholder {
  padding: 12px;
  border-radius: 12px;
  background: rgba(0, 0, 0, 0.2);
  text-align: center;
  font-size: 12px;
  color: var(--doudizhu-muted);
}

.gomoku-output-label {
  display: flex;
  align-items: center;
  gap: 10px;
  justify-content: flex-start;
  margin-bottom: 12px;
  padding-bottom: 10px;
  border-bottom: 1px solid rgba(255, 255, 255, 0.12);
}
.player-pill {
  padding: 4px 10px;
  border-radius: 6px;
  font-size: 11px;
  font-weight: 700;
  text-transform: uppercase;
  flex-shrink: 0;
  background: rgba(255, 255, 255, 0.15);
  color: var(--doudizhu-text);
}
.player-pill.black { background: #101615; color: #f3f0ea; border: 1px solid rgba(255, 255, 255, 0.1); }
.player-pill.white { background: #f0e7d6; color: #2a1f12; border: 1px solid rgba(0, 0, 0, 0.08); }
.player-name-inline {
  font-size: 14px;
  font-weight: 600;
  color: var(--doudizhu-text);
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
}
.move-pill {
  padding: 4px 10px;
  border-radius: 999px;
  background: rgba(246, 198, 106, 0.2);
  font-size: 12px;
  font-weight: 700;
  color: #f6c66a;
  border: 1px solid rgba(246, 198, 106, 0.5);
  letter-spacing: 0.5px;
}

#gomoku-chat-panel {
  margin-top: 12px;
}

.chat-panel {
  width: 100%;
  background: var(--doudizhu-panel);
  border: 1px solid var(--doudizhu-panel-border);
  border-radius: 14px;
  padding: 12px;
  box-shadow: 0 10px 24px var(--doudizhu-shadow);
}

.chat-panel + .chat-panel {
  margin-top: 10px;
}

.chat-header {
  font-size: 11px;
  text-transform: uppercase;
  letter-spacing: 0.08em;
  color: var(--doudizhu-muted);
  margin-bottom: 8px;
}

.chat-body {
  display: flex;
  flex-direction: column;
  gap: 8px;
  max-height: 220px;
  overflow-y: auto;
}

.chat-line {
  display: flex;
  flex-direction: column;
  gap: 4px;
  padding: 8px 10px;
  border-radius: 10px;
  background: rgba(0, 0, 0, 0.22);
}

.chat-speaker {
  font-size: 10px;
  font-weight: 700;
  text-transform: uppercase;
  letter-spacing: 0.06em;
  color: var(--doudizhu-gold);
}

.chat-speaker.black {
  color: #f4d58d;
}

.chat-speaker.white {
  color: #f6efe2;
}

.chat-text {
  font-size: 13px;
  line-height: 1.4;
  color: var(--doudizhu-text);
}

.chat-empty-text {
  font-size: 12px;
  color: var(--doudizhu-muted);
}

.status-line {
  font-size: 14px;
  font-weight: 500;
  color: #1f1b15;
  padding: 12px 16px;
  background: #f6efe2;
  border-radius: 10px;
  border-left: 4px solid #c9a76a;
  margin-bottom: 16px;
  display: block;
  width: 100%;
  box-sizing: border-box;
}

#gomoku-output-box textarea,
#gomoku-raw-display textarea {
  font-family: "IBM Plex Mono", "JetBrains Mono", monospace;
  font-size: 12.5px;
  line-height: 1.6;
  color: #2a2119;
  background: #fdf7ee;
  border: 1px solid #d9c7a7;
  border-radius: 10px;
  padding: 14px;
  white-space: pre-wrap;
  word-break: break-word;
  box-shadow: inset 0 1px 3px rgba(0, 0, 0, 0.12);
}

#gomoku-output-box textarea {
  min-height: 520px !important;
  height: 100% !important;
}

#gomoku-raw-display textarea {
  min-height: 240px !important;
  height: 100% !important;
}

@media (max-width: 900px) {
  .doudizhu-wrapper {
    min-height: 640px;
    padding: 12px;
  }

  #doudizhu-left,
  #doudizhu-right {
    position: relative;
    width: 100%;
    top: 0;
    left: 0;
    right: 0;
    margin-bottom: 12px;
  }

  #doudizhu-right {
    align-items: flex-start;
    text-align: left;
  }

  #doudizhu-bottom {
    position: relative;
    bottom: 0;
    left: 0;
    transform: none;
    width: 100%;
  }

  .doudizhu-player {
    position: relative;
  }
}
"""


@registry.asset(
    "renderer_impls",
    "doudizhu_showdown_v1",
    desc="Showdown-style Doudizhu renderer for Gradio visualizer",
    tags=("doudizhu", "renderer", "card"),
)
class DoudizhuShowdownRenderer:
    """Render Doudizhu state snapshots using a showdown-inspired layout."""

    def __init__(self, board_size: int = 0, coord_scheme: str = "A1", **_: object) -> None:
        """Initialize the renderer.

        Args:
            board_size: Unused placeholder for interface compatibility.
            coord_scheme: Unused placeholder for interface compatibility.
            **_: Ignored extra keyword arguments.
        """

        self._board_size = int(board_size)
        self._coord_scheme = str(coord_scheme)
        self._raw_text = ""
        self._last_move: Optional[str] = None
        self._ui_state: dict[str, Any] = {}

    def update(
        self,
        board_text: str,
        *,
        last_move: Optional[str] = None,
        winning_line: Optional[Sequence[str]] = None,
    ) -> None:
        """Update the renderer with a new snapshot."""

        _ = winning_line
        self._raw_text = board_text or ""
        self._last_move = last_move
        self._ui_state = self._parse_ui_state(self._raw_text)

    def resize(self, board_size: int) -> None:
        """Resize the renderer when requested by the visualizer."""

        self._board_size = int(board_size)

    def set_coord_scheme(self, coord_scheme: str) -> None:
        """Update the coordinate scheme metadata."""

        self._coord_scheme = str(coord_scheme)

    def render_html(self, *, interactive: bool) -> str:
        """Render the current snapshot into HTML markup."""

        _ = interactive
        if not self._ui_state:
            return self._render_fallback()
        player_ids = list(self._ui_state.get("player_ids") or [])
        player_names = dict(self._ui_state.get("player_names") or {})
        roles = dict(self._ui_state.get("roles") or {})
        seat_order = dict(self._ui_state.get("seat_order") or {})
        hands = list(self._ui_state.get("hands") or [])
        latest_actions = list(self._ui_state.get("latest_actions") or [])
        active_player_id = self._ui_state.get("active_player_id")

        hand_map = {player_id: hands[idx] if idx < len(hands) else [] for idx, player_id in enumerate(player_ids)}
        action_map = {
            player_id: latest_actions[idx] if idx < len(latest_actions) else [] for idx, player_id in enumerate(player_ids)
        }

        bottom_id = seat_order.get("bottom") or (player_ids[0] if player_ids else "")
        left_id = seat_order.get("left") or (player_ids[1] if len(player_ids) > 1 else bottom_id)
        right_id = seat_order.get("right") or (player_ids[2] if len(player_ids) > 2 else bottom_id)

        left_html = self._render_player(
            player_id=left_id,
            position="left",
            player_names=player_names,
            roles=roles,
            hand_cards=hand_map.get(left_id, []),
            latest_action=action_map.get(left_id, []),
            active_player_id=active_player_id,
        )
        right_html = self._render_player(
            player_id=right_id,
            position="right",
            player_names=player_names,
            roles=roles,
            hand_cards=hand_map.get(right_id, []),
            latest_action=action_map.get(right_id, []),
            active_player_id=active_player_id,
        )
        bottom_html = self._render_player(
            player_id=bottom_id,
            position="bottom",
            player_names=player_names,
            roles=roles,
            hand_cards=hand_map.get(bottom_id, []),
            latest_action=action_map.get(bottom_id, []),
            active_player_id=active_player_id,
        )

        return (
            "<div class=\"doudizhu-shell\">"
            "<div class=\"doudizhu-wrapper\">"
            "<div id=\"gameboard-background\">"
            f"{left_html}{right_html}{bottom_html}"
            "</div>"
            "</div>"
            "</div>"
        )

    def raw_text(self) -> str:
        """Return the latest raw board text."""

        return self._pretty_board_text(self._raw_text)

    def get_css(self) -> str:
        """Return CSS required for the renderer."""

        return DOUDIZHU_SHOWDOWN_CSS

    def build_interaction_js(
        self,
        *,
        board_container_id: str,
        move_input_id: str,
        submit_button_id: str,
        enable_click: bool,
        refresh_button_id: str,
        refresh_interval_ms: int,
    ) -> str:
        """Return JS for click-to-move interactions (unused)."""

        _ = (
            board_container_id,
            move_input_id,
            submit_button_id,
            enable_click,
            refresh_button_id,
            refresh_interval_ms,
        )
        return ""

    def _parse_ui_state(self, board_text: str) -> dict[str, Any]:
        marker = "UI_STATE_JSON:"
        if marker in board_text:
            _, payload = board_text.split(marker, 1)
            try:
                return json.loads(payload.strip())
            except json.JSONDecodeError:
                return {}
        try:
            return json.loads(board_text)
        except json.JSONDecodeError:
            return {}

    def _render_player(
        self,
        *,
        player_id: str,
        position: str,
        player_names: dict[str, str],
        roles: dict[str, str],
        hand_cards: Sequence[str] | str,
        latest_action: Sequence[str] | str,
        active_player_id: Optional[str],
    ) -> str:
        name = player_names.get(player_id, player_id)
        role = roles.get(player_id, "peasant")
        role_label = "Landlord" if role == "landlord" else "Peasant"
        active_class = " active" if player_id and player_id == active_player_id else ""
        card_count = len(hand_cards) if isinstance(hand_cards, Sequence) and not isinstance(hand_cards, str) else 0
        hand_html = self._render_hand(hand_cards, position=position)
        action_html = self._render_action(latest_action)
        hand_class = "player-hand-bottom" if position == "bottom" else "player-hand-side"

        return (
            f"<div id=\"doudizhu-{position}\" class=\"doudizhu-player\">"
            f"<div class=\"player-panel{active_class}\">"
            "<div class=\"player-header\">"
            f"<div class=\"player-role role-{html.escape(role)}\">{role_label}</div>"
            f"<div class=\"player-name\">{html.escape(name)}</div>"
            f"<div class=\"player-count\">{card_count} cards</div>"
            "</div>"
            f"<div class=\"player-hand {hand_class}\">{hand_html}</div>"
            "</div>"
            f"<div class=\"played-card-area\">{action_html}</div>"
            "</div>"
        )

    def _render_hand(self, hand_cards: Sequence[str] | str, *, position: str) -> str:
        if not hand_cards:
            return "<div class=\"player-hand-placeholder\">No cards yet</div>"
        if isinstance(hand_cards, str):
            return f"<div class=\"non-card\"><span>{html.escape(hand_cards)}</span></div>"
        cards = self._sort_cards(hand_cards)
        if position == "bottom":
            return self._render_card_list(cards, compact=False)
        return self._render_side_hand(cards)

    def _render_side_hand(self, cards: Sequence[str]) -> str:
        if not cards:
            return "<div class=\"player-hand-placeholder\">No cards</div>"
        up_cards = list(cards[:10])
        down_cards = list(cards[10:])
        up_html = self._render_card_list(up_cards, compact=True)
        down_html = self._render_card_list(down_cards, compact=True) if down_cards else ""
        return f"{up_html}{down_html}"

    def _render_action(self, action: Sequence[str] | str) -> str:
        if not action:
            return "<div class=\"non-card\"><span>No move</span></div>"
        if isinstance(action, str):
            label = "Pass" if action == "pass" else action
            return f"<div class=\"non-card\"><span>{html.escape(label)}</span></div>"
        return self._render_card_list(self._sort_cards(action), compact=True)

    def _render_card_list(self, cards: Sequence[str], *, compact: bool) -> str:
        if not cards:
            return ""
        card_html = []
        for card in cards:
            rank_class, suit_class, rank_text, suit_text = self._translate_card_data(card)
            card_html.append(
                "<li>"
                f"<div class=\"card {html.escape(rank_class)} {html.escape(suit_class)}\">"
                f"<span class=\"rank\">{rank_text}</span>"
                f"<span class=\"suit\">{suit_text}</span>"
                "</div>"
                "</li>"
            )
        return (
            "<div class=\"playingCards\">"
            "<ul class=\"hand\">"
            f"{''.join(card_html)}"
            "</ul>"
            "</div>"
        )

    def _translate_card_data(self, card: str) -> tuple[str, str, str, str]:
        if card == "RJ":
            return ("big", "joker", "RJ", "JOKER")
        if card == "BJ":
            return ("little", "joker", "BJ", "JOKER")
        if len(card) >= 2:
            suit_code = card[0]
            rank_code = card[1]
        else:
            suit_code = ""
            rank_code = card
        rank_text = "10" if rank_code == "T" else rank_code.upper()
        rank_class = f"rank-{rank_text.lower()}"
        suit_class = _SUIT_CLASS.get(suit_code, "")
        suit_text = _SUIT_HTML.get(suit_code, "")
        return (rank_class, suit_class, rank_text, suit_text)

    def _sort_cards(self, cards: Sequence[str]) -> list[str]:
        return sorted(
            [str(card) for card in cards],
            key=lambda card: _DOUDIZHU_DECK_INDEX.get(card, len(_DOUDIZHU_DECK_ORDER)),
        )

    def _render_fallback(self) -> str:
        safe_text = html.escape(self._raw_text)
        last_move = (
            f"<div class=\"non-card\"><span>Last move: {html.escape(self._last_move)}</span></div>"
            if self._last_move
            else ""
        )
        return (
            "<div class=\"doudizhu-shell\">"
            "<div class=\"player-panel\">"
            f"{last_move}"
            f"<pre class=\"doudizhu-board\">{safe_text}</pre>"
            "</div>"
            "</div>"
        )

    def _pretty_board_text(self, board_text: str) -> str:
        if not board_text:
            return ""
        lines = board_text.splitlines()
        formatted: list[str] = []
        idx = 0
        markers = {"Public State:", "Private State:", "Chat Log:", "UI_STATE_JSON:"}
        while idx < len(lines):
            line = lines[idx]
            formatted.append(line)
            if line in markers and idx + 1 < len(lines):
                candidate = lines[idx + 1].strip()
                try:
                    payload = json.loads(candidate)
                except json.JSONDecodeError:
                    idx += 1
                    continue
                pretty = json.dumps(payload, ensure_ascii=False, indent=2)
                formatted.extend(pretty.splitlines())
                idx += 2
                continue
            idx += 1
        return "\n".join(formatted)


__all__ = ["DoudizhuShowdownRenderer", "DOUDIZHU_SHOWDOWN_CSS"]
