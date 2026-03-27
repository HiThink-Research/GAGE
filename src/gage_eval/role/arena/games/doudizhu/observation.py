"""Prompt package builder for Doudizhu arena observations."""

from __future__ import annotations

from typing import Any, Optional, Sequence

from gage_eval.role.arena.types import ArenaPromptSpec


class DoudizhuPromptBuilder:
    """Builds game-owned prompt payloads for Doudizhu observations."""

    def build(
        self,
        *,
        player_id: str,
        player_names: dict[str, str],
        player_ids: Sequence[str],
        public_state: dict[str, Any],
        last_move: Optional[str],
        legal_moves: Sequence[str],
        chat_mode: str,
        mode: str,
        step: int,
        view_text: str,
        metadata: Optional[dict[str, Any]] = None,
    ) -> ArenaPromptSpec:
        """Build a prompt spec for one Doudizhu turn.

        Args:
            player_id: Active player id for this decision.
            player_names: Optional player display-name mapping.
            player_ids: Ordered player ids.
            public_state: Public game state payload.
            last_move: Previous move text.
            legal_moves: Legal action strings.
            chat_mode: Chat mode value from environment config.
            mode: Scheduler mode.
            step: Scheduler step index.
            view_text: Human-readable board/state text.
            metadata: Observation metadata snapshot.

        Returns:
            Prompt package consumed by the LLM backend driver.
        """

        legal_items = [str(move) for move in legal_moves]
        legal_hint = ", ".join(legal_items) if legal_items else "none"
        normalized_chat_mode = str(chat_mode or "off").lower()
        include_chat = normalized_chat_mode in {"ai-only", "all"}
        active_player = _format_player_label(player_id, player_names)
        team_hint = _build_team_hint(
            player_id=player_id,
            player_ids=player_ids,
            public_state=public_state,
        )

        instructions = [
            "- Choose exactly one legal action string from the legal moves.",
        ]
        if include_chat:
            instructions.extend(
                [
                    "- Include a short table-talk line every turn.",
                    '- Output JSON: {"action": "<action>", "chat": "<short line>"}',
                ]
            )
        else:
            instructions.append("- Output the action string only.")

        lines = [
            f"Active player: {active_player}",
            f"Opponent last move: {last_move or 'First move'}",
        ]
        if team_hint:
            lines.append(team_hint)
        lines.extend(
            [
                "",
                "Current State:",
                str(view_text),
                "",
                "Status:",
                f"- Legal moves (preview): {legal_hint}",
                "",
                "Instructions:",
                *instructions,
            ]
        )
        instruction = "\n".join(lines)

        renderer_instruction = "\n".join(
            [
                f"Active player: {active_player}",
                "",
                "Current State:",
                str(view_text),
                "",
                "Status:",
                f"- Legal moves (preview): {legal_hint}",
                "",
                "Instructions:",
                "- Follow the system prompt output policy.",
                "- Return one legal action only.",
            ]
        )

        payload = {
            "player_id": str(player_id),
            "game_type": "doudizhu",
            "env_id": "doudizhu",
            "mode": str(mode),
            "scheduler_mode": str(mode),
            "observation_mode": str(mode),
            "legal_moves": list(legal_items),
            "chat_mode": normalized_chat_mode,
            "include_chat": bool(include_chat),
            "team_hint": str(team_hint),
            "arena_observation": {
                "view_text": str(view_text),
                "legal_moves": list(legal_items),
                "active_player": str(player_id),
                "last_action": None if last_move is None else str(last_move),
                "metadata": dict(metadata or {}),
                "context": {"mode": str(mode), "step": int(step)},
            },
            "doudizhu": {
                "landlord_id": public_state.get("landlord_id"),
                "chat_mode": normalized_chat_mode,
                "include_chat": bool(include_chat),
            },
        }
        return ArenaPromptSpec(
            instruction=instruction,
            renderer_instruction=renderer_instruction,
            payload=payload,
        )


def _format_player_label(player_id: str, player_names: dict[str, str]) -> str:
    display_name = player_names.get(player_id)
    if display_name and display_name != player_id:
        return f"{display_name} ({player_id})"
    return player_id


def _build_team_hint(
    *,
    player_id: str,
    player_ids: Sequence[str],
    public_state: dict[str, Any],
) -> str:
    landlord_id = public_state.get("landlord_id")
    if not landlord_id:
        return ""
    landlord = str(landlord_id)
    if player_id == landlord:
        opponents = [str(pid) for pid in player_ids if pid and pid != landlord]
        opponent_label = ", ".join(opponents) if opponents else "the two peasants"
        return f"Role: Landlord. Opponents: {opponent_label}."
    teammates = [
        str(pid)
        for pid in player_ids
        if pid and pid not in {player_id, landlord}
    ]
    teammate_label = ", ".join(teammates) if teammates else "the other peasant"
    return f"Role: Peasant. Teammate: {teammate_label}. Coordinate to beat landlord {landlord}."


__all__ = ["DoudizhuPromptBuilder"]
