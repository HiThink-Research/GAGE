"""Prompt package builder for PettingZoo arena observations."""

from __future__ import annotations

from typing import Any, Optional, Sequence

from gage_eval.role.arena.types import ArenaPromptSpec


class PettingZooPromptBuilder:
    """Builds game-owned prompt payloads for PettingZoo observations."""

    def build(
        self,
        *,
        env_id: str,
        active_player: str,
        last_action: Optional[str],
        legal_moves: Sequence[str],
        mode: str,
        step: int,
        view_text: str,
        metadata: Optional[dict[str, Any]] = None,
    ) -> ArenaPromptSpec:
        """Build a prompt spec for one PettingZoo decision.

        Args:
            env_id: PettingZoo environment id.
            active_player: Player id for this decision.
            last_action: Opponent last action when available.
            legal_moves: Legal action labels/ids for this turn.
            mode: Scheduler mode (for example turn).
            step: Step index at observation time.
            view_text: Human-readable board/status text.
            metadata: Observation metadata snapshot.

        Returns:
            Prompt package consumed by the LLM backend driver.
        """

        legal_items = [str(move) for move in legal_moves]
        legal_hint = ", ".join(legal_items) if legal_items else "none"
        normalized = {item.strip().lower() for item in legal_items}

        instructions = [
            "- Choose exactly one action from the legal moves list.",
            "- Output ONLY the action label or id on the last line.",
            "- Do not output explanations or extra text.",
        ]
        if "fire" in normalized:
            instructions.append("- Prefer FIRE when available.")
        elif "noop" in normalized:
            instructions.append("- Avoid NOOP unless it is the only legal move.")

        instruction = "\n".join(
            [
                f"Active player: {active_player}",
                f"Opponent last move: {last_action or 'None'}",
                "",
                "Environment:",
                str(view_text),
                "",
                "Legal moves:",
                legal_hint,
                "",
                "Instructions:",
                *instructions,
            ]
        )
        renderer_instruction = "\n".join(
            [
                f"Active player: {active_player}",
                "",
                "Environment:",
                str(view_text),
                "",
                "Legal moves:",
                legal_hint,
                "",
                "Instructions:",
                "- Follow the system prompt output policy.",
                "- Return one legal action only.",
            ]
        )
        payload = {
            "player_id": str(active_player),
            "game_type": "pettingzoo",
            "env_id": str(env_id),
            "mode": str(mode),
            "scheduler_mode": str(mode),
            "observation_mode": str(mode),
            "legal_moves": list(legal_items),
            "arena_observation": {
                "view_text": str(view_text),
                "legal_moves": list(legal_items),
                "active_player": str(active_player),
                "last_action": None if last_action is None else str(last_action),
                "metadata": dict(metadata or {}),
                "context": {"mode": str(mode), "step": int(step)},
            },
            "pettingzoo": {
                "env_id": str(env_id),
                "step": int(step),
            },
        }
        return ArenaPromptSpec(
            instruction=instruction,
            renderer_instruction=renderer_instruction,
            payload=payload,
        )


__all__ = ["PettingZooPromptBuilder"]
