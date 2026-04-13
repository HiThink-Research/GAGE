"""Prompt package builder for ViZDoom arena observations."""

from __future__ import annotations

from typing import Any, Optional, Sequence

from gage_eval.role.arena.types import ArenaPromptSpec


class ViZDoomPromptBuilder:
    """Builds game-owned prompt payloads for ViZDoom observations."""

    def build(
        self,
        *,
        game_id: str,
        active_player: str,
        legal_actions: Sequence[str],
        action_mapping: dict[str, str],
        tick: int,
        step: int,
        last_reward: Optional[float],
        view_text: str,
        metadata: Optional[dict[str, Any]] = None,
    ) -> ArenaPromptSpec:
        """Build a prompt spec for a ViZDoom decision step.

        Args:
            game_id: Environment game identifier.
            active_player: Player id for the current observation.
            legal_actions: Legal action ids for this step.
            action_mapping: Action-id to action-label mapping.
            tick: Tick index at observation time.
            step: Scheduler step index.
            last_reward: Reward reported by the environment.
            view_text: Human-readable observation text.
            metadata: Observation metadata snapshot.

        Returns:
            Prompt package consumed by the LLM backend driver.
        """

        legal_items = [str(item) for item in legal_actions]
        mapping_items = {str(key): str(value) for key, value in action_mapping.items()}
        legal_hint = ", ".join(legal_items) if legal_items else "none"

        status_lines = [f"- Legal actions: {legal_hint}", f"- Tick: {int(tick)}"]
        if last_reward is not None:
            status_lines.append(f"- Last reward: {float(last_reward)}")

        instruction = "\n".join(
            [
                "You are playing ViZDoom.",
                str(view_text),
                "",
                "Status:",
                *status_lines,
                "",
                "Instructions:",
                "- Choose exactly one action id from the legal actions list.",
                "- Output ONLY the action id as an integer (no extra text).",
            ]
        )
        renderer_instruction = "\n".join(
            [
                "You are playing ViZDoom.",
                str(view_text),
                "",
                "Status:",
                *status_lines,
                "",
                "Instructions:",
                "- Follow the system prompt output policy.",
                "- Return one legal action id only.",
            ]
        )
        payload = {
            "player_id": str(active_player),
            "game_type": "vizdoom",
            "env_id": str(game_id),
            "mode": "tick",
            "scheduler_mode": "tick",
            "observation_mode": "tick",
            "legal_moves": list(legal_items),
            "action_mapping": dict(mapping_items),
            "arena_observation": {
                "view_text": str(view_text),
                "legal_moves": list(legal_items),
                "active_player": str(active_player),
                "last_action": None,
                "metadata": dict(metadata or {}),
                "context": {"mode": "tick", "step": int(step), "tick": int(tick)},
            },
            "vizdoom": {
                "tick": int(tick),
                "step": int(step),
                "last_reward": None if last_reward is None else float(last_reward),
                "action_mapping": dict(mapping_items),
            },
        }
        return ArenaPromptSpec(
            instruction=instruction,
            renderer_instruction=renderer_instruction,
            payload=payload,
        )


__all__ = ["ViZDoomPromptBuilder"]
