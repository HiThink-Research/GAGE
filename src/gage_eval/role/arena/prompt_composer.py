"""Centralized arena prompt composition utilities."""

from __future__ import annotations

import base64
import copy
import io
from dataclasses import dataclass
from typing import Any, Dict, Optional, Sequence

from gage_eval.assets.prompts.renderers import PromptContext, PromptRenderer
from gage_eval.role.arena.types import ArenaObservation


@dataclass(frozen=True)
class TurnMessagesResult:
    """Represents rendered turn messages plus fallback metadata."""

    messages: list[Dict[str, Any]]
    fallback_reason: Optional[str] = None
    render_error_message: Optional[str] = None


class ArenaPromptComposer:
    """Composes prompt text, payloads, and outbound turn messages for arena players."""

    def __init__(
        self,
        *,
        sample: Dict[str, Any],
        prompt_renderer: Optional[PromptRenderer],
        scheduler_mode: Optional[str],
        legal_moves_limit: int,
    ) -> None:
        """Initializes the prompt composer.

        Args:
            sample: Current sample payload.
            prompt_renderer: Optional prompt renderer bound from prompt registry.
            scheduler_mode: Scheduler mode name when configured.
            legal_moves_limit: Maximum legal moves shown in fallback prompt text.
        """

        self._sample = sample
        self._prompt_renderer = prompt_renderer
        self._scheduler_mode = str(scheduler_mode or "").strip() or None
        self._legal_moves_limit = max(0, int(legal_moves_limit))

    def format_observation(
        self,
        observation: ArenaObservation,
        *,
        prefer_renderer_instruction: bool,
    ) -> str:
        """Formats observation text used as the user turn prompt seed.

        Args:
            observation: Current arena observation.
            prefer_renderer_instruction: Whether renderer-specific instruction should be preferred.

        Returns:
            Prompt seed text.
        """

        prompt_spec = self.extract_prompt_spec(observation)
        instruction = prompt_spec.get("instruction")
        renderer_instruction = prompt_spec.get("renderer_instruction")
        if isinstance(instruction, str) and instruction.strip():
            if (
                prefer_renderer_instruction
                and isinstance(renderer_instruction, str)
                and renderer_instruction.strip()
            ):
                return renderer_instruction
            return instruction
        if self.should_use_card_prompt(observation):
            return self.format_card_observation(observation)
        return self.format_grid_observation(observation)

    def build_turn_messages(
        self,
        *,
        player_id: str,
        adapter_id: str,
        base_messages: Sequence[Dict[str, Any]],
        observation: ArenaObservation,
        prompt_text: str,
        image_fragment: Optional[Dict[str, Any]],
        retry_reason: Optional[str] = None,
        last_output: Optional[str] = None,
    ) -> TurnMessagesResult:
        """Builds outbound model messages for one decision turn.

        Args:
            player_id: Arena player id.
            adapter_id: Backend adapter id.
            base_messages: Static sample messages.
            observation: Current observation.
            prompt_text: User prompt seed text.
            image_fragment: Optional image fragment.
            retry_reason: Optional retry reason from previous parse failure.
            last_output: Optional previous model output.

        Returns:
            Rendered messages and fallback metadata.
        """

        legacy_messages = list(base_messages) + [self.build_user_message(prompt_text, image_fragment)]
        if not self._prompt_renderer:
            return TurnMessagesResult(messages=legacy_messages, fallback_reason="missing_prompt_renderer")

        prompt_payload = self.build_prompt_payload(
            player_id=player_id,
            base_messages=base_messages,
            observation=observation,
            prompt_text=prompt_text,
            retry_reason=retry_reason,
            last_output=last_output,
            legacy_messages=legacy_messages,
            has_image=image_fragment is not None,
        )
        context = PromptContext(
            sample=self._sample,
            payload=prompt_payload,
            history=[],
            extras={
                "adapter_id": adapter_id,
                "role_type": "arena_player",
                "player_id": player_id,
            },
        )
        try:
            rendered = self._prompt_renderer.render(context)
        except Exception as exc:
            return TurnMessagesResult(
                messages=legacy_messages,
                fallback_reason="render_error",
                render_error_message=str(exc),
            )

        if rendered.messages is not None:
            messages = self.append_image_fragment(
                rendered.messages,
                image_fragment=image_fragment,
                fallback_text=prompt_text,
            )
            if not self.has_user_message(messages):
                messages.append(self.build_user_message(prompt_text, image_fragment))
            return TurnMessagesResult(messages=messages)

        if rendered.prompt:
            messages = list(base_messages) + [self.build_user_message(rendered.prompt, image_fragment)]
            return TurnMessagesResult(messages=messages)

        return TurnMessagesResult(messages=legacy_messages, fallback_reason="empty_render")

    def build_prompt_payload(
        self,
        *,
        player_id: str,
        base_messages: Sequence[Dict[str, Any]],
        observation: ArenaObservation,
        prompt_text: str,
        retry_reason: Optional[str],
        last_output: Optional[str],
        legacy_messages: Sequence[Dict[str, Any]],
        has_image: bool,
    ) -> Dict[str, Any]:
        """Builds merged payload for prompt template rendering.

        Args:
            player_id: Arena player id.
            base_messages: Static sample messages.
            observation: Current observation.
            prompt_text: User prompt seed text.
            retry_reason: Optional retry reason from previous parse failure.
            last_output: Optional previous model output.
            legacy_messages: Backward-compatible assembled messages.
            has_image: Whether this turn includes an image fragment.

        Returns:
            Prompt payload used by prompt renderer.
        """

        metadata = dict(observation.metadata) if isinstance(observation.metadata, dict) else {}
        context = dict(observation.context) if isinstance(observation.context, dict) else {}
        legal_moves = list(observation.legal_actions_items or [])
        action_schema = metadata.get("action_schema")
        action_schema_config = (
            dict(metadata.get("action_schema_config"))
            if isinstance(metadata.get("action_schema_config"), dict)
            else {}
        )
        hold_ticks = {
            "min": action_schema_config.get("hold_ticks_min"),
            "max": action_schema_config.get("hold_ticks_max"),
            "default": action_schema_config.get("hold_ticks_default"),
        }
        mode = self.resolve_mode(context)
        observation_mode = context.get("mode")
        env_id = metadata.get("env_id")
        payload: Dict[str, Any] = {
            "instruction": prompt_text,
            "prompt_text": prompt_text,
            "messages": list(base_messages),
            "legacy_messages": list(legacy_messages),
            "player_id": player_id,
            "game_type": self.resolve_game_type(metadata),
            "env_id": str(env_id) if env_id else "",
            "retry_reason": str(retry_reason or ""),
            "last_model_output": str(last_output or ""),
            "has_image": bool(has_image),
            "mode": mode,
            "scheduler_mode": mode,
            "observation_mode": str(observation_mode) if observation_mode is not None else "",
            "active_player": observation.active_player,
            "last_action": observation.last_action,
            "legal_moves": legal_moves,
            "action_schema": str(action_schema) if action_schema is not None else "",
            "action_schema_config": action_schema_config,
            "hold_ticks": hold_ticks,
            "arena_observation": {
                "view_text": observation.view_text,
                "legal_moves": legal_moves,
                "active_player": observation.active_player,
                "last_action": observation.last_action,
                "metadata": metadata,
                "context": context,
            },
        }
        prompt_spec = self.extract_prompt_spec(observation)
        prompt_payload = prompt_spec.get("payload")
        if isinstance(prompt_payload, dict) and prompt_payload:
            payload = self.merge_payload_dicts(payload, prompt_payload)

        # STEP 1: Enforce per-request fields derived from runtime state.
        payload["instruction"] = prompt_text
        payload["prompt_text"] = prompt_text
        payload["messages"] = list(base_messages)
        payload["legacy_messages"] = list(legacy_messages)
        payload["player_id"] = player_id
        payload["retry_reason"] = str(retry_reason or "")
        payload["last_model_output"] = str(last_output or "")
        payload["has_image"] = bool(has_image)

        # STEP 2: Keep scheduler mode aligned with the active scheduler.
        payload["mode"] = mode
        payload["scheduler_mode"] = mode
        if observation_mode is not None and str(observation_mode).strip():
            payload["observation_mode"] = str(observation_mode)
        elif "observation_mode" not in payload:
            payload["observation_mode"] = ""

        # STEP 3: Guarantee essential observation fields for prompt templates.
        payload.setdefault("game_type", self.resolve_game_type(metadata))
        payload.setdefault("env_id", str(env_id) if env_id else "")
        payload.setdefault("active_player", observation.active_player)
        payload.setdefault("last_action", observation.last_action)
        payload.setdefault("legal_moves", legal_moves)
        payload.setdefault("action_schema", str(action_schema) if action_schema is not None else "")
        payload.setdefault("action_schema_config", action_schema_config)
        payload.setdefault("hold_ticks", hold_ticks)
        payload.setdefault(
            "arena_observation",
            {
                "view_text": observation.view_text,
                "legal_moves": legal_moves,
                "active_player": observation.active_player,
                "last_action": observation.last_action,
                "metadata": metadata,
                "context": context,
            },
        )
        return payload

    def build_image_fragment(self, observation: ArenaObservation) -> Optional[Dict[str, Any]]:
        """Builds image fragment from observation view payload.

        Args:
            observation: Current arena observation.

        Returns:
            OpenAI-compatible image fragment, or None when unavailable.
        """

        view = observation.view or {}
        image = view.get("image")
        if image is None:
            return None
        data_url = self.resolve_image_data_url(image)
        if not data_url:
            return None
        return {"type": "image_url", "image_url": {"url": data_url}}

    def resolve_image_data_url(self, image: Any) -> Optional[str]:
        """Resolves supported image payloads into a data URL.

        Args:
            image: Observation image payload.

        Returns:
            Data URL string if conversion succeeds, otherwise None.
        """

        if isinstance(image, str):
            if image.startswith("data:"):
                return image
            return None
        if not isinstance(image, dict):
            return None
        if image.get("data_url"):
            return str(image["data_url"])
        if image.get("url"):
            return str(image["url"])
        if image.get("encoding") != "raw_base64":
            return None
        raw_b64 = image.get("data")
        shape = image.get("shape") or []
        dtype = str(image.get("dtype", ""))
        if not raw_b64 or not isinstance(shape, list) or len(shape) < 2:
            return None
        if "uint8" not in dtype:
            return None
        try:
            from PIL import Image
        except ImportError:
            return None
        try:
            raw = base64.b64decode(str(raw_b64))
            height = int(shape[0])
            width = int(shape[1])
            channels = int(shape[2]) if len(shape) > 2 else 1
            if channels == 4:
                mode = "RGBA"
            elif channels == 3:
                mode = "RGB"
            else:
                mode = "L"
            img = Image.frombytes(mode, (width, height), raw)
            buffer = io.BytesIO()
            img.save(buffer, format="JPEG")
            data = base64.b64encode(buffer.getvalue()).decode("ascii")
            return f"data:image/jpeg;base64,{data}"
        except Exception:
            return None

    def truncate_legal_moves(self, legal_moves: Sequence[str]) -> Sequence[str]:
        """Truncates legal moves for fallback prompt text.

        Args:
            legal_moves: Full legal moves list.

        Returns:
            Truncated legal moves sequence.
        """

        if self._legal_moves_limit <= 0:
            return []
        if len(legal_moves) <= self._legal_moves_limit:
            return list(legal_moves)
        return list(legal_moves[: self._legal_moves_limit])

    def resolve_mode(self, observation_context: Dict[str, Any]) -> str:
        """Resolves scheduler mode used in payload fields.

        Args:
            observation_context: Observation context payload.

        Returns:
            Mode string.
        """

        scheduler_mode = self._scheduler_mode
        if scheduler_mode:
            return scheduler_mode
        context_mode = observation_context.get("mode")
        if context_mode is not None and str(context_mode).strip():
            return str(context_mode)
        return "unknown"

    def resolve_game_type(self, observation_metadata: Dict[str, Any]) -> str:
        """Resolves game type field from observation/sample metadata.

        Args:
            observation_metadata: Observation metadata payload.

        Returns:
            Game type string.
        """

        obs_game_type = observation_metadata.get("game_type")
        if isinstance(obs_game_type, str) and obs_game_type.strip():
            return obs_game_type
        sample_metadata = self._sample.get("metadata") if isinstance(self._sample, dict) else {}
        sample_game_type = sample_metadata.get("game_type") if isinstance(sample_metadata, dict) else None
        if isinstance(sample_game_type, str) and sample_game_type.strip():
            return sample_game_type
        env_id = observation_metadata.get("env_id")
        if isinstance(env_id, str) and env_id.strip():
            return env_id
        return "unknown"

    @staticmethod
    def merge_payload_dicts(base: Dict[str, Any], overrides: Dict[str, Any]) -> Dict[str, Any]:
        """Deep merges override payload onto base payload.

        Args:
            base: Base payload.
            overrides: Override payload.

        Returns:
            Merged payload.
        """

        merged: Dict[str, Any] = copy.deepcopy(base)
        for key, value in overrides.items():
            if (
                key in merged
                and isinstance(merged[key], dict)
                and isinstance(value, dict)
            ):
                merged[key] = ArenaPromptComposer.merge_payload_dicts(merged[key], value)
            else:
                merged[key] = copy.deepcopy(value)
        return merged

    @staticmethod
    def append_image_fragment(
        messages: Sequence[Dict[str, Any]],
        *,
        image_fragment: Optional[Dict[str, Any]],
        fallback_text: str,
    ) -> list[Dict[str, Any]]:
        """Appends image fragment to the latest user message when needed.

        Args:
            messages: Renderer-produced messages.
            image_fragment: Optional image fragment.
            fallback_text: Fallback text for synthesized user message.

        Returns:
            Normalized messages with image fragment attached once.
        """

        normalized: list[Dict[str, Any]] = [
            copy.deepcopy(message) for message in messages if isinstance(message, dict)
        ]
        if not image_fragment:
            return normalized

        fragment = copy.deepcopy(image_fragment)
        for idx in range(len(normalized) - 1, -1, -1):
            message = normalized[idx]
            if message.get("role") != "user":
                continue
            content = message.get("content")
            if isinstance(content, list):
                if ArenaPromptComposer.has_image_content(content):
                    return normalized
                content.append(fragment)
                return normalized
            if isinstance(content, str):
                message["content"] = [{"type": "text", "text": content}, fragment]
                return normalized

        normalized.append(ArenaPromptComposer.build_user_message(fallback_text, fragment))
        return normalized

    @staticmethod
    def has_user_message(messages: Sequence[Dict[str, Any]]) -> bool:
        """Checks whether messages include at least one user message.

        Args:
            messages: Message list.

        Returns:
            True if user role exists.
        """

        for message in messages:
            if isinstance(message, dict) and message.get("role") == "user":
                return True
        return False

    @staticmethod
    def has_image_content(content: Sequence[Any]) -> bool:
        """Checks whether message content already includes image content.

        Args:
            content: Message content fragments.

        Returns:
            True if image fragment exists.
        """

        for item in content:
            if not isinstance(item, dict):
                continue
            if item.get("type") == "image_url":
                return True
        return False

    @staticmethod
    def build_user_message(text: str, image_fragment: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Builds a user message with optional image fragment.

        Args:
            text: User text.
            image_fragment: Optional image fragment.

        Returns:
            User message payload.
        """

        content = [{"type": "text", "text": text}]
        if image_fragment:
            content.append(image_fragment)
        return {"role": "user", "content": content}

    def format_grid_observation(self, observation: ArenaObservation) -> str:
        """Formats fallback board-game prompt text.

        Args:
            observation: Current observation.

        Returns:
            Prompt text.
        """

        legal_moves = self.truncate_legal_moves(observation.legal_actions_items)
        legal_hint = ", ".join(legal_moves) if legal_moves else "none"
        active_player = self._format_player_label(observation, observation.active_player)

        lines = [
            f"Active player: {active_player}",
            f"Opponent last move: {observation.last_action or 'First move'}",
            "\nCurrent Board:",
            observation.view_text,
            "\nStatus:",
            f"- Legal moves (truncated): {legal_hint}",
            "\nInstructions:",
            "- Analyze the board.",
            "- Select the best coordinate for your move.",
            "- Output your move as a single coordinate (e.g., 'H8').",
        ]
        return "\n".join(lines)

    def format_card_observation(self, observation: ArenaObservation) -> str:
        """Formats fallback card-game prompt text.

        Args:
            observation: Current observation.

        Returns:
            Prompt text.
        """

        legal_moves = self.truncate_legal_moves(observation.legal_actions_items)
        legal_hint = ", ".join(legal_moves) if legal_moves else "none"
        active_player = self._format_player_label(observation, observation.active_player)
        chat_mode = str(observation.metadata.get("chat_mode", "off")).lower()
        include_chat = chat_mode in {"ai-only", "all"}
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
            f"Opponent last move: {observation.last_action or 'First move'}",
        ]
        team_hint = self.build_team_hint(observation)
        if team_hint:
            lines.append(team_hint)
        lines.extend(
            [
                "\nCurrent State:",
                observation.view_text,
                "\nStatus:",
                f"- Legal moves (preview): {legal_hint}",
                "\nInstructions:",
                *instructions,
            ]
        )
        return "\n".join(lines)

    def should_use_card_prompt(self, observation: ArenaObservation) -> bool:
        """Determines whether fallback prompt should use card-game style.

        Args:
            observation: Current observation.

        Returns:
            True when card-game formatting should be used.
        """

        metadata = self._sample.get("metadata") if isinstance(self._sample, dict) else {}
        game_type = str(metadata.get("game_type", "")).lower()
        if game_type == "doudizhu":
            return True
        if game_type == "vizdoom":
            return False
        if isinstance(observation.metadata.get("public_state"), dict):
            return True
        return "Public State:" in observation.view_text

    @staticmethod
    def extract_prompt_spec(observation: ArenaObservation) -> Dict[str, Any]:
        """Extracts normalized prompt spec from observation.

        Args:
            observation: Current observation.

        Returns:
            Prompt spec dictionary.
        """

        prompt = getattr(observation, "prompt", None)
        if prompt is None:
            return {}
        if isinstance(prompt, dict):
            return {
                "instruction": prompt.get("instruction"),
                "renderer_instruction": prompt.get("renderer_instruction"),
                "payload": prompt.get("payload") if isinstance(prompt.get("payload"), dict) else {},
            }
        payload = getattr(prompt, "payload", {})
        return {
            "instruction": getattr(prompt, "instruction", None),
            "renderer_instruction": getattr(prompt, "renderer_instruction", None),
            "payload": payload if isinstance(payload, dict) else {},
        }

    def build_team_hint(self, observation: ArenaObservation) -> str:
        """Builds team hint string for Doudizhu-like metadata.

        Args:
            observation: Current observation.

        Returns:
            Team hint text when available.
        """

        metadata = observation.metadata if isinstance(observation.metadata, dict) else {}
        public_state = metadata.get("public_state")
        if not isinstance(public_state, dict):
            return ""
        landlord_id = public_state.get("landlord_id")
        if not landlord_id:
            return ""
        player_id = metadata.get("player_id") or observation.active_player
        if not player_id:
            return ""
        player_ids = metadata.get("player_ids")
        if not isinstance(player_ids, list):
            player_ids = []
        if player_id == landlord_id:
            opponents = [str(pid) for pid in player_ids if pid and pid != landlord_id]
            opponent_label = ", ".join(opponents) if opponents else "the two peasants"
            return f"Role: Landlord. Opponents: {opponent_label}."
        teammates = [
            str(pid)
            for pid in player_ids
            if pid and pid not in {player_id, landlord_id}
        ]
        teammate_label = ", ".join(teammates) if teammates else "the other peasant"
        return f"Role: Peasant. Teammate: {teammate_label}. Coordinate to beat landlord {landlord_id}."

    @staticmethod
    def _format_player_label(observation: ArenaObservation, player_id: str) -> str:
        names = observation.metadata.get("player_names")
        if isinstance(names, dict):
            display_name = names.get(player_id)
            if display_name and display_name != player_id:
                return f"{display_name} ({player_id})"
        return player_id
