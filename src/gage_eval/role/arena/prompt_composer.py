"""Centralized arena prompt composition utilities."""

from __future__ import annotations

import base64
import copy
import io
from collections import deque
from dataclasses import dataclass
from typing import Any, Dict, Optional, Sequence

from loguru import logger

from gage_eval.assets.prompts.renderers import PromptContext, PromptRenderer
from gage_eval.role.arena.types import ArenaAction, ArenaObservation

_DEFAULT_SCHEME_ID = "S3_text_image_current"
_DEFAULT_HISTORY_LEN = 4
_DEFAULT_DELTA_KEY_LIMIT = 6
_DEFAULT_TELEMETRY_LIMIT = 12
_SUPPORTED_SCHEME_IDS = {
    "S1_rich_text_only",
    "S2_image_only_current",
    "S3_text_image_current",
    "S5_image_compact_state",
    "S6_text_image_action_hist",
    "S6_v2_text_image_action_outcome_hist",
    "S7_text_image_delta_summary",
}
_IMAGE_ENABLED_SCHEME_IDS = {
    "S2_image_only_current",
    "S3_text_image_current",
    "S5_image_compact_state",
    "S6_text_image_action_hist",
    "S6_v2_text_image_action_outcome_hist",
    "S7_text_image_delta_summary",
}
_COMPACT_TEXT_SCHEME_IDS = {"S5_image_compact_state"}
_ACTION_HISTORY_SCHEME_IDS = {"S6_text_image_action_hist"}
_ACTION_OUTCOME_HISTORY_SCHEME_IDS = {"S6_v2_text_image_action_outcome_hist"}
_DELTA_SUMMARY_SCHEME_IDS = {"S7_text_image_delta_summary"}
_MINIMAL_TEXT_SCHEME_IDS = {"S2_image_only_current"}
_VIZDOOM_ACTION_MAPPING_HINT = "1=ATTACK, 2=MOVE_LEFT, 3=MOVE_RIGHT"
_VIZDOOM_PRIORITY_KEYS = (
    "HEALTH",
    "ARMOR",
    "AMMO2",
    "SELECTED_WEAPON",
    "SELECTED_WEAPON_AMMO",
    "FRAGCOUNT",
    "KILLCOUNT",
    "DAMAGECOUNT",
    "HITCOUNT",
    "POSITION_X",
    "POSITION_Y",
    "ANGLE",
    "VELOCITY_X",
    "VELOCITY_Y",
    "t",
    "reward",
)
_VIZDOOM_S3_TELEMETRY_KEYS = (
    "HEALTH",
    "ARMOR",
    "AMMO2",
    "SELECTED_WEAPON",
    "SELECTED_WEAPON_AMMO",
    "FRAGCOUNT",
    "KILLCOUNT",
    "DAMAGECOUNT",
    "HITCOUNT",
    "POSITION_X",
    "POSITION_Y",
    "ANGLE",
    "VELOCITY_X",
    "VELOCITY_Y",
    "p0_health",
    "p1_health",
    "p0_frag",
    "p1_frag",
)
_VIZDOOM_OUTCOME_PRIORITY_KEYS = (
    "p0_health",
    "p1_health",
    "p0_frag",
    "p1_frag",
    "FRAGCOUNT",
    "KILLCOUNT",
    "HITCOUNT",
    "DAMAGECOUNT",
    "HEALTH",
    "reward",
    "POSITION_X",
    "POSITION_Y",
    "ANGLE",
    "t",
)
_PETTINGZOO_PRIORITY_KEYS = (
    "step",
    "reward",
    "termination",
    "truncation",
)


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
        scheme_id: Optional[str] = None,
        scheme_params: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Initializes the prompt composer.

        Args:
            sample: Current sample payload.
            prompt_renderer: Optional prompt renderer bound from prompt registry.
            scheduler_mode: Scheduler mode name when configured.
            legal_moves_limit: Maximum legal moves shown in fallback prompt text.
            scheme_id: Observation packaging scheme id for multimodal strategies.
            scheme_params: Extra parameters used by scheme-specific prompt composition.
        """

        self._sample = sample
        self._prompt_renderer = prompt_renderer
        self._scheduler_mode = str(scheduler_mode or "").strip() or None
        self._legal_moves_limit = max(0, int(legal_moves_limit))
        self._scheme_params = dict(scheme_params or {})
        self._scheme_id = self._resolve_scheme_id(scheme_id)
        self._scheme_explicitly_configured = bool(str(scheme_id or "").strip())
        self._action_history_len = self._coerce_positive_int(
            self._scheme_params.get("action_history_len"),
            default=_DEFAULT_HISTORY_LEN,
        )
        self._history_limit = max(
            self._action_history_len,
            self._coerce_positive_int(
                self._scheme_params.get("history_limit"),
                default=_DEFAULT_HISTORY_LEN,
            ),
        )
        self._delta_key_limit = self._coerce_positive_int(
            self._scheme_params.get("delta_key_limit"),
            default=_DEFAULT_DELTA_KEY_LIMIT,
        )
        self._telemetry_limit = self._coerce_positive_int(
            self._scheme_params.get("telemetry_limit"),
            default=_DEFAULT_TELEMETRY_LIMIT,
        )
        self._interaction_history: deque[dict[str, Any]] = deque(maxlen=self._history_limit)
        self._interaction_index = 0

    @property
    def scheme_id(self) -> str:
        """Returns the resolved scheme id."""

        return self._scheme_id

    @property
    def scheme_explicitly_configured(self) -> bool:
        """Returns whether scheme id was explicitly configured by caller."""

        return self._scheme_explicitly_configured

    @property
    def action_history_len(self) -> int:
        """Returns the configured action history length."""

        return self._action_history_len

    @property
    def delta_key_limit(self) -> int:
        """Returns the configured delta summary key limit."""

        return self._delta_key_limit

    @property
    def telemetry_limit(self) -> int:
        """Returns the configured telemetry line limit."""

        return self._telemetry_limit

    def scheme_supports_image(self) -> bool:
        """Returns whether current scheme is expected to include an image."""

        return self._scheme_id in _IMAGE_ENABLED_SCHEME_IDS

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

        if self._should_prefer_scheme_prompt(observation):
            if self.is_vizdoom_observation(observation):
                return self.format_vizdoom_observation(observation)
            if self.should_use_pettingzoo_prompt(observation):
                return self.format_pettingzoo_observation(observation)

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
        if self.is_vizdoom_observation(observation):
            return self.format_vizdoom_observation(observation)
        if self.should_use_pettingzoo_prompt(observation):
            return self.format_pettingzoo_observation(observation)
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
            "scheme_id": self._scheme_id,
            "scheme_supports_image": self.scheme_supports_image(),
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
            "vizdoom_strategy": self._build_empty_vizdoom_strategy_payload(),
        }
        if self.is_vizdoom_observation(observation):
            payload["vizdoom_strategy"] = self._build_vizdoom_strategy_payload(observation)
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
        payload["scheme_id"] = self._scheme_id
        payload["scheme_supports_image"] = self.scheme_supports_image()
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
        payload.setdefault("vizdoom_strategy", self._build_empty_vizdoom_strategy_payload())
        return payload

    def build_image_fragment(self, observation: ArenaObservation) -> Optional[Dict[str, Any]]:
        """Builds image fragment from observation view payload.

        Args:
            observation: Current arena observation.

        Returns:
            OpenAI-compatible image fragment, or None when unavailable.
        """

        if self._scheme_id not in _IMAGE_ENABLED_SCHEME_IDS:
            return None
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

        sample_metadata = self._sample.get("metadata") if isinstance(self._sample, dict) else {}
        metadata = sample_metadata if isinstance(sample_metadata, dict) else {}
        game_type = str(metadata.get("game_type", "")).lower()
        if game_type == "doudizhu":
            return True
        if game_type == "vizdoom":
            return False
        if isinstance(observation.metadata.get("public_state"), dict):
            return True
        return "Public State:" in observation.view_text

    def should_use_pettingzoo_prompt(self, observation: ArenaObservation) -> bool:
        """Determines whether prompt should use PettingZoo-style formatting."""

        metadata = observation.metadata if isinstance(observation.metadata, dict) else {}
        env_id = str(metadata.get("env_id", "")).lower()
        return "pettingzoo" in env_id

    def is_vizdoom_observation(self, observation: ArenaObservation) -> bool:
        """Determines whether current observation belongs to ViZDoom."""

        metadata = observation.metadata if isinstance(observation.metadata, dict) else {}
        obs_game_type = str(metadata.get("game_type", "")).lower()
        if obs_game_type == "vizdoom":
            return True
        sample_metadata = self._sample.get("metadata") if isinstance(self._sample, dict) else {}
        sample_meta = sample_metadata if isinstance(sample_metadata, dict) else {}
        sample_game_type = str(sample_meta.get("game_type", "")).lower()
        return sample_game_type == "vizdoom"

    def format_vizdoom_observation(self, observation: ArenaObservation) -> str:
        """Formats ViZDoom prompt text according to the configured scheme."""

        strategy_payload = self._build_vizdoom_strategy_payload(observation)
        legal_hint = str(strategy_payload.get("legal_hint") or "none")
        perspective_block = str(strategy_payload.get("perspective_block") or "unknown")
        state_block = str(strategy_payload.get("state_block") or "")
        action_history_block = str(strategy_payload.get("action_history_block") or "")
        outcome_history_block = str(strategy_payload.get("action_outcome_history_block") or "")
        delta_block = str(strategy_payload.get("delta_summary_block") or "")

        if self._prompt_renderer and self._scheme_explicitly_configured:
            sections = [
                "Observation snapshot for the current ViZDoom turn.",
                f"Perspective:\n{perspective_block}",
                f"Legal actions:\n{legal_hint}",
            ]
            if state_block:
                sections.append(f"Current state:\n{state_block}")
            if outcome_history_block:
                sections.append(
                    "Recent action and outcome history (oldest to newest):\n"
                    f"{outcome_history_block}"
                )
            elif action_history_block:
                sections.append(f"Recent action history (oldest to newest):\n{action_history_block}")
            if delta_block:
                sections.append(f"Delta summary from the previous observation:\n{delta_block}")
            return "\n\n".join([section for section in sections if section.strip()])

        if self._scheme_id in _MINIMAL_TEXT_SCHEME_IDS:
            sections = [
                "You are playing ViZDoom.",
                f"Perspective:\n{perspective_block}",
                f"Legal actions:\n{legal_hint}",
                "Instructions:\n"
                "- Goal: Eliminate the opposing player to win the match.\n"
                f"- Action mapping: {_VIZDOOM_ACTION_MAPPING_HINT}.\n"
                "- Move until you are facing the enemy before firing to improve hit accuracy.\n"
                "- A shot can only hit when the enemy is vertically aligned with your weapon barrel.\n"
                "- If the enemy is on the left side of your view, turn or move left to center them before firing.\n"
                "- If the enemy is on the right side of your view, turn or move right to center them before firing.\n"
                "- Avoid firing when the enemy is far from the center of your view.\n"
                "- Choose exactly one action id from the legal actions list.\n"
                "- Output exactly two lines:\n"
                "  Action: <action_id>\n"
                "  Reason: <short reason>",
            ]
            return "\n\n".join([section for section in sections if section.strip()])

        sections = [
            "You are playing ViZDoom.",
            f"Perspective:\n{perspective_block}",
            f"Current state:\n{state_block}",
            f"Legal actions:\n{legal_hint}",
            f"Action mapping:\n{_VIZDOOM_ACTION_MAPPING_HINT}",
        ]
        if outcome_history_block:
            sections.append(
                f"Recent action and outcome history (oldest to newest):\n{outcome_history_block}"
            )
        elif action_history_block:
            sections.append(f"Recent action history (oldest to newest):\n{action_history_block}")
        if delta_block:
            sections.append(f"Delta summary from the previous observation:\n{delta_block}")
        sections.append(
            "Instructions:\n"
            "- Goal: Eliminate the opposing player to win the match.\n"
            "- Move until you are facing the enemy before firing to improve hit accuracy.\n"
            "- A shot can only hit when the enemy is vertically aligned with your weapon barrel.\n"
            "- If the enemy is on the left side of your view, turn or move left to center them before firing.\n"
            "- If the enemy is on the right side of your view, turn or move right to center them before firing.\n"
            "- Avoid firing when the enemy is far from the center of your view.\n"
            "- Choose exactly one action id from the legal actions list.\n"
            "- Output exactly two lines:\n"
            "  Action: <action_id>\n"
            "  Reason: <short reason>"
        )
        return "\n\n".join([section for section in sections if section.strip()])

    @staticmethod
    def _build_empty_vizdoom_strategy_payload() -> Dict[str, Any]:
        """Builds an empty ViZDoom strategy payload for strict template rendering."""

        return {
            "scheme_id": "",
            "legal_hint": "none",
            "action_mapping_hint": _VIZDOOM_ACTION_MAPPING_HINT,
            "perspective_block": "",
            "state_block": "",
            "action_history_block": "",
            "action_outcome_history_block": "",
            "delta_summary_block": "",
            "has_state_block": False,
            "has_action_history_block": False,
            "has_action_outcome_history_block": False,
            "has_delta_summary_block": False,
        }

    def _build_vizdoom_strategy_payload(self, observation: ArenaObservation) -> Dict[str, Any]:
        """Builds renderer-facing strategy payload for ViZDoom prompts."""

        legal_moves = self.truncate_legal_moves(observation.legal_actions_items)
        legal_hint = ", ".join(legal_moves) if legal_moves else "none"
        perspective_block = self._build_vizdoom_perspective_block(observation)
        state_block = (
            ""
            if self._scheme_id in _MINIMAL_TEXT_SCHEME_IDS
            else self._build_vizdoom_state_block(observation)
        )

        action_history_block = ""
        if self._scheme_id in _ACTION_HISTORY_SCHEME_IDS:
            action_history_block = self._build_action_history_block()

        action_outcome_history_block = ""
        if self._scheme_id in _ACTION_OUTCOME_HISTORY_SCHEME_IDS:
            action_outcome_history_block = self._build_action_outcome_history_block(observation)

        delta_summary_block = ""
        if self._scheme_id in _DELTA_SUMMARY_SCHEME_IDS:
            delta_summary_block = self._build_delta_summary_block(observation)

        return {
            "scheme_id": self._scheme_id,
            "legal_hint": legal_hint,
            "action_mapping_hint": _VIZDOOM_ACTION_MAPPING_HINT,
            "perspective_block": perspective_block,
            "state_block": state_block,
            "action_history_block": action_history_block,
            "action_outcome_history_block": action_outcome_history_block,
            "delta_summary_block": delta_summary_block,
            "has_state_block": bool(state_block),
            "has_action_history_block": bool(action_history_block),
            "has_action_outcome_history_block": bool(action_outcome_history_block),
            "has_delta_summary_block": bool(delta_summary_block),
        }

    def format_pettingzoo_observation(self, observation: ArenaObservation) -> str:
        """Formats PettingZoo prompt text according to the configured scheme."""

        legal_moves = self.truncate_legal_moves(observation.legal_actions_items)
        legal_hint = ", ".join(legal_moves) if legal_moves else "none"
        active_player = self._format_player_label(observation, observation.active_player)
        last_action = observation.last_action or "None"

        if self._scheme_id in _MINIMAL_TEXT_SCHEME_IDS:
            return self._format_minimal_visual_prompt(
                game_name="PettingZoo arena game",
                legal_hint=legal_hint,
                instructions=(
                    "- Choose exactly one action from the legal moves list.",
                    "- Output ONLY the action label or id on the last line.",
                ),
            )

        instructions = [
            "- Choose exactly one action from the legal moves list.",
            "- Output ONLY the action label or id on the last line.",
            "- Do not output explanations or extra text.",
        ]
        if any(self._normalize_move(move) == "fire" for move in legal_moves):
            instructions.append("- Prefer FIRE when available.")
        elif any(self._normalize_move(move) == "noop" for move in legal_moves):
            instructions.append("- Avoid NOOP unless it is the only legal move.")

        state_block = self._build_pettingzoo_state_block(observation)
        sections = [
            f"Active player: {active_player}",
            f"Opponent last move: {last_action}",
            f"Environment:\n{state_block}",
            f"Legal moves:\n{legal_hint}",
        ]
        outcome_history_block = self._build_action_outcome_history_block(observation)
        if self._scheme_id in _ACTION_OUTCOME_HISTORY_SCHEME_IDS and outcome_history_block:
            sections.append(
                f"Recent action and outcome history (oldest to newest):\n{outcome_history_block}"
            )
        action_history_block = self._build_action_history_block()
        if (
            self._scheme_id in _ACTION_HISTORY_SCHEME_IDS
            and self._scheme_id not in _ACTION_OUTCOME_HISTORY_SCHEME_IDS
            and action_history_block
        ):
            sections.append(f"Recent action history (oldest to newest):\n{action_history_block}")
        delta_block = self._build_delta_summary_block(observation)
        if self._scheme_id in _DELTA_SUMMARY_SCHEME_IDS and delta_block:
            sections.append(f"Delta summary from the previous observation:\n{delta_block}")
        sections.append("Instructions:\n" + "\n".join(instructions))
        return "\n\n".join([section for section in sections if section.strip()])

    def remember_interaction(self, observation: ArenaObservation, action: ArenaAction) -> None:
        """Stores the latest observation/action pair for temporal prompts."""

        snapshot = {
            "sequence": self._interaction_index,
            "step": self.resolve_observation_step(observation),
            "view_text": observation.view_text,
            "numeric_state": self._extract_numeric_state(
                observation,
                priority_keys=_VIZDOOM_PRIORITY_KEYS
                if self.is_vizdoom_observation(observation)
                else _PETTINGZOO_PRIORITY_KEYS,
            ),
            "action": str(action.move or "").strip(),
        }
        self._interaction_index += 1
        self._interaction_history.append(snapshot)

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
    def resolve_observation_step(observation: ArenaObservation) -> Optional[int]:
        """Extracts the current step index from observation context or metadata."""

        context = observation.context if isinstance(observation.context, dict) else {}
        metadata = observation.metadata if isinstance(observation.metadata, dict) else {}
        for key in ("step", "tick"):
            if key in context:
                try:
                    return int(context[key])
                except (TypeError, ValueError):
                    pass
        for key in ("t", "step"):
            if key in metadata:
                try:
                    return int(metadata[key])
                except (TypeError, ValueError):
                    pass
        return None

    def _should_prefer_scheme_prompt(self, observation: ArenaObservation) -> bool:
        if not self._scheme_explicitly_configured:
            return False
        return self.is_vizdoom_observation(observation) or self.should_use_pettingzoo_prompt(observation)

    def _build_vizdoom_perspective_block(self, observation: ArenaObservation) -> str:
        """Builds explicit self/opponent identity and stat-key mapping for ViZDoom."""

        player_id, opponent_id, self_index, opponent_index = self._resolve_vizdoom_player_context(
            observation
        )
        telemetry = self._extract_numeric_state(observation, priority_keys=_VIZDOOM_PRIORITY_KEYS)
        lines = [f"- You control player_id={player_id}."]
        if opponent_id:
            lines.append(f"- Opponent player_id={opponent_id}.")
        lines.append(
            "- Your POV-local telemetry keys HEALTH/FRAGCOUNT/KILLCOUNT/HITCOUNT/DAMAGECOUNT "
            "describe your own state."
        )
        if self_index is not None:
            lines.append(f"- Your global duel keys: p{self_index}_health, p{self_index}_frag.")
        if opponent_index is not None:
            lines.append(
                f"- Opponent global duel keys: p{opponent_index}_health, p{opponent_index}_frag."
            )
        if self_index is None and opponent_index is None:
            lines.append(
                "- Global duel keys p0_health/p1_health and p0_frag/p1_frag track both players."
            )

        key_values: list[str] = []
        if self_index is not None:
            self_health_key = f"p{self_index}_health"
            self_frag_key = f"p{self_index}_frag"
            self_health_value = telemetry.get(self_health_key)
            self_frag_value = telemetry.get(self_frag_key)
            if self_health_value is not None:
                key_values.append(
                    f"self_health={_format_number(self_health_value)} ({self_health_key})"
                )
            if self_frag_value is not None:
                key_values.append(f"self_frag={_format_number(self_frag_value)} ({self_frag_key})")
        if opponent_index is not None:
            opponent_health_key = f"p{opponent_index}_health"
            opponent_frag_key = f"p{opponent_index}_frag"
            opponent_health_value = telemetry.get(opponent_health_key)
            opponent_frag_value = telemetry.get(opponent_frag_key)
            if opponent_health_value is not None:
                key_values.append(
                    f"opponent_health={_format_number(opponent_health_value)} "
                    f"({opponent_health_key})"
                )
            if opponent_frag_value is not None:
                key_values.append(
                    f"opponent_frag={_format_number(opponent_frag_value)} ({opponent_frag_key})"
                )
        if key_values:
            lines.append(f"- Current duel stats: {', '.join(key_values)}.")
        return "\n".join(lines)

    def _resolve_vizdoom_player_context(
        self,
        observation: ArenaObservation,
    ) -> tuple[str, Optional[str], Optional[int], Optional[int]]:
        """Resolves self/opponent ids and indexes for ViZDoom prompt rendering."""

        metadata = observation.metadata if isinstance(observation.metadata, dict) else {}
        player_id = str(
            metadata.get("player_id") or observation.active_player or "p0"
        ).strip()
        if not player_id:
            player_id = "p0"

        raw_player_ids = metadata.get("player_ids")
        player_ids = [str(item).strip() for item in raw_player_ids] if isinstance(raw_player_ids, list) else []
        player_ids = [item for item in player_ids if item]
        self_index = self._resolve_vizdoom_player_index(player_id=player_id, player_ids=player_ids)
        if self_index is None:
            self_index = self._infer_vizdoom_player_index(player_id)

        opponent_id: Optional[str] = None
        opponent_index: Optional[int] = None
        for idx, candidate in enumerate(player_ids):
            if candidate == player_id:
                continue
            opponent_id = candidate
            opponent_index = idx
            break
        if opponent_index is None and self_index is not None:
            if self_index == 0:
                opponent_index = 1
                opponent_id = opponent_id or "p1"
            elif self_index == 1:
                opponent_index = 0
                opponent_id = opponent_id or "p0"
        return player_id, opponent_id, self_index, opponent_index

    @staticmethod
    def _resolve_vizdoom_player_index(
        *,
        player_id: str,
        player_ids: Sequence[str],
    ) -> Optional[int]:
        """Resolves player index from observation player_ids list when possible."""

        for idx, candidate in enumerate(player_ids):
            if str(candidate).strip() == str(player_id).strip():
                return int(idx)
        return None

    @staticmethod
    def _infer_vizdoom_player_index(player_id: str) -> Optional[int]:
        """Infers player index from canonical ids like p0/p1."""

        token = str(player_id).strip().lower()
        if len(token) >= 2 and token.startswith("p") and token[1:].isdigit():
            try:
                return int(token[1:])
            except (TypeError, ValueError):
                return None
        return None

    def _build_vizdoom_state_block(self, observation: ArenaObservation) -> str:
        """Builds the state section for a ViZDoom observation."""

        metadata = observation.metadata if isinstance(observation.metadata, dict) else {}
        telemetry = self._extract_numeric_state(
            observation,
            priority_keys=_VIZDOOM_PRIORITY_KEYS,
        )
        if self._scheme_id == "S3_text_image_current":
            telemetry = self._select_telemetry_keys(
                telemetry,
                selected_keys=_VIZDOOM_S3_TELEMETRY_KEYS,
            )
        if self._scheme_id in _COMPACT_TEXT_SCHEME_IDS:
            return self._render_telemetry_block(
                telemetry,
                fallback_lines=(
                    f"Tick text: {observation.view_text}",
                    f"Last reward: {metadata.get('reward', 'unknown')}",
                ),
            )

        sections = [f"Observation text:\n{observation.view_text}"]
        telemetry_block = self._render_telemetry_block(telemetry)
        if telemetry_block:
            sections.append(f"Telemetry:\n{telemetry_block}")
        return "\n\n".join(sections)

    def _build_pettingzoo_state_block(self, observation: ArenaObservation) -> str:
        """Builds the state section for a PettingZoo observation."""

        telemetry = self._extract_numeric_state(
            observation,
            priority_keys=_PETTINGZOO_PRIORITY_KEYS,
        )
        if self._scheme_id in _COMPACT_TEXT_SCHEME_IDS:
            return self._render_telemetry_block(
                telemetry,
                fallback_lines=(observation.view_text,),
            )
        sections = [observation.view_text]
        telemetry_block = self._render_telemetry_block(telemetry)
        if telemetry_block:
            sections.append(f"Telemetry:\n{telemetry_block}")
        return "\n\n".join([section for section in sections if section.strip()])

    @staticmethod
    def _format_minimal_visual_prompt(
        *,
        game_name: str,
        legal_hint: str,
        instructions: Sequence[str],
    ) -> str:
        """Builds the minimal text scaffold used by image-only schemes."""

        sections = [
            f"You are playing {game_name}.",
            f"Legal actions:\n{legal_hint}",
            "Instructions:\n" + "\n".join([str(item) for item in instructions]),
        ]
        return "\n\n".join([section for section in sections if section.strip()])

    def _build_action_history_block(self) -> str:
        """Renders the recent action history block for temporal ablations."""

        lines: list[str] = []
        recent_entries = [
            entry
            for entry in list(self._interaction_history)
            if str(entry.get("action") or "").strip()
        ][-self._action_history_len :]
        for entry in recent_entries:
            step_label = entry.get("step")
            prefix = f"step={step_label}" if step_label is not None else f"seq={entry['sequence']}"
            lines.append(f"- {prefix}: {entry['action']}")
        return "\n".join(lines)

    def _build_action_outcome_history_block(self, observation: ArenaObservation) -> str:
        """Renders recent action-result pairs to expose causal feedback."""

        if not self._interaction_history:
            return ""

        # STEP 1: Resolve candidate entries and the current reference state.
        history_entries = [
            entry
            for entry in list(self._interaction_history)
            if str(entry.get("action") or "").strip()
        ]
        if not history_entries:
            return ""

        current_state = self._extract_numeric_state(
            observation,
            priority_keys=_VIZDOOM_PRIORITY_KEYS
            if self.is_vizdoom_observation(observation)
            else _PETTINGZOO_PRIORITY_KEYS,
        )
        current_step = self.resolve_observation_step(observation)
        recent_entries = history_entries[-self._action_history_len :]
        start_index = len(history_entries) - len(recent_entries)

        # STEP 2: Pair each historical action with the state change that followed it.
        lines: list[str] = []
        for offset, entry in enumerate(recent_entries):
            history_index = start_index + offset
            action = str(entry.get("action") or "").strip()
            if not action:
                continue
            from_state = entry.get("numeric_state") or {}
            to_state: dict[str, Any]
            to_step = current_step
            if history_index + 1 < len(history_entries):
                next_entry = history_entries[history_index + 1]
                to_state = next_entry.get("numeric_state") or {}
                to_step = next_entry.get("step")
            else:
                to_state = current_state
            outcome = self._summarize_action_outcome(
                from_state=from_state,
                to_state=to_state,
                is_vizdoom=self.is_vizdoom_observation(observation),
            )
            step_label = entry.get("step")
            prefix = f"step={step_label}" if step_label is not None else f"seq={entry['sequence']}"
            if to_step is None:
                lines.append(f"- {prefix}: action={action} -> outcome: {outcome}")
            else:
                lines.append(
                    f"- {prefix}: action={action} -> outcome@step={to_step}: {outcome}"
                )
        return "\n".join(lines)

    def _summarize_action_outcome(
        self,
        *,
        from_state: dict[str, Any],
        to_state: dict[str, Any],
        is_vizdoom: bool,
    ) -> str:
        """Builds one compact outcome sentence from state deltas."""

        priority_keys = (
            _VIZDOOM_OUTCOME_PRIORITY_KEYS if is_vizdoom else _PETTINGZOO_PRIORITY_KEYS
        )
        changed_items: list[str] = []
        seen: set[str] = set()
        for key in list(priority_keys) + sorted(set(from_state.keys()) | set(to_state.keys())):
            if key in seen:
                continue
            seen.add(str(key))
            prev_value = _coerce_numeric(from_state.get(key))
            next_value = _coerce_numeric(to_state.get(key))
            if prev_value is None or next_value is None:
                continue
            diff = next_value - prev_value
            if diff == 0.0:
                continue
            sign = "+" if diff > 0 else ""
            changed_items.append(f"{key}:{sign}{_format_number(diff)}")
            if len(changed_items) >= self._delta_key_limit:
                break
        if not changed_items:
            return "no notable tracked change"
        return ", ".join(changed_items)

    def _build_delta_summary_block(self, observation: ArenaObservation) -> str:
        """Renders numeric state deltas relative to the previous observation."""

        if not self._interaction_history:
            return ""

        # STEP 1: Resolve the previous snapshot and current numeric state.
        previous = self._interaction_history[-1]
        current_state = self._extract_numeric_state(
            observation,
            priority_keys=_VIZDOOM_PRIORITY_KEYS
            if self.is_vizdoom_observation(observation)
            else _PETTINGZOO_PRIORITY_KEYS,
        )

        # STEP 2: Build a compact human-readable delta summary.
        lines: list[str] = []
        last_action = str(previous.get("action") or "").strip()
        if last_action:
            lines.append(f"- Previous action: {last_action}")
        delta = _numeric_delta(
            previous.get("numeric_state") or {},
            current_state,
            limit=self._delta_key_limit,
        )
        for key, diff in delta.items():
            sign = "+" if diff > 0 else ""
            lines.append(f"- {key}: {sign}{_format_number(diff)}")
        if not lines and observation.view_text != str(previous.get("view_text") or ""):
            lines.append("- Observation text changed.")
        return "\n".join(lines)

    def _extract_numeric_state(
        self,
        observation: ArenaObservation,
        *,
        priority_keys: Sequence[str],
    ) -> dict[str, float]:
        """Extracts compact numeric telemetry from the observation."""

        metadata = observation.metadata if isinstance(observation.metadata, dict) else {}
        view = observation.view if isinstance(observation.view, dict) else {}
        state: dict[str, float] = {}

        self._collect_numeric_pairs(
            state,
            {
                "step": self.resolve_observation_step(observation),
                "t": metadata.get("t"),
                "reward": metadata.get("reward"),
                "termination": metadata.get("termination"),
                "truncation": metadata.get("truncation"),
                "done": metadata.get("done"),
            },
        )
        raw_obs = metadata.get("raw_obs")
        if isinstance(raw_obs, dict):
            self._collect_numeric_pairs(state, raw_obs)
        vector_obs = view.get("vector")
        if isinstance(vector_obs, dict):
            self._collect_numeric_pairs(state, vector_obs)
        info = metadata.get("info")
        if isinstance(info, dict):
            self._collect_numeric_pairs(state, info)

        ordered_state: dict[str, float] = {}
        for key in priority_keys:
            if key in state:
                ordered_state[str(key)] = state[key]
        for key in sorted(state.keys()):
            if key not in ordered_state:
                ordered_state[key] = state[key]
        return ordered_state

    @staticmethod
    def _collect_numeric_pairs(target: dict[str, float], source: dict[str, Any]) -> None:
        """Appends scalar numeric values from a mapping into the target dict."""

        for key, value in source.items():
            numeric = _coerce_numeric(value)
            if numeric is None:
                continue
            target.setdefault(str(key), numeric)

    def _render_telemetry_block(
        self,
        telemetry: dict[str, float],
        *,
        fallback_lines: Sequence[str] = (),
    ) -> str:
        """Renders telemetry into a compact bullet list."""

        lines = [
            f"- {key}: {_format_number(value)}"
            for key, value in list(telemetry.items())[: self._telemetry_limit]
        ]
        for item in fallback_lines:
            text = str(item or "").strip()
            if text:
                lines.append(f"- {text}")
        return "\n".join(lines)

    @staticmethod
    def _select_telemetry_keys(
        telemetry: dict[str, float],
        *,
        selected_keys: Sequence[str],
    ) -> dict[str, float]:
        """Returns telemetry values whose keys are explicitly selected."""

        selected: dict[str, float] = {}
        for key in selected_keys:
            if key in telemetry:
                selected[str(key)] = telemetry[key]
        return selected

    @staticmethod
    def _coerce_positive_int(value: Any, *, default: int) -> int:
        """Coerces a value into a positive integer fallback."""

        try:
            return max(1, int(value))
        except (TypeError, ValueError):
            return int(default)

    def _resolve_scheme_id(self, scheme_id: Optional[str]) -> str:
        """Normalizes one configured scheme id and falls back when unsupported."""

        value = str(scheme_id or _DEFAULT_SCHEME_ID).strip()
        if value in _SUPPORTED_SCHEME_IDS:
            return value
        logger.warning(
            "ArenaPromptComposer received unsupported scheme_id {}; falling back to {}",
            value,
            _DEFAULT_SCHEME_ID,
        )
        return _DEFAULT_SCHEME_ID

    @staticmethod
    def _normalize_move(move: str) -> str:
        """Normalizes one action label token to lowercase key form."""

        return str(move).strip().lower()

    @staticmethod
    def _format_player_label(observation: ArenaObservation, player_id: str) -> str:
        names = observation.metadata.get("player_names")
        if isinstance(names, dict):
            display_name = names.get(player_id)
            if display_name and display_name != player_id:
                return f"{display_name} ({player_id})"
        return player_id


def _coerce_numeric(value: Any) -> Optional[float]:
    """Coerces one scalar value into float when possible."""

    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return None
        try:
            return float(text)
        except ValueError:
            return None
    return None


def _numeric_delta(
    previous: dict[str, float],
    current: dict[str, float],
    *,
    limit: int,
) -> dict[str, float]:
    """Computes one bounded numeric delta dictionary between snapshots."""

    keys = list(dict.fromkeys([*previous.keys(), *current.keys()]))
    output: dict[str, float] = {}
    for key in keys:
        before = _coerce_numeric(previous.get(key))
        after = _coerce_numeric(current.get(key))
        if before is None or after is None:
            continue
        diff = after - before
        if diff == 0.0:
            continue
        output[str(key)] = diff
        if len(output) >= max(1, int(limit)):
            break
    return output


def _format_number(value: Any) -> str:
    """Formats numeric values for prompt rendering."""

    numeric = _coerce_numeric(value)
    if numeric is None:
        return str(value)
    if abs(numeric - round(numeric)) < 1e-6:
        return str(int(round(numeric)))
    return f"{numeric:.3f}".rstrip("0").rstrip(".")
