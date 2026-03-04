"""LLM-backed arena player implementation."""

from __future__ import annotations

import copy
import base64
import io
import time
from queue import Queue
from threading import Lock, Thread
from typing import Any, Dict, Optional, Sequence

from loguru import logger

from gage_eval.assets.prompts.renderers import PromptContext, PromptRenderer
from gage_eval.observability.trace import ObservabilityTrace
from gage_eval.role.arena.interfaces import MoveParser
from gage_eval.role.arena.types import ArenaAction, ArenaObservation
from gage_eval.utils.messages import stringify_message_content


class LLMPlayer:
    """Arena player that delegates decisions to a DUT model adapter."""

    def __init__(
        self,
        *,
        name: str,
        adapter_id: str,
        role_manager,
        sample: Dict[str, Any],
        parser: MoveParser,
        trace: Optional[ObservabilityTrace] = None,
        max_retries: int = 0,
        legal_moves_limit: int = 40,
        sampling_params: Optional[Dict[str, Any]] = None,
        fallback_policy: str = "none",
        timeout_ms: Optional[int] = None,
        timeout_fallback_move: Optional[str] = None,
        prompt_renderer: Optional[PromptRenderer] = None,
    ) -> None:
        self.name = name
        self._adapter_id = adapter_id
        self._role_manager = role_manager
        self._sample = sample
        self._parser = parser
        self._trace = trace
        self._max_retries = max(0, int(max_retries))
        self._legal_moves_limit = max(0, int(legal_moves_limit))
        self._sampling_params = dict(sampling_params or {})
        self._fallback_policy = str(fallback_policy or "none").lower()
        self._timeout_ms = None if timeout_ms is None else max(1, int(timeout_ms))
        self._timeout_fallback_move = (
            None if timeout_fallback_move is None else str(timeout_fallback_move)
        )
        self._prompt_renderer = prompt_renderer
        self._base_messages = list(sample.get("messages") or [])
        self._backward_prompt_logged_reasons: set[str] = set()
        self._async_lock = Lock()
        self._async_queue: Queue[ArenaAction] = Queue()
        self._async_inflight = False
        self._async_thread: Optional[Thread] = None
        self._async_start_ts: Optional[float] = None
        self._async_timeout_ms: Optional[float] = None
        self._async_timeout_logged = False

    def think(self, observation: ArenaObservation) -> ArenaAction:
        """Produce an action using the LLM adapter."""

        prompt_text = self._format_observation(observation)
        image_fragment = self._build_image_fragment(observation)
        logger.debug("LLMPlayer {} prompt (text-only):\n{}", self.name, prompt_text)
        messages = self._build_turn_messages(
            observation=observation,
            prompt_text=prompt_text,
            image_fragment=image_fragment,
        )

        # STEP 1: Run the primary request.
        raw_text = self._invoke_model(messages)
        parse_result = self._parser.parse(raw_text, legal_moves=observation.legal_actions_items)

        # STEP 2: Retry with rethink prompts when parsing fails or is illegal.
        retries = 0
        while parse_result.error and retries < self._max_retries:
            rethink_prompt = self._parser.build_rethink_prompt(
                last_output=raw_text,
                reason=parse_result.error,
                legal_moves=self._truncate_legal_moves(observation.legal_actions_items),
            )
            retry_messages = self._build_turn_messages(
                observation=observation,
                prompt_text=rethink_prompt,
                image_fragment=image_fragment,
                retry_reason=parse_result.error,
                last_output=raw_text,
            )
            raw_text = self._invoke_model(retry_messages)
            parse_result = self._parser.parse(raw_text, legal_moves=observation.legal_actions_items)
            retries += 1

        if parse_result.error and observation.legal_actions_items:
            fallback_move = None
            if self._fallback_policy == "first_legal":
                fallback_move = observation.legal_actions_items[0]
            elif self._fallback_policy == "random":
                import random
                fallback_move = random.choice(observation.legal_actions_items)

            if fallback_move is not None:
                logger.warning(
                    "LLMPlayer {} fallback to legal move {} due to {}",
                    self.name,
                    fallback_move,
                    parse_result.error,
                )
                metadata = self._build_action_metadata(parse_result, retry_count=retries)
                metadata["error"] = parse_result.error
                metadata["fallback"] = self._fallback_policy
                return ArenaAction(
                    player=self.name,
                    move=fallback_move,
                    raw=raw_text,
                    metadata=metadata,
                )

        metadata = self._build_action_metadata(parse_result, retry_count=retries)
        if parse_result.error:
            metadata["error"] = parse_result.error
        return ArenaAction(
            player=self.name,
            move=parse_result.coord or "",
            raw=raw_text,
            metadata=metadata,
        )

    def start_thinking(self, observation: ArenaObservation, *, deadline_ms: Optional[int] = None) -> bool:
        """Start thinking asynchronously if no request is in-flight."""

        effective_timeout_ms = (
            self._timeout_ms if self._timeout_ms is not None else deadline_ms
        )
        with self._async_lock:
            if self._async_inflight or not self._async_queue.empty():
                return False
            self._async_inflight = True
            self._async_start_ts = time.monotonic()
            self._async_timeout_ms = (
                None if effective_timeout_ms is None else float(effective_timeout_ms)
            )
            self._async_timeout_logged = False

        def _run() -> None:
            try:
                action = self.think(observation)
            except Exception as exc:
                if self._is_shutdown_error(exc):
                    logger.debug("LLMPlayer {} async think canceled during shutdown: {}", self.name, exc)
                else:
                    logger.warning("LLMPlayer {} async think failed: {}", self.name, exc)
                action = None
            if action is not None:
                self._async_queue.put(action)
            with self._async_lock:
                self._async_inflight = False

        thread = Thread(target=_run, daemon=True)
        self._async_thread = thread
        thread.start()
        return True

    def has_action(self) -> bool:
        """Return True if an async action is ready."""

        if not self._async_queue.empty():
            return True
        with self._async_lock:
            if not self._async_inflight:
                return False
            started = self._async_start_ts
            timeout_ms = self._async_timeout_ms
            timeout_logged = self._async_timeout_logged
        if started is None or timeout_ms is None:
            return False
        elapsed_ms = (time.monotonic() - started) * 1000.0
        if elapsed_ms >= timeout_ms and not timeout_logged:
            logger.warning(
                "LLMPlayer {} async timeout after {}ms; awaiting late response",
                self.name,
                int(elapsed_ms),
            )
            with self._async_lock:
                self._async_timeout_logged = True
        return False

    def pop_action(self) -> ArenaAction:
        """Pop the next async action."""

        action = self._async_queue.get_nowait()
        with self._async_lock:
            self._async_inflight = False
            self._async_start_ts = None
            self._async_timeout_ms = None
            self._async_timeout_logged = False
        return action

    def wait_for_pending(self, timeout_s: float = 1.0) -> None:
        """Wait briefly for any in-flight async call to finish."""

        thread = self._async_thread
        if thread is None:
            return
        thread.join(timeout=max(0.0, float(timeout_s)))

    def _invoke_model(self, messages: Sequence[Dict[str, Any]]) -> str:
        payload = {
            "sample": self._sample,
            "messages": messages,
            "sampling_params": self._sampling_params,
            "usage": "arena_move",
        }
        if self._trace:
            payload["trace"] = self._trace
        with self._role_manager.borrow_role(self._adapter_id) as role:
            output = role.invoke(payload, self._trace) if role else {}
        raw_text = _extract_text(output)
        logger.debug("LLMPlayer {} output={}", self.name, raw_text)
        return raw_text

    @staticmethod
    def _is_shutdown_error(exc: Exception) -> bool:
        """Return True if the error indicates runtime shutdown in progress."""

        message = str(exc).strip().lower()
        if not message:
            return False
        return (
            "cannot schedule new futures after shutdown" in message
            or "rolepool" in message
            or "is shut down" in message
        )

    def _format_observation(self, observation: ArenaObservation) -> str:
        if self._is_vizdoom_observation(observation):
            return self._format_vizdoom_observation(observation)
        if self._should_use_pettingzoo_prompt(observation):
            return self._format_pettingzoo_observation(observation)
        if self._should_use_card_prompt(observation):
            return self._format_card_observation(observation)
        return self._format_grid_observation(observation)

    def _format_vizdoom_observation(self, observation: ArenaObservation) -> str:
        legal_moves = self._truncate_legal_moves(observation.legal_actions_items)
        legal_hint = ", ".join(legal_moves) if legal_moves else "none"
        metadata = observation.metadata if isinstance(observation.metadata, dict) else {}
        tick = metadata.get("t")
        reward = metadata.get("reward")
        status_lines = [
            f"- Legal actions: {legal_hint}",
        ]
        if tick is not None:
            status_lines.append(f"- Tick: {tick}")
        if reward is not None:
            status_lines.append(f"- Last reward: {reward}")
        lines = [
            "You are playing ViZDoom.",
            observation.view_text,
            "",
            "Status:",
            *status_lines,
            "",
            "Instructions:",
            "- Choose exactly one action id from the legal actions list.",
            "- Output ONLY the action id as an integer (no extra text).",
        ]
        return "\n".join(lines)

    def _format_pettingzoo_observation(self, observation: ArenaObservation) -> str:
        legal_moves = self._truncate_legal_moves(observation.legal_actions_items)
        legal_hint = ", ".join(legal_moves) if legal_moves else "none"
        active_player = _format_player_label(observation, observation.active_player)
        last_action = observation.last_action or "None"

        instructions = [
            "- Choose exactly one action from the legal moves list.",
            "- Output ONLY the action label or id on the last line.",
            "- Do not output explanations or extra text.",
        ]
        if any(self._normalize_move(move) == "fire" for move in legal_moves):
            instructions.append("- Prefer FIRE when available.")
        elif any(self._normalize_move(move) == "noop" for move in legal_moves):
            instructions.append("- Avoid NOOP unless it is the only legal move.")

        lines = [
            f"Active player: {active_player}",
            f"Opponent last move: {last_action}",
            "\nEnvironment:",
            observation.view_text,
            "\nLegal moves:",
            legal_hint,
            "\nInstructions:",
            *instructions,
        ]
        return "\n".join(lines)

    def _format_grid_observation(self, observation: ArenaObservation) -> str:
        legal_moves = self._truncate_legal_moves(observation.legal_actions_items)
        legal_hint = ", ".join(legal_moves) if legal_moves else "none"
        active_player = _format_player_label(observation, observation.active_player)

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

    def _format_card_observation(self, observation: ArenaObservation) -> str:
        legal_moves = self._truncate_legal_moves(observation.legal_actions_items)
        legal_hint = ", ".join(legal_moves) if legal_moves else "none"
        active_player = _format_player_label(observation, observation.active_player)
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
        team_hint = self._build_team_hint(observation)
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

    def _should_use_card_prompt(self, observation: ArenaObservation) -> bool:
        metadata = self._sample.get("metadata") if isinstance(self._sample, dict) else {}
        game_type = str(metadata.get("game_type", "")).lower()
        if game_type == "doudizhu":
            return True
        if game_type == "vizdoom":
            return False
        if isinstance(observation.metadata.get("public_state"), dict):
            return True
        return "Public State:" in observation.view_text

    def _is_vizdoom_observation(self, observation: ArenaObservation) -> bool:
        metadata = observation.metadata if isinstance(observation.metadata, dict) else {}
        obs_game_type = str(metadata.get("game_type", "")).lower()
        if obs_game_type == "vizdoom":
            return True
        sample_meta = self._sample.get("metadata") if isinstance(self._sample, dict) else {}
        sample_game_type = str(sample_meta.get("game_type", "")).lower()
        return sample_game_type == "vizdoom"

    def _should_use_pettingzoo_prompt(self, observation: ArenaObservation) -> bool:
        metadata = observation.metadata if isinstance(observation.metadata, dict) else {}
        env_id = str(metadata.get("env_id", "")).lower()
        return "pettingzoo" in env_id

    @staticmethod
    def _normalize_move(move: str) -> str:
        return str(move).strip().lower()

    def _build_team_hint(self, observation: ArenaObservation) -> str:
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

    def _truncate_legal_moves(self, legal_moves: Sequence[str]) -> Sequence[str]:
        if self._legal_moves_limit <= 0:
            return []
        if len(legal_moves) <= self._legal_moves_limit:
            return list(legal_moves)
        return list(legal_moves[: self._legal_moves_limit])

    def _build_turn_messages(
        self,
        *,
        observation: ArenaObservation,
        prompt_text: str,
        image_fragment: Optional[Dict[str, Any]],
        retry_reason: Optional[str] = None,
        last_output: Optional[str] = None,
    ) -> list[Dict[str, Any]]:
        legacy_messages = self._base_messages + [self._build_user_message(prompt_text, image_fragment)]
        if not self._prompt_renderer:
            self._log_backward_prompt_usage("missing_prompt_renderer")
            return legacy_messages

        prompt_payload = self._build_prompt_payload(
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
                "adapter_id": self._adapter_id,
                "role_type": "arena_player",
                "player_id": self.name,
            },
        )
        try:
            rendered = self._prompt_renderer.render(context)
        except Exception as exc:
            logger.warning(
                "LLMPlayer {} prompt renderer failed, fallback to backward-compatible prompt: {}",
                self.name,
                exc,
            )
            self._log_backward_prompt_usage("render_error")
            return legacy_messages

        if rendered.messages is not None:
            return self._append_image_fragment(
                rendered.messages,
                image_fragment=image_fragment,
                fallback_text=prompt_text,
            )
        if rendered.prompt:
            return self._base_messages + [self._build_user_message(rendered.prompt, image_fragment)]

        self._log_backward_prompt_usage("empty_render")
        return legacy_messages

    def _build_prompt_payload(
        self,
        *,
        observation: ArenaObservation,
        prompt_text: str,
        retry_reason: Optional[str],
        last_output: Optional[str],
        legacy_messages: Sequence[Dict[str, Any]],
        has_image: bool,
    ) -> Dict[str, Any]:
        metadata = dict(observation.metadata) if isinstance(observation.metadata, dict) else {}
        legal_moves = list(observation.legal_actions_items or [])
        return {
            "instruction": prompt_text,
            "prompt_text": prompt_text,
            "messages": list(legacy_messages),
            "player_id": self.name,
            "game_type": self._resolve_game_type(metadata),
            "retry_reason": str(retry_reason or ""),
            "last_model_output": str(last_output or ""),
            "has_image": bool(has_image),
            "active_player": observation.active_player,
            "last_action": observation.last_action,
            "legal_moves": legal_moves,
            "arena_observation": {
                "view_text": observation.view_text,
                "legal_moves": legal_moves,
                "active_player": observation.active_player,
                "last_action": observation.last_action,
                "metadata": metadata,
            },
        }

    def _resolve_game_type(self, observation_metadata: Dict[str, Any]) -> str:
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

    def _log_backward_prompt_usage(self, reason: str) -> None:
        if reason in self._backward_prompt_logged_reasons:
            return
        self._backward_prompt_logged_reasons.add(reason)
        if reason == "missing_prompt_renderer":
            logger.info(
                "LLMPlayer {} using backward-compatible prompt assembly (no prompt renderer configured)",
                self.name,
            )
            return
        logger.warning(
            "LLMPlayer {} using backward-compatible prompt assembly (reason={})",
            self.name,
            reason,
        )

    @staticmethod
    def _append_image_fragment(
        messages: Sequence[Dict[str, Any]],
        *,
        image_fragment: Optional[Dict[str, Any]],
        fallback_text: str,
    ) -> list[Dict[str, Any]]:
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
                content.append(fragment)
                return normalized
            if isinstance(content, str):
                message["content"] = [{"type": "text", "text": content}, fragment]
                return normalized

        normalized.append(LLMPlayer._build_user_message(fallback_text, fragment))
        return normalized

    @staticmethod
    def _build_user_message(text: str, image_fragment: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        content = [{"type": "text", "text": text}]
        if image_fragment:
            content.append(image_fragment)
        return {"role": "user", "content": content}

    def _build_image_fragment(self, observation: ArenaObservation) -> Optional[Dict[str, Any]]:
        view = observation.view or {}
        image = view.get("image")
        if image is None:
            return None
        data_url = self._resolve_image_data_url(image)
        if not data_url:
            return None
        return {"type": "image_url", "image_url": {"url": data_url}}

    def _resolve_image_data_url(self, image: Any) -> Optional[str]:
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
            logger.warning("Pillow not installed; skipping image conversion for arena prompt")
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

    def _build_action_metadata(self, parse_result, *, retry_count: int = 0) -> Dict[str, Any]:
        metadata = {"player_type": "backend"}
        chat_text = getattr(parse_result, "chat_text", None)
        if chat_text:
            metadata["chat"] = str(chat_text)
        hold_ticks = getattr(parse_result, "hold_ticks", None)
        if hold_ticks is not None:
            try:
                metadata["hold_ticks"] = max(1, int(hold_ticks))
            except (TypeError, ValueError):
                pass
        metadata["retry_count"] = max(0, int(retry_count))
        return metadata


def _format_player_label(observation: ArenaObservation, player_id: str) -> str:
    names = observation.metadata.get("player_names")
    if isinstance(names, dict):
        display_name = names.get(player_id)
        if display_name and display_name != player_id:
            return f"{display_name} ({player_id})"
    return player_id


def _extract_text(output: Any) -> str:
    if isinstance(output, dict):
        if isinstance(output.get("answer"), str):
            return output["answer"]
        if isinstance(output.get("text"), str):
            return output["text"]
        message = output.get("message")
        if isinstance(message, dict):
            return stringify_message_content(message.get("content"))
        messages = output.get("messages")
        if isinstance(messages, list) and messages:
            last = messages[-1]
            if isinstance(last, dict):
                return stringify_message_content(last.get("content"))
    return "" if output is None else str(output)
