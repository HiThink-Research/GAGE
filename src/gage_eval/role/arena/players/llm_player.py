"""LLM-backed arena player implementation."""

from __future__ import annotations

import hashlib
import time
from queue import Queue
from threading import Lock, Thread
from typing import Any, Dict, Optional, Sequence

from loguru import logger

from gage_eval.assets.prompts.renderers import PromptRenderer
from gage_eval.observability.trace import ObservabilityTrace
from gage_eval.role.arena.interfaces import MoveParser
from gage_eval.role.arena.prompt_composer import ArenaPromptComposer
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
        scheduler_mode: Optional[str] = None,
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
        self._scheduler_mode = str(scheduler_mode or "").strip() or None
        self._base_messages = list(sample.get("messages") or [])
        self._prompt_composer = ArenaPromptComposer(
            sample=self._sample,
            prompt_renderer=self._prompt_renderer,
            scheduler_mode=self._scheduler_mode,
            legal_moves_limit=self._legal_moves_limit,
        )
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
        logger.debug("LLMPlayer {} observation prompt seed (text-only):\n{}", self.name, prompt_text)
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
        logger.debug(
            "LLMPlayer {} outbound messages:\n{}",
            self.name,
            self._summarize_messages_for_log(messages),
        )
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
    def _summarize_messages_for_log(messages: Sequence[Dict[str, Any]]) -> str:
        lines: list[str] = []
        for index, message in enumerate(messages):
            if not isinstance(message, dict):
                lines.append(f"[{index}:unknown] {message}")
                continue
            role = str(message.get("role") or "unknown")
            content = LLMPlayer._format_message_content_for_log(message.get("content"))
            lines.append(f"[{index}:{role}] {content}")
        return "\n".join(lines)

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

    @staticmethod
    def _format_message_content_for_log(content: Any) -> str:
        if isinstance(content, list):
            parts: list[str] = []
            for fragment in content:
                if isinstance(fragment, dict):
                    fragment_type = str(fragment.get("type") or "")
                    if fragment_type == "text":
                        text = fragment.get("text")
                        if text is not None:
                            parts.append(str(text))
                        continue
                    if fragment_type in {"image", "image_url"}:
                        parts.append(LLMPlayer._format_image_reference_for_log(fragment))
                        continue
                if fragment is not None:
                    parts.append(str(fragment))
            return " ".join(part for part in parts if part).strip()
        return "" if content is None else str(content)

    @staticmethod
    def _format_image_reference_for_log(fragment: Dict[str, Any]) -> str:
        image_url = fragment.get("image_url")
        url: Optional[str] = None
        if isinstance(image_url, dict):
            raw_url = image_url.get("url")
            if raw_url is not None:
                url = str(raw_url)
        elif isinstance(image_url, str):
            url = image_url
        elif isinstance(fragment.get("url"), str):
            url = str(fragment["url"])
        elif isinstance(fragment.get("image"), str):
            url = str(fragment["image"])

        if not url:
            return "<image_ref:missing>"
        if url.startswith("data:"):
            header = url.split(",", 1)[0]
            digest = hashlib.sha1(url.encode("utf-8")).hexdigest()[:12]
            return f"<image_ref:{header};sha1={digest};chars={len(url)}>"
        if len(url) > 180:
            return f"<image_ref:{url[:180]}...>"
        return f"<image_ref:{url}>"

    def _format_observation(self, observation: ArenaObservation) -> str:
        return self._prompt_composer.format_observation(
            observation,
            prefer_renderer_instruction=self._prompt_renderer is not None,
        )

    def _format_grid_observation(self, observation: ArenaObservation) -> str:
        return self._prompt_composer.format_grid_observation(observation)

    def _format_card_observation(self, observation: ArenaObservation) -> str:
        return self._prompt_composer.format_card_observation(observation)

    def _should_use_card_prompt(self, observation: ArenaObservation) -> bool:
        return self._prompt_composer.should_use_card_prompt(observation)

    @staticmethod
    def _extract_prompt_spec(observation: ArenaObservation) -> Dict[str, Any]:
        return ArenaPromptComposer.extract_prompt_spec(observation)

    def _build_team_hint(self, observation: ArenaObservation) -> str:
        return self._prompt_composer.build_team_hint(observation)

    def _truncate_legal_moves(self, legal_moves: Sequence[str]) -> Sequence[str]:
        return self._prompt_composer.truncate_legal_moves(legal_moves)

    def _build_turn_messages(
        self,
        *,
        observation: ArenaObservation,
        prompt_text: str,
        image_fragment: Optional[Dict[str, Any]],
        retry_reason: Optional[str] = None,
        last_output: Optional[str] = None,
    ) -> list[Dict[str, Any]]:
        result = self._prompt_composer.build_turn_messages(
            player_id=self.name,
            adapter_id=self._adapter_id,
            base_messages=self._base_messages,
            observation=observation,
            prompt_text=prompt_text,
            image_fragment=image_fragment,
            retry_reason=retry_reason,
            last_output=last_output,
        )
        if result.fallback_reason == "render_error":
            logger.warning(
                "LLMPlayer {} prompt renderer failed, fallback to backward-compatible prompt: {}",
                self.name,
                result.render_error_message or "unknown error",
            )
        if result.fallback_reason:
            self._log_backward_prompt_usage(result.fallback_reason)
        return result.messages

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
        return self._prompt_composer.build_prompt_payload(
            player_id=self.name,
            base_messages=self._base_messages,
            observation=observation,
            prompt_text=prompt_text,
            retry_reason=retry_reason,
            last_output=last_output,
            legacy_messages=legacy_messages,
            has_image=has_image,
        )

    @staticmethod
    def _merge_payload_dicts(base: Dict[str, Any], overrides: Dict[str, Any]) -> Dict[str, Any]:
        return ArenaPromptComposer.merge_payload_dicts(base, overrides)

    def _resolve_mode(self, observation_context: Dict[str, Any]) -> str:
        return self._prompt_composer.resolve_mode(observation_context)

    def _resolve_game_type(self, observation_metadata: Dict[str, Any]) -> str:
        return self._prompt_composer.resolve_game_type(observation_metadata)

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
        return ArenaPromptComposer.append_image_fragment(
            messages,
            image_fragment=image_fragment,
            fallback_text=fallback_text,
        )

    @staticmethod
    def _has_user_message(messages: Sequence[Dict[str, Any]]) -> bool:
        return ArenaPromptComposer.has_user_message(messages)

    @staticmethod
    def _has_image_content(content: Sequence[Any]) -> bool:
        return ArenaPromptComposer.has_image_content(content)

    @staticmethod
    def _build_user_message(text: str, image_fragment: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        return ArenaPromptComposer.build_user_message(text, image_fragment)

    def _build_image_fragment(self, observation: ArenaObservation) -> Optional[Dict[str, Any]]:
        return self._prompt_composer.build_image_fragment(observation)

    def _resolve_image_data_url(self, image: Any) -> Optional[str]:
        return self._prompt_composer.resolve_image_data_url(image)

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
