"""LLM-backed arena player implementation."""

from __future__ import annotations

import base64
import hashlib
import inspect
import os
from pathlib import Path
import time
from queue import Queue
from threading import Lock, Thread
from typing import Any, Dict, Optional, Sequence

from loguru import logger

from gage_eval.assets.prompts.renderers import PromptRenderer
from gage_eval.observability.trace import ObservabilityTrace
from gage_eval.role.arena.action_trace import attach_trace_action_applied
from gage_eval.role.arena.interfaces import MoveParser
from gage_eval.role.arena.prompt_composer import ArenaPromptComposer
from gage_eval.role.arena.types import ArenaAction, ArenaObservation
from gage_eval.utils.messages import stringify_message_content

_DEFAULT_DEBUG_IMAGE_DUMP_MAX = 0
_DEFAULT_DEBUG_IMAGE_DUMP_STRIDE = 1
_ENV_DEBUG_IMAGE_DUMP_DIR = "GAGE_ARENA_DEBUG_IMAGE_DUMP_DIR"
_ENV_DEBUG_IMAGE_DUMP_MAX = "GAGE_ARENA_DEBUG_IMAGE_DUMP_MAX"
_ENV_DEBUG_IMAGE_DUMP_STRIDE = "GAGE_ARENA_DEBUG_IMAGE_DUMP_STRIDE"
_MISSING = object()


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
        scheme_id: Optional[str] = None,
        scheme_params: Optional[Dict[str, Any]] = None,
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
            scheme_id=scheme_id,
            scheme_params=scheme_params,
        )
        self._scheme_id = self._prompt_composer.scheme_id
        self._scheme_explicitly_configured = self._prompt_composer.scheme_explicitly_configured
        self._action_history_len = self._prompt_composer.action_history_len
        self._delta_key_limit = self._prompt_composer.delta_key_limit
        self._telemetry_limit = self._prompt_composer.telemetry_limit
        self._emit_model_io_events = self._resolve_model_io_event_emission(role_manager)
        self._debug_image_dump_dir = self._resolve_debug_image_dump_dir(
            (scheme_params or {}).get("debug_image_dump_dir")
            if isinstance(scheme_params, dict)
            else None
        )
        self._debug_image_dump_max = self._coerce_non_negative_int(
            (scheme_params or {}).get("debug_image_dump_max", os.getenv(_ENV_DEBUG_IMAGE_DUMP_MAX))
            if isinstance(scheme_params, dict)
            else os.getenv(_ENV_DEBUG_IMAGE_DUMP_MAX),
            default=_DEFAULT_DEBUG_IMAGE_DUMP_MAX,
        )
        self._debug_image_dump_stride = self._coerce_positive_int(
            (scheme_params or {}).get(
                "debug_image_dump_stride",
                os.getenv(_ENV_DEBUG_IMAGE_DUMP_STRIDE),
            )
            if isinstance(scheme_params, dict)
            else os.getenv(_ENV_DEBUG_IMAGE_DUMP_STRIDE),
            default=_DEFAULT_DEBUG_IMAGE_DUMP_STRIDE,
        )
        self._debug_image_dump_count = 0
        self._backward_prompt_logged_reasons: set[str] = set()
        self._async_lock = Lock()
        self._async_queue: Queue[ArenaAction] = Queue()
        self._async_inflight = False
        self._async_thread: Optional[Thread] = None
        self._async_start_ts: Optional[float] = None
        self._async_timeout_ms: Optional[float] = None
        self._async_timeout_logged = False
        if self._debug_image_dump_dir and self._debug_image_dump_max > 0:
            logger.info(
                "LLMPlayer {} image dump enabled: dir={} max={} stride={}",
                self.name,
                self._debug_image_dump_dir,
                self._debug_image_dump_max,
                self._debug_image_dump_stride,
            )

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
        self._log_request_media_summary(
            observation=observation,
            messages=messages,
            phase="primary",
        )

        # STEP 1: Run the primary request.
        raw_text = self._invoke_model(
            messages,
            observation=observation,
            phase="primary",
        )
        parse_result = self._parser.parse(raw_text, legal_moves=observation.legal_actions_items)
        self._log_parse_result(parse_result, phase="primary")

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
            self._log_request_media_summary(
                observation=observation,
                messages=retry_messages,
                phase=f"retry_{retries + 1}",
            )
            raw_text = self._invoke_model(
                retry_messages,
                observation=observation,
                phase=f"retry_{retries + 1}",
            )
            parse_result = self._parser.parse(raw_text, legal_moves=observation.legal_actions_items)
            self._log_parse_result(parse_result, phase=f"retry_{retries + 1}")
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
                metadata = attach_trace_action_applied(
                    metadata,
                    observation=observation,
                    move=fallback_move,
                )
                metadata["error"] = parse_result.error
                metadata["fallback"] = self._fallback_policy
                action = ArenaAction(
                    player=self.name,
                    move=fallback_move,
                    raw=raw_text,
                    metadata=metadata,
                )
                self._remember_interaction(observation, action)
                return action

        metadata = self._build_action_metadata(parse_result, retry_count=retries)
        metadata = attach_trace_action_applied(
            metadata,
            observation=observation,
            move=parse_result.coord or "",
        )
        if parse_result.error:
            metadata["error"] = parse_result.error
        action = ArenaAction(
            player=self.name,
            move=parse_result.coord or "",
            raw=raw_text,
            metadata=metadata,
        )
        self._remember_interaction(observation, action)
        return action

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

    def _invoke_model(
        self,
        messages: Sequence[Dict[str, Any]],
        *,
        observation: ArenaObservation,
        phase: str,
    ) -> str:
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
            self._emit_model_request_event(
                messages=messages,
                observation=observation,
                phase=phase,
            )
        with self._role_manager.borrow_role(self._adapter_id) as role:
            output = role.invoke(payload, self._trace) if role else {}
        raw_text = _extract_text(output)
        self._emit_model_response_event(
            output=output,
            response_text=raw_text,
            observation=observation,
            phase=phase,
        )
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

    def _emit_model_request_event(
        self,
        *,
        messages: Sequence[Dict[str, Any]],
        observation: ArenaObservation,
        phase: str,
    ) -> None:
        """Emit one structured model-request event for arena inference."""

        if self._trace is None or not self._emit_model_io_events:
            return
        payload = {
            "player_id": self.name,
            "adapter_id": self._adapter_id,
            "scheme_id": self._scheme_id,
            "phase": str(phase),
            "step": self._resolve_observation_step(observation),
            "message_count": len(messages),
            "messages": self._serialize_messages_for_trace(messages),
            "message_summary": self._summarize_messages_for_log(messages),
            "legal_actions": self._normalize_legal_actions_for_trace(observation),
        }
        self._trace.emit("arena_model_request", payload)

    def _emit_model_response_event(
        self,
        *,
        output: Any,
        response_text: str,
        observation: ArenaObservation,
        phase: str,
    ) -> None:
        """Emit one structured model-response event for arena inference."""

        if self._trace is None or not self._emit_model_io_events:
            return
        payload = {
            "player_id": self.name,
            "adapter_id": self._adapter_id,
            "scheme_id": self._scheme_id,
            "phase": str(phase),
            "step": self._resolve_observation_step(observation),
            "response_text": str(response_text or ""),
        }
        if isinstance(output, dict):
            payload["output_keys"] = [str(key) for key in output.keys()]
            if output.get("latency_ms") is not None:
                payload["latency_ms"] = output.get("latency_ms")
            if output.get("usage") is not None:
                payload["usage"] = self._normalize_trace_value(output.get("usage"))
        self._trace.emit("arena_model_response", payload)

    def _resolve_model_io_event_emission(self, role_manager: Any) -> bool:
        """Return whether model I/O observability events should be emitted."""

        adapter_getter = getattr(role_manager, "get_adapter", None)
        if not callable(adapter_getter):
            return True
        adapter = adapter_getter(self._adapter_id)
        backend = self._unwrap_backend_proxy(
            self._get_static_attr(adapter, "backend", default=None)
        )
        if backend is None:
            return True
        backend_cls = backend.__class__
        backend_name = str(getattr(backend_cls, "__name__", ""))
        backend_module = str(getattr(backend_cls, "__module__", ""))
        if backend_name == "DummyBackend":
            return False
        if backend_module == "gage_eval.role.model.backends.dummy_backend":
            return False
        return True

    @staticmethod
    def _unwrap_backend_proxy(backend: Any) -> Any:
        """Unwrap backend proxy layers to inspect the concrete backend instance."""

        current = backend
        seen: set[int] = set()
        while current is not None:
            current_id = id(current)
            if current_id in seen:
                break
            seen.add(current_id)
            next_backend = LLMPlayer._get_static_attr(
                current,
                "_backend",
                default=_MISSING,
            )
            if next_backend is _MISSING or next_backend is current:
                break
            current = next_backend
        return current

    @staticmethod
    def _get_static_attr(target: Any, attr_name: str, *, default: Any) -> Any:
        """Read one attribute without triggering dynamic mock expansion."""

        try:
            marker = inspect.getattr_static(target, attr_name, _MISSING)
        except Exception:
            return default
        if marker is _MISSING:
            return default
        try:
            return getattr(target, attr_name)
        except Exception:
            return default

    @staticmethod
    def _serialize_messages_for_trace(messages: Sequence[Dict[str, Any]]) -> list[Dict[str, Any]]:
        """Serialize outbound messages while stripping large binary image payloads."""

        serialized: list[Dict[str, Any]] = []
        for message in messages:
            if not isinstance(message, dict):
                serialized.append({"role": "unknown", "content": str(message)})
                continue
            serialized.append(
                {
                    "role": str(message.get("role") or "unknown"),
                    "content": LLMPlayer._serialize_message_content_for_trace(message.get("content")),
                }
            )
        return serialized

    @staticmethod
    def _serialize_message_content_for_trace(content: Any) -> Any:
        """Serialize one message content block for structured observability events."""

        if isinstance(content, list):
            fragments: list[Dict[str, Any]] = []
            for fragment in content:
                if isinstance(fragment, dict):
                    fragment_type = str(fragment.get("type") or "unknown")
                    if fragment_type == "text":
                        fragments.append(
                            {
                                "type": "text",
                                "text": "" if fragment.get("text") is None else str(fragment.get("text")),
                            }
                        )
                        continue
                    if fragment_type in {"image", "image_url"}:
                        fragments.append(
                            {
                                "type": fragment_type,
                                "image_ref": LLMPlayer._format_image_reference_for_log(fragment),
                            }
                        )
                        continue
                    fragments.append(
                        {
                            "type": fragment_type,
                            "value": LLMPlayer._normalize_trace_value(fragment),
                        }
                    )
                    continue
                fragments.append({"type": "text", "text": "" if fragment is None else str(fragment)})
            return fragments
        return "" if content is None else str(content)

    @staticmethod
    def _normalize_trace_value(value: Any) -> Any:
        """Convert nested values into JSON-safe event payloads."""

        if value is None or isinstance(value, (bool, int, float, str)):
            return value
        shape = getattr(value, "shape", None)
        if shape is not None:
            return {"array_shape": LLMPlayer._normalize_trace_shape(shape)}
        if isinstance(value, dict):
            return {str(key): LLMPlayer._normalize_trace_value(item) for key, item in value.items()}
        if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
            return [LLMPlayer._normalize_trace_value(item) for item in value]
        return str(value)

    @staticmethod
    def _normalize_trace_shape(shape: Any) -> list[int]:
        """Convert array-like shapes into JSON-safe integer lists."""

        try:
            return [int(dim) for dim in shape]
        except Exception:
            return []

    @staticmethod
    def _normalize_legal_actions_for_trace(observation: ArenaObservation) -> Dict[str, Any]:
        """Serialize legal action payloads for model-request observability."""

        legal_actions = observation.legal_actions
        if isinstance(legal_actions, dict) and legal_actions:
            return LLMPlayer._normalize_trace_value(dict(legal_actions))
        return {"items": [str(item) for item in observation.legal_actions_items]}

    def _format_observation(self, observation: ArenaObservation) -> str:
        return self._prompt_composer.format_observation(
            observation,
            prefer_renderer_instruction=self._prompt_renderer is not None,
        )

    def _format_grid_observation(self, observation: ArenaObservation) -> str:
        return self._prompt_composer.format_grid_observation(observation)

    def _format_card_observation(self, observation: ArenaObservation) -> str:
        return self._prompt_composer.format_card_observation(observation)

    def _format_vizdoom_observation(self, observation: ArenaObservation) -> str:
        return self._prompt_composer.format_vizdoom_observation(observation)

    def _format_pettingzoo_observation(self, observation: ArenaObservation) -> str:
        return self._prompt_composer.format_pettingzoo_observation(observation)

    def _should_use_card_prompt(self, observation: ArenaObservation) -> bool:
        return self._prompt_composer.should_use_card_prompt(observation)

    def _should_use_pettingzoo_prompt(self, observation: ArenaObservation) -> bool:
        return self._prompt_composer.should_use_pettingzoo_prompt(observation)

    def _is_vizdoom_observation(self, observation: ArenaObservation) -> bool:
        return self._prompt_composer.is_vizdoom_observation(observation)

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
        image_fragment = self._prompt_composer.build_image_fragment(observation)
        if not image_fragment:
            return None
        image_url = image_fragment.get("image_url")
        data_url = image_url.get("url") if isinstance(image_url, dict) else None
        if isinstance(data_url, str):
            self._maybe_dump_request_image(observation=observation, data_url=data_url)
        return image_fragment

    def _resolve_image_data_url(self, image: Any) -> Optional[str]:
        return self._prompt_composer.resolve_image_data_url(image)

    def _build_action_metadata(self, parse_result, *, retry_count: int = 0) -> Dict[str, Any]:
        metadata = {"player_type": "backend", "scheme_id": self._scheme_id}
        decision_reason = getattr(parse_result, "reason", None)
        if decision_reason:
            metadata["decision_reason"] = str(decision_reason)
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

    def _remember_interaction(self, observation: ArenaObservation, action: ArenaAction) -> None:
        """Stores the latest observation/action pair for temporal prompts."""

        self._prompt_composer.remember_interaction(observation, action)

    def _resolve_observation_step(self, observation: ArenaObservation) -> Optional[int]:
        """Extracts one step index token from the observation."""

        return self._prompt_composer.resolve_observation_step(observation)

    def _log_parse_result(self, parse_result: Any, *, phase: str) -> None:
        """Logs parser output details for debugging prompt adherence."""

        logger.debug(
            "LLMPlayer {} parse_result: phase={} action={} error={} decision_reason={}",
            self.name,
            str(phase),
            getattr(parse_result, "coord", None),
            getattr(parse_result, "error", None),
            getattr(parse_result, "reason", None),
        )

    def _log_request_media_summary(
        self,
        *,
        observation: ArenaObservation,
        messages: Sequence[Dict[str, Any]],
        phase: str,
    ) -> None:
        """Log request image statistics for one model invocation."""

        fallback_shape = self._extract_observation_image_shape(observation)
        image_details = _summarize_message_images(messages, fallback_shape=fallback_shape)
        image_status = self._resolve_image_status(
            observation=observation,
            image_count=len(image_details),
        )
        logger.debug(
            "LLMPlayer {} request media: phase={} scheme_id={} step={} image_status={} "
            "image_count={} image_details={}",
            self.name,
            str(phase),
            self._scheme_id,
            self._resolve_observation_step(observation),
            image_status,
            len(image_details),
            image_details,
        )

    def _resolve_image_status(
        self,
        *,
        observation: ArenaObservation,
        image_count: int,
    ) -> str:
        """Classify whether images are expected or attached for this request."""

        if image_count > 0:
            return "attached"
        if not self._prompt_composer.scheme_supports_image():
            return "scheme_without_image"
        view = observation.view if isinstance(observation.view, dict) else {}
        image = view.get("image") if isinstance(view, dict) else None
        if image is None:
            return "observation_image_missing"
        return "image_fragment_unavailable"

    @staticmethod
    def _extract_observation_image_shape(
        observation: ArenaObservation,
    ) -> Optional[tuple[int, int]]:
        """Extract `(width, height)` from observation image metadata when available."""

        view = observation.view if isinstance(observation.view, dict) else {}
        image = view.get("image") if isinstance(view, dict) else None
        if isinstance(image, dict):
            shape = image.get("shape")
            if isinstance(shape, list) and len(shape) >= 2:
                try:
                    height = int(shape[0])
                    width = int(shape[1])
                    if width > 0 and height > 0:
                        return (width, height)
                except (TypeError, ValueError):
                    pass
            width = image.get("width")
            height = image.get("height")
            if width is not None and height is not None:
                try:
                    parsed_width = int(width)
                    parsed_height = int(height)
                    if parsed_width > 0 and parsed_height > 0:
                        return (parsed_width, parsed_height)
                except (TypeError, ValueError):
                    pass
        return None

    def _maybe_dump_request_image(self, *, observation: ArenaObservation, data_url: str) -> None:
        """Save a bounded number of request images for debugging."""

        if not self._debug_image_dump_dir:
            return
        if self._debug_image_dump_max <= 0:
            return
        if self._debug_image_dump_count >= self._debug_image_dump_max:
            return

        # STEP 1: Check step-based sampling before decoding payload.
        step = self._resolve_observation_step(observation)
        if (
            self._debug_image_dump_stride > 1
            and step is not None
            and int(step) % int(self._debug_image_dump_stride) != 0
        ):
            return

        # STEP 2: Decode and persist one snapshot image.
        image_bytes, extension, error = _decode_data_url_image(data_url)
        if image_bytes is None:
            logger.debug(
                "LLMPlayer {} skipped image dump due to decode error: {}",
                self.name,
                error or "unknown",
            )
            return
        try:
            dump_dir = Path(self._debug_image_dump_dir).expanduser()
            dump_dir.mkdir(parents=True, exist_ok=True)
            step_token = "na" if step is None else str(int(step))
            safe_player = str(self.name or "player").replace("/", "_")
            safe_scheme = str(self._scheme_id or "scheme").replace("/", "_")
            file_name = (
                f"{safe_player}_{safe_scheme}_step_{step_token}_"
                f"{self._debug_image_dump_count:03d}.{extension}"
            )
            path = dump_dir / file_name
            path.write_bytes(image_bytes)
            self._debug_image_dump_count += 1
            logger.info(
                "LLMPlayer {} saved prompt image snapshot {}/{} path={} step={}",
                self.name,
                self._debug_image_dump_count,
                self._debug_image_dump_max,
                str(path),
                step_token,
            )
        except Exception as exc:
            logger.warning(
                "LLMPlayer {} failed to dump prompt image: {}",
                self.name,
                str(exc),
            )

    @staticmethod
    def _coerce_positive_int(value: Any, *, default: int) -> int:
        """Coerces a value into a positive integer fallback."""

        try:
            return max(1, int(value))
        except (TypeError, ValueError):
            return int(default)

    @staticmethod
    def _coerce_non_negative_int(value: Any, *, default: int) -> int:
        """Coerces a value into a non-negative integer fallback."""

        try:
            return max(0, int(value))
        except (TypeError, ValueError):
            return int(default)

    @staticmethod
    def _resolve_debug_image_dump_dir(configured_dir: Any) -> Optional[str]:
        """Resolve one debug dump directory from scheme params or environment."""

        candidates = [configured_dir, os.getenv(_ENV_DEBUG_IMAGE_DUMP_DIR)]
        for candidate in candidates:
            text = str(candidate or "").strip()
            if text:
                return text
        return None


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


def _summarize_message_images(
    messages: Sequence[Dict[str, Any]],
    *,
    fallback_shape: Optional[tuple[int, int]] = None,
) -> list[dict[str, Any]]:
    """Collect image count and size metadata from chat messages."""

    details: list[dict[str, Any]] = []
    for message_index, message in enumerate(messages):
        if not isinstance(message, dict):
            continue
        content = message.get("content")
        if not isinstance(content, list):
            continue
        for content_index, item in enumerate(content):
            if not isinstance(item, dict):
                continue
            if str(item.get("type") or "").strip().lower() != "image_url":
                continue
            url = _resolve_image_url(item.get("image_url"))
            summary = _summarize_image_url(url)
            summary["index"] = len(details)
            summary["message_index"] = message_index
            summary["content_index"] = content_index
            if fallback_shape and "width" not in summary and "height" not in summary:
                summary["width"] = fallback_shape[0]
                summary["height"] = fallback_shape[1]
            details.append(summary)
    return details


def _resolve_image_url(value: Any) -> Optional[str]:
    """Resolve image URL payload from one message image fragment."""

    if isinstance(value, dict):
        url = value.get("url")
        return str(url) if isinstance(url, str) else None
    if isinstance(value, str):
        return value
    return None


def _summarize_image_url(url: Optional[str]) -> dict[str, Any]:
    """Build one compact summary for an image URL payload."""

    if not url:
        return {"source": "missing", "error": "url_missing"}
    if not isinstance(url, str):
        return {"source": "invalid", "error": "url_not_string"}
    if not url.startswith("data:"):
        return {"source": "remote_url", "url_length": len(url)}

    header, separator, payload = url.partition(",")
    if not separator:
        return {"source": "data_url", "error": "invalid_data_url_header"}

    mime_segment = header[len("data:") :]
    media_type = mime_segment.split(";", 1)[0] or "application/octet-stream"
    is_base64 = ";base64" in mime_segment.lower()
    summary: dict[str, Any] = {
        "source": "data_url",
        "media_type": media_type,
        "payload_chars": len(payload),
        "is_base64": is_base64,
    }
    if is_base64:
        summary["decoded_bytes"] = _estimate_base64_decoded_bytes(payload)
    return summary


def _estimate_base64_decoded_bytes(payload: str) -> int:
    """Estimate decoded bytes for one base64 payload without full decoding."""

    compact = "".join(str(payload).split())
    if not compact:
        return 0
    padding = len(compact) - len(compact.rstrip("="))
    estimated = (len(compact) * 3) // 4 - padding
    return max(0, int(estimated))


def _decode_data_url_image(data_url: str) -> tuple[Optional[bytes], str, Optional[str]]:
    """Decode one data URL image payload into raw file bytes."""

    if not isinstance(data_url, str):
        return None, "jpg", "data_url_not_string"
    if not data_url.startswith("data:"):
        return None, "jpg", "unsupported_image_url"
    header, separator, payload = data_url.partition(",")
    if not separator:
        return None, "jpg", "invalid_data_url"
    if ";base64" not in header.lower():
        return None, "jpg", "data_url_not_base64"

    media_type = header[len("data:") :].split(";", 1)[0].strip().lower()
    extension = {
        "image/jpeg": "jpg",
        "image/jpg": "jpg",
        "image/png": "png",
        "image/webp": "webp",
        "image/bmp": "bmp",
    }.get(media_type, "jpg")
    try:
        image_bytes = base64.b64decode(payload, validate=False)
    except Exception:
        return None, extension, "base64_decode_failed"
    if not image_bytes:
        return None, extension, "empty_image_payload"
    return image_bytes, extension, None
