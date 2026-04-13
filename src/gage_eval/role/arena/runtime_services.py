"""Thread-safe shared runtime services for arena adapters."""

from __future__ import annotations

import threading
from collections.abc import Callable, Mapping
from typing import Any, Optional

from loguru import logger

from gage_eval.role.arena.human_input_protocol import build_action_payload
from gage_eval.role.arena.visualization.contracts import (
    ActionIntentReceipt,
    ChatMessage,
    ControlCommand,
)


class _LazyService:
    """Guard lazy shared resource initialization with a dedicated lock."""

    def __init__(self, *, name: str, adapter_id: str) -> None:
        self._name = str(name)
        self._adapter_id = str(adapter_id)
        self._lock = threading.Lock()
        self._value: Any = None

    def get_or_create(self, factory: Callable[[], Any]) -> Any:
        value = self._value
        if value is not None:
            return value
        with self._lock:
            if self._value is None:
                self._value = factory()
                logger.info(
                    "ArenaRoleAdapter {} initialized shared {}",
                    self._adapter_id,
                    self._name,
                )
            return self._value

    def clear(self) -> Any:
        with self._lock:
            value = self._value
            self._value = None
            return value

    def peek(self) -> Any:
        return self._value


class ArenaRuntimeServiceHub:
    """Manage adapter-scoped shared services and sample route bindings."""

    def __init__(self, *, adapter_id: str) -> None:
        self._adapter_id = str(adapter_id)
        self._visualizer = _LazyService(name="visualizer", adapter_id=self._adapter_id)
        self._action_server = _LazyService(name="action_server", adapter_id=self._adapter_id)
        self._ws_rgb_hub = _LazyService(name="ws_rgb_hub", adapter_id=self._adapter_id)
        self._display_lock = threading.Lock()
        self._registered_displays: set[str] = set()
        self._intent_lock = threading.Lock()
        self._intent_counter = 0

    def ensure_visualizer(self, factory: Callable[[], Any]) -> Any:
        return self._visualizer.get_or_create(factory)

    def ensure_action_server(self, factory: Callable[[], Any]) -> Any:
        return self._action_server.get_or_create(factory)

    def ensure_ws_rgb_hub(self, factory: Callable[[], Any]) -> Any:
        return self._ws_rgb_hub.get_or_create(factory)

    def bind_sample_routes(
        self,
        *,
        sample_id: str,
        action_server: Any = None,
        action_router: Any = None,
        visualizer: Any = None,
    ) -> None:
        if action_server is not None and action_router is not None:
            action_server.register_action_queue(sample_id, action_router)
        if visualizer is not None and action_router is not None:
            visualizer.bind_action_queue(action_router, sample_id=sample_id)

    def clear_sample_routes(
        self,
        *,
        sample_id: str,
        action_server: Any = None,
        visualizer: Any = None,
    ) -> None:
        if action_server is not None:
            action_server.unregister_action_queue(sample_id)
        if visualizer is not None:
            visualizer.clear_action_queue(sample_id=sample_id)

    def register_display(self, *, display_id: str, hub: Any, registration: Any) -> None:
        hub.register_display(registration)
        with self._display_lock:
            self._registered_displays.add(str(display_id))

    def submit_action_intent(
        self,
        session_id: str,
        run_id: str | None,
        payload: Mapping[str, Any],
    ) -> ActionIntentReceipt:
        normalized = _normalize_action_intent(
            session_id=session_id,
            run_id=run_id,
            payload=payload,
        )
        intent_id = self._next_intent_id(session_id)
        action_server = self.peek_action_server()
        if action_server is None:
            return ActionIntentReceipt(
                intent_id=intent_id,
                state="rejected",
                reason="action_queue_not_available",
            )
        has_action_routes = getattr(action_server, "has_action_routes", None)
        if callable(has_action_routes) and not has_action_routes():
            return ActionIntentReceipt(
                intent_id=intent_id,
                state="rejected",
                reason="sample_route_not_found",
            )

        submit_action_payload = getattr(action_server, "submit_action_payload", None)
        if callable(submit_action_payload):
            error = submit_action_payload(normalized)
        else:
            queue, error = action_server.resolve_action_queue(normalized.get("sample_id"))
            if queue is not None:
                queue.put(normalized)
        if error is not None:
            return ActionIntentReceipt(
                intent_id=intent_id,
                state="rejected",
                reason=str(error),
            )
        return ActionIntentReceipt(
            intent_id=intent_id,
            state="accepted",
            reason="queued",
        )

    def submit_chat_message(
        self,
        session_id: str,
        run_id: str | None,
        payload: Mapping[str, Any],
    ) -> ActionIntentReceipt:
        del run_id
        normalized = _normalize_chat_message(payload)
        intent_id = self._next_intent_id(session_id)
        action_server = self.peek_action_server()
        if action_server is None:
            return ActionIntentReceipt(
                intent_id=intent_id,
                state="rejected",
                reason="action_queue_not_available",
            )

        route_error = _resolve_sample_route_error(action_server, session_id=session_id)
        if route_error is not None:
            return ActionIntentReceipt(
                intent_id=intent_id,
                state="rejected",
                reason=route_error,
            )

        chat_queue = getattr(action_server, "chat_queue", None)
        if chat_queue is None or not hasattr(chat_queue, "put"):
            return ActionIntentReceipt(
                intent_id=intent_id,
                state="rejected",
                reason="chat_queue_not_available",
            )

        chat_queue.put(normalized)
        return ActionIntentReceipt(
            intent_id=intent_id,
            state="accepted",
            reason="queued",
        )

    def submit_control_command(
        self,
        session_id: str,
        run_id: str | None,
        payload: Mapping[str, Any],
    ) -> ActionIntentReceipt:
        normalized = _normalize_control_command(payload)
        intent_id = self._next_intent_id(session_id)
        command = ControlCommand.from_dict(normalized)
        live_receipt = _submit_live_playback_control(
            visualizer=self.peek_visualizer(),
            session_id=session_id,
            run_id=run_id,
            command=command,
            intent_id=intent_id,
        )
        if live_receipt is not None:
            return live_receipt
        action_server = self.peek_action_server()
        if action_server is None:
            return ActionIntentReceipt(
                intent_id=intent_id,
                state="rejected",
                reason="action_queue_not_available",
            )

        route_error = _resolve_sample_route_error(action_server, session_id=session_id)
        if route_error is not None:
            return ActionIntentReceipt(
                intent_id=intent_id,
                state="rejected",
                reason=route_error,
            )

        submit_control_payload = getattr(action_server, "submit_control_payload", None)
        if callable(submit_control_payload):
            error = submit_control_payload(normalized)
            if error is not None:
                return ActionIntentReceipt(
                    intent_id=intent_id,
                    state="rejected",
                    reason=str(error),
                )
            return ActionIntentReceipt(
                intent_id=intent_id,
                state="accepted",
                reason="queued",
            )

        control_queue = getattr(action_server, "control_queue", None)
        if control_queue is None or not hasattr(control_queue, "put"):
            return ActionIntentReceipt(
                intent_id=intent_id,
                state="rejected",
                reason="control_queue_not_available",
            )
        control_queue.put(normalized)
        return ActionIntentReceipt(
            intent_id=intent_id,
            state="accepted",
            reason="queued",
        )

    def registered_displays(self) -> set[str]:
        with self._display_lock:
            return set(self._registered_displays)

    def shutdown(self) -> None:
        visualizer = self._visualizer.clear()
        if visualizer is not None:
            try:
                visualizer.stop()
            except Exception as exc:  # pragma: no cover - defensive
                logger.warning(
                    "ArenaRoleAdapter {} visualizer stop failed: {}",
                    self._adapter_id,
                    exc,
                )

        action_server = self._action_server.clear()
        if action_server is not None:
            try:
                action_server.stop()
            except Exception as exc:  # pragma: no cover - defensive
                logger.warning(
                    "ArenaRoleAdapter {} action server stop failed: {}",
                    self._adapter_id,
                    exc,
                )

        ws_rgb_hub = self._ws_rgb_hub.clear()
        if ws_rgb_hub is not None:
            try:
                for display_id in sorted(self.registered_displays()):
                    ws_rgb_hub.unregister_display(display_id)
                ws_rgb_hub.stop()
            except Exception as exc:  # pragma: no cover - defensive
                logger.warning(
                    "ArenaRoleAdapter {} ws hub stop failed: {}",
                    self._adapter_id,
                    exc,
                )
        with self._display_lock:
            self._registered_displays.clear()

    def peek_visualizer(self) -> Optional[Any]:
        return self._visualizer.peek()

    def peek_action_server(self) -> Optional[Any]:
        return self._action_server.peek()

    def peek_ws_rgb_hub(self) -> Optional[Any]:
        return self._ws_rgb_hub.peek()

    def _next_intent_id(self, session_id: str) -> str:
        with self._intent_lock:
            self._intent_counter += 1
            return f"{session_id}:intent-{self._intent_counter}"


def _normalize_action_intent(
    *,
    session_id: str,
    run_id: str | None,
    payload: Mapping[str, Any],
) -> dict[str, Any]:
    if not isinstance(payload, Mapping):
        raise ValueError("invalid_action_payload")
    player_id = _first_text(
        payload.get("playerId"),
        payload.get("player_id"),
        payload.get("player"),
    )
    if player_id is None:
        raise ValueError("missing_player_id")
    action_text, metadata = _normalize_action_value(payload.get("action", payload.get("move")))
    if action_text is None:
        raise ValueError("missing_action")

    top_level_metadata = payload.get("metadata")
    if isinstance(top_level_metadata, Mapping):
        metadata.update(dict(top_level_metadata))

    return build_action_payload(
        action=action_text,
        player_id=player_id,
        sample_id=str(session_id),
        source="arena_visual_gateway",
        run_id=run_id,
        metadata=metadata or None,
    )


def _normalize_chat_message(payload: Mapping[str, Any]) -> dict[str, str]:
    if not isinstance(payload, Mapping):
        raise ValueError("invalid_chat_payload")
    try:
        chat = ChatMessage.from_dict(payload)
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("invalid_chat_payload") from exc
    return {
        "player_id": chat.player_id,
        "text": chat.text,
        "channel": chat.channel,
    }


def _normalize_control_command(payload: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(payload, Mapping):
        raise ValueError("invalid_control_payload")
    try:
        command = ControlCommand.from_dict(payload)
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("invalid_control_payload") from exc
    return command.to_dict()


def _normalize_action_value(action_payload: Any) -> tuple[str | None, dict[str, Any]]:
    if isinstance(action_payload, Mapping):
        metadata = dict(action_payload.get("metadata") or {})
        for key, value in action_payload.items():
            if key in {"id", "move", "value", "text", "label", "metadata"}:
                continue
            metadata[key] = value
        action_text = _first_text(
            action_payload.get("id"),
            action_payload.get("move"),
            action_payload.get("value"),
            action_payload.get("text"),
        )
        return action_text, metadata
    return _first_text(action_payload), {}


def _resolve_sample_route_error(action_server: Any, *, session_id: str) -> str | None:
    has_action_routes = getattr(action_server, "has_action_routes", None)
    if callable(has_action_routes) and not has_action_routes():
        return "sample_route_not_found"

    resolve_action_queue = getattr(action_server, "resolve_action_queue", None)
    if callable(resolve_action_queue):
        _, error = resolve_action_queue(str(session_id))
        if error is not None:
            return str(error)
    return None


def _submit_live_playback_control(
    *,
    visualizer: Any,
    session_id: str,
    run_id: str | None,
    command: ControlCommand,
    intent_id: str,
) -> ActionIntentReceipt | None:
    if visualizer is None:
        return None
    resolve_live_session = getattr(visualizer, "resolve_live_session", None)
    if not callable(resolve_live_session):
        return None
    live_source = resolve_live_session(str(session_id), run_id=run_id)
    if live_source is None:
        return None
    apply_control_command = getattr(live_source, "apply_control_command", None)
    if not callable(apply_control_command):
        return None
    related_event_seq = apply_control_command(command)
    return ActionIntentReceipt(
        intent_id=intent_id,
        state="accepted",
        related_event_seq=related_event_seq,
        reason="playback_applied",
    )


def _first_text(*values: Any) -> str | None:
    for value in values:
        if value is None:
            continue
        text = str(value).strip()
        if text:
            return text
    return None
