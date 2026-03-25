"""Thread-safe shared runtime services for arena adapters."""

from __future__ import annotations

import threading
from collections.abc import Callable
from typing import Any, Optional

from loguru import logger


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

