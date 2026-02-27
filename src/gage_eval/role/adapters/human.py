"""Human input adapter for arena games."""

from __future__ import annotations

from queue import Empty, Queue
import time
from typing import Any, Dict, Optional, Sequence

from loguru import logger

from gage_eval.registry import registry
from gage_eval.role.adapters.base import RoleAdapter, RoleAdapterState


@registry.asset(
    "roles",
    "human",
    desc="Human input adapter for arena games",
    tags=("role", "human"),
    role_type="human",
)
class HumanAdapter(RoleAdapter):
    """Role adapter that collects human moves from CLI or queue."""

    def __init__(
        self,
        adapter_id: str,
        *,
        source: str = "auto",
        static_moves: Optional[Sequence[str]] = None,
        action_queue: Optional[Queue[str]] = None,
        key_map: Optional[Dict[str, str]] = None,
        window_title: Optional[str] = None,
        capabilities=(),
        role_type: str = "human",
        **_,
    ) -> None:
        resolved_caps = tuple(capabilities) if capabilities else ("text",)
        super().__init__(adapter_id=adapter_id, role_type=role_type, capabilities=resolved_caps)
        self._source = source
        self._static_moves = list(static_moves or [])
        self._queue = action_queue
        self._key_map = dict(key_map or {})
        self._window_title = str(window_title or "GAGE Human Input")
        self._pygame_ready = False
        self._pygame = None
        self._pygame_screen = None
        self._pressed_keys: set[str] = set()

    async def ainvoke(self, payload: Dict[str, Any], state: RoleAdapterState) -> Dict[str, Any]:
        prompt = payload.get("prompt") or "Your move: "
        queue = payload.get("action_queue") or self._queue
        if self._source == "static":
            move = self._static_moves.pop(0) if self._static_moves else ""
        elif self._source == "auto":
            if queue is not None:
                move = queue.get()
            else:
                move = input(str(prompt))
        elif self._source in {"queue", "visualizer"}:
            if queue is None:
                raise ValueError("HumanAdapter queue source requires action_queue")
            move = queue.get()
        elif self._source == "pygame":
            timeout_ms = payload.get("timeout_ms")
            default_action = payload.get("default_action", "")
            move = self._read_pygame_action(timeout_ms=timeout_ms, default_action=default_action)
        else:
            move = input(str(prompt))
        return {"answer": move}

    def ensure_input_ready(self) -> None:
        """Initialize pygame input if configured."""

        if self._source != "pygame":
            return
        self._ensure_pygame()

    def bind_action_queue(self, action_queue: Optional[Queue[str]]) -> None:
        """Bind runtime action queue for async polling paths."""

        if action_queue is None:
            return
        self._queue = action_queue

    def poll_action(
        self, *, timeout_ms: Optional[int] = None, default_action: Optional[str] = None
    ) -> Optional[str]:
        """Poll one action for async schedulers without blocking by default."""

        # STEP 1: Poll configured queue sources used by websocket/gradio human input.
        if self._source in {"queue", "visualizer"}:
            return self._poll_queue_action(timeout_ms=timeout_ms, default_action=default_action)
        if self._source == "auto" and self._queue is not None:
            return self._poll_queue_action(timeout_ms=timeout_ms, default_action=default_action)

        # STEP 2: Poll pygame sources for local window controls.
        if self._source == "pygame":
            return self._read_pygame_action(timeout_ms=timeout_ms, default_action=default_action)
        return None

    def _poll_queue_action(
        self, *, timeout_ms: Optional[int], default_action: Optional[str]
    ) -> Optional[str]:
        queue = self._queue
        if queue is None:
            if default_action is None:
                return None
            return str(default_action)
        try:
            if timeout_ms is None:
                value = queue.get_nowait()
            else:
                timeout_s = max(0.0, float(timeout_ms) / 1000.0)
                if timeout_s <= 0:
                    value = queue.get_nowait()
                else:
                    value = queue.get(timeout=timeout_s)
        except Empty:
            if default_action is None:
                return None
            return str(default_action)
        return str(value)

    def _read_pygame_action(
        self, *, timeout_ms: Optional[int], default_action: Optional[str]
    ) -> Optional[str]:
        self._ensure_pygame()
        pygame = self._pygame
        if pygame is None:
            raise ValueError("pygame is required for source=pygame")

        key_map = self._key_map or {
            "left": "2",
            "a": "2",
            "right": "3",
            "d": "3",
            "space": "1",
            "j": "1",
        }

        deadline = None
        if timeout_ms is not None:
            try:
                deadline = time.time() + max(0.0, float(timeout_ms)) / 1000.0
            except (TypeError, ValueError):
                deadline = None

        while True:
            pygame.event.pump()
            # Clear stuck keys if window is not focused.
            if hasattr(pygame, "key") and hasattr(pygame.key, "get_focused"):
                try:
                    if not pygame.key.get_focused():
                        self._pressed_keys.clear()
                except Exception:
                    pass
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    return ""
                if event.type == pygame.KEYDOWN:
                    key_name = pygame.key.name(event.key)
                    if key_name in self._pressed_keys:
                        continue
                    self._pressed_keys.add(key_name)
                    logger.info("HumanInput key pressed: {}", key_name)
                    mapped = key_map.get(key_name)
                    if mapped is not None:
                        return str(mapped)
                elif event.type == pygame.KEYUP:
                    key_name = pygame.key.name(event.key)
                    self._pressed_keys.discard(key_name)
            if deadline is not None and time.time() >= deadline:
                if default_action is None:
                    return None
                return str(default_action)
            time.sleep(0.01)

    def _ensure_pygame(self) -> None:
        if self._pygame_ready:
            return
        try:
            import pygame  # type: ignore
        except ImportError as exc:
            raise ValueError("pygame is required for source=pygame") from exc

        pygame.init()
        pygame.display.set_caption(self._window_title)
        self._pygame_screen = pygame.display.set_mode((320, 200))
        pygame.key.set_repeat(0)
        self._pygame = pygame
        self._pygame_ready = True
