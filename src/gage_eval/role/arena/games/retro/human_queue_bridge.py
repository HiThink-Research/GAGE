"""Retro-scoped compatibility bridge for queue-driven human input."""

from __future__ import annotations

from queue import Empty
from typing import Any, Optional


_RETRO_HUMAN_QUEUE_BRIDGE_APPLIED = False


def ensure_retro_human_queue_bridge() -> None:
    """Enable queue polling for retro human play under async schedulers.

    This bridge is intentionally scoped to retro runtime initialization and
    keeps compatibility with the current shared HumanAdapter/HumanPlayer API.
    """

    global _RETRO_HUMAN_QUEUE_BRIDGE_APPLIED
    if _RETRO_HUMAN_QUEUE_BRIDGE_APPLIED:
        return

    from gage_eval.role.adapters.human import HumanAdapter
    from gage_eval.role.arena.players.human_player import HumanPlayer

    # STEP 1: Ensure HumanAdapter can receive the runtime queue binding.
    if not hasattr(HumanAdapter, "bind_action_queue"):
        def _bind_action_queue(self: Any, action_queue: Optional[Any]) -> None:
            if action_queue is None:
                return
            self._queue = action_queue

        HumanAdapter.bind_action_queue = _bind_action_queue  # type: ignore[attr-defined]

    # STEP 2: Patch HumanAdapter.poll_action to support queue/visualizer sources.
    original_poll_action = HumanAdapter.poll_action

    def _poll_action(
        self: Any,
        *,
        timeout_ms: Optional[int] = None,
        default_action: Optional[str] = None,
    ) -> Optional[str]:
        source = str(getattr(self, "_source", "auto") or "auto")
        queue = getattr(self, "_queue", None)
        uses_queue = source in {"queue", "visualizer"} or (source == "auto" and queue is not None)
        if uses_queue:
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
            except (Empty, ValueError):
                if default_action is None:
                    return None
                return str(default_action)
            return str(value)
        return original_poll_action(self, timeout_ms=timeout_ms, default_action=default_action)

    HumanAdapter.poll_action = _poll_action  # type: ignore[assignment]

    # STEP 3: Rebind the runtime queue at HumanPlayer.start_thinking entry.
    original_start_thinking = HumanPlayer.start_thinking

    def _start_thinking(self: Any, observation: Any, *, deadline_ms: Optional[int] = None) -> bool:
        adapter = getattr(self._role_manager, "get_adapter", lambda _id: None)(self._adapter_id)
        bind_action_queue = getattr(adapter, "bind_action_queue", None)
        if callable(bind_action_queue):
            try:
                bind_action_queue(self._action_queue)
            except Exception:
                pass
        return original_start_thinking(self, observation, deadline_ms=deadline_ms)

    HumanPlayer.start_thinking = _start_thinking  # type: ignore[assignment]
    _RETRO_HUMAN_QUEUE_BRIDGE_APPLIED = True


__all__ = ["ensure_retro_human_queue_bridge"]
