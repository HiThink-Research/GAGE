"""In-process websocket RGB hub primitives for arena displays."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Mapping, Optional

from gage_eval.role.arena.input_mapping import GameInputMapper, HumanActionEvent


@dataclass
class DisplayRegistration:
    """Display registration metadata for WsRgbHubServer."""

    display_id: str
    label: str
    human_player_id: str
    frame_source: Callable[[], Any]
    input_mapper: Optional[GameInputMapper] = None
    legal_moves: Optional[list[str]] = None
    action_queue: Any = None
    default_context: dict[str, Any] = field(default_factory=dict)


class WsRgbHubServer:
    """Manage display registration, frame pull, and mapped input routing."""

    def __init__(self, *, host: str = "127.0.0.1", port: int = 5800) -> None:
        """Initialize hub server state."""

        self._host = str(host)
        self._port = int(port)
        self._running = False
        self._displays: dict[str, DisplayRegistration] = {}

    def start(self) -> None:
        """Mark the hub as running."""

        self._running = True

    def stop(self) -> None:
        """Stop the hub and clear runtime-only display bindings."""

        self._running = False
        self._displays.clear()

    def register_display(self, registration: DisplayRegistration) -> None:
        """Register or replace one display descriptor."""

        display_id = str(registration.display_id)
        if not display_id:
            raise ValueError("display_id is required")
        self._displays[display_id] = registration

    def unregister_display(self, display_id: str) -> None:
        """Unregister one display by id."""

        self._displays.pop(str(display_id), None)

    def list_displays(self) -> list[dict[str, Any]]:
        """List all display descriptors."""

        items: list[dict[str, Any]] = []
        for display_id in sorted(self._displays.keys()):
            reg = self._displays[display_id]
            items.append(
                {
                    "display_id": reg.display_id,
                    "label": reg.label,
                    "human_player_id": reg.human_player_id,
                    "legal_moves": list(reg.legal_moves or []),
                    "running": self._running,
                }
            )
        return items

    def broadcast_frame(self, display_id: str) -> dict[str, Any]:
        """Fetch one latest frame payload from display frame_source."""

        registration = self._displays.get(str(display_id))
        if registration is None:
            return {"ok": False, "error": "display_not_found", "display_id": str(display_id)}
        frame = registration.frame_source()
        return {"ok": True, "display_id": registration.display_id, "frame": frame}

    def handle_input(
        self,
        *,
        display_id: str,
        payload: Mapping[str, Any],
        context: Optional[Mapping[str, Any]] = None,
    ) -> dict[str, Any]:
        """Route browser input through mapper and enqueue mapped actions."""

        registration = self._displays.get(str(display_id))
        if registration is None:
            return {"ok": False, "error": "display_not_found", "display_id": str(display_id)}
        mapper = registration.input_mapper
        if mapper is None:
            return {"ok": False, "error": "input_mapper_missing", "display_id": registration.display_id}

        merged_context = dict(registration.default_context)
        merged_context.setdefault("display_id", registration.display_id)
        merged_context.setdefault("human_player_id", registration.human_player_id)
        if context:
            merged_context.update(dict(context))
        actions = mapper.handle_browser_event(payload, context=merged_context)
        queued = self._enqueue_actions(registration.action_queue, actions)
        return {
            "ok": True,
            "display_id": registration.display_id,
            "queued": queued,
            "actions": [item.to_dict() for item in actions],
        }

    @staticmethod
    def _enqueue_actions(action_queue: Any, actions: list[HumanActionEvent]) -> int:
        if action_queue is None:
            return 0
        queued = 0
        for action in actions:
            payload = action.to_queue_payload()
            if hasattr(action_queue, "put_nowait"):
                action_queue.put_nowait(payload)
            else:
                action_queue.put(payload)
            queued += 1
        return queued

