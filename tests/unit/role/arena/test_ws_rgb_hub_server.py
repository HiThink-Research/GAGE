from __future__ import annotations

from queue import Queue

from gage_eval.role.arena.input_mapping import BrowserKeyEvent, GameInputMapper, HumanActionEvent
from gage_eval.tools.ws_rgb_server import DisplayRegistration, WsRgbHubServer


class _Mapper(GameInputMapper):
    def _map_event_to_actions(self, *, event: BrowserKeyEvent, context):
        if event.key is None or event.event_type not in {"keydown", "key_down"}:
            return []
        return [
            HumanActionEvent(
                player_id=str(context.get("human_player_id") or "player_0"),
                move=f"MOVE:{event.key}",
                raw=f"MOVE:{event.key}",
                metadata={},
            )
        ]


def test_ws_rgb_hub_register_list_broadcast_and_route_input() -> None:
    queue: Queue[str] = Queue()
    hub = WsRgbHubServer(host="127.0.0.1", port=5900)
    hub.start()
    hub.register_display(
        DisplayRegistration(
            display_id="display-1",
            label="retro",
            human_player_id="player_0",
            frame_source=lambda: {"frame_id": 1},
            input_mapper=_Mapper(),
            action_queue=queue,
        )
    )

    displays = hub.list_displays()
    assert len(displays) == 1
    assert displays[0]["display_id"] == "display-1"
    frame_payload = hub.broadcast_frame("display-1")
    assert frame_payload["ok"] is True
    assert frame_payload["frame"]["frame_id"] == 1

    response = hub.handle_input(
        display_id="display-1",
        payload={"type": "keydown", "key": "j"},
    )
    assert response["ok"] is True
    assert response["queued"] == 1
    assert queue.get_nowait() == "MOVE:j"


def test_ws_rgb_hub_reports_missing_display() -> None:
    hub = WsRgbHubServer()
    result = hub.handle_input(display_id="missing", payload={"type": "keydown", "key": "x"})
    assert result["ok"] is False
    assert result["error"] == "display_not_found"

