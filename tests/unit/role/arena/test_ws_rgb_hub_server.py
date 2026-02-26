from __future__ import annotations

import json
from queue import Queue
from urllib.error import HTTPError
from urllib.request import Request, urlopen

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
    hub = WsRgbHubServer(host="127.0.0.1", port=0)
    hub.start()
    try:
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
    finally:
        hub.stop()


def test_ws_rgb_hub_reports_missing_display() -> None:
    hub = WsRgbHubServer(port=0)
    hub.start()
    try:
        result = hub.handle_input(display_id="missing", payload={"type": "keydown", "key": "x"})
        assert result["ok"] is False
        assert result["error"] == "display_not_found"
    finally:
        hub.stop()


def test_ws_rgb_hub_http_endpoints() -> None:
    queue: Queue[str] = Queue()
    hub = WsRgbHubServer(host="127.0.0.1", port=0)
    hub.start()
    try:
        hub.register_display(
            DisplayRegistration(
                display_id="display-http",
                label="retro",
                human_player_id="player_0",
                frame_source=lambda: {"frame_id": 9},
                input_mapper=_Mapper(),
                action_queue=queue,
            )
        )

        with urlopen(f"{hub.base_url}/ws_rgb/displays") as response:  # noqa: S310 - local test endpoint
            payload = json.loads(response.read().decode("utf-8"))
        assert payload["ok"] is True
        assert payload["displays"][0]["display_id"] == "display-http"

        with urlopen(
            f"{hub.base_url}/ws_rgb/frame?display_id=display-http"
        ) as response:  # noqa: S310 - local test endpoint
            payload = json.loads(response.read().decode("utf-8"))
        assert payload["ok"] is True
        assert payload["frame"]["frame_id"] == 9

        request_payload = {
            "display_id": "display-http",
            "payload": {"type": "keydown", "key": "k"},
            "context": {},
        }
        request = Request(
            f"{hub.base_url}/ws_rgb/input",
            data=json.dumps(request_payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urlopen(request) as response:  # noqa: S310 - local test endpoint
            payload = json.loads(response.read().decode("utf-8"))
        assert payload["ok"] is True
        assert payload["queued"] == 1
        assert queue.get_nowait() == "MOVE:k"
    finally:
        hub.stop()


def test_ws_rgb_hub_http_frame_missing_display_returns_not_found() -> None:
    hub = WsRgbHubServer(host="127.0.0.1", port=0)
    hub.start()
    try:
        with urlopen(f"{hub.base_url}/ws_rgb/frame?display_id=missing") as _:
            raise AssertionError("Expected HTTPError for missing display")
    except HTTPError as exc:
        assert exc.code == 404
        payload = json.loads(exc.read().decode("utf-8"))
        assert payload["error"] == "display_not_found"
    finally:
        hub.stop()
