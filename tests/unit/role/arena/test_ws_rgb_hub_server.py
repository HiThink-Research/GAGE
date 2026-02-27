from __future__ import annotations

import json
from queue import Queue
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import pytest

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


def test_ws_rgb_hub_http_viewer_page_available() -> None:
    hub = WsRgbHubServer(host="127.0.0.1", port=0)
    hub.start()
    try:
        with urlopen(f"{hub.base_url}/ws_rgb/viewer") as response:  # noqa: S310 - local test endpoint
            html = response.read().decode("utf-8")
        assert "GAGE ws_rgb Viewer" in html
        assert "/ws_rgb/displays" in html
        assert "/ws_rgb/frame" in html
        assert "/ws_rgb/frame_image" in html
    finally:
        hub.stop()


def test_ws_rgb_hub_http_frame_image_endpoint_returns_jpeg() -> None:
    np = pytest.importorskip("numpy")
    frame = np.zeros((4, 6, 3), dtype=np.uint8)
    frame[:, :, 1] = 255

    hub = WsRgbHubServer(host="127.0.0.1", port=0)
    hub.start()
    try:
        hub.register_display(
            DisplayRegistration(
                display_id="display-image",
                label="pettingzoo",
                human_player_id="player_0",
                frame_source=lambda: {"frame_id": 11, "_rgb": frame},
            )
        )

        with urlopen(
            f"{hub.base_url}/ws_rgb/frame?display_id=display-image"
        ) as response:  # noqa: S310 - local test endpoint
            frame_payload = json.loads(response.read().decode("utf-8"))
        assert frame_payload["ok"] is True
        assert "_rgb" not in frame_payload["frame"]

        with urlopen(
            f"{hub.base_url}/ws_rgb/frame_image?display_id=display-image"
        ) as response:  # noqa: S310 - local test endpoint
            jpeg_bytes = response.read()
            content_type = response.headers.get("Content-Type")
        assert content_type == "image/jpeg"
        assert isinstance(jpeg_bytes, bytes)
        assert len(jpeg_bytes) > 0
    finally:
        hub.stop()


def test_ws_rgb_hub_http_frame_image_missing_returns_not_found() -> None:
    hub = WsRgbHubServer(host="127.0.0.1", port=0)
    hub.start()
    try:
        hub.register_display(
            DisplayRegistration(
                display_id="display-no-image",
                label="text-only",
                human_player_id="player_0",
                frame_source=lambda: {"frame_id": 12},
            )
        )
        with urlopen(f"{hub.base_url}/ws_rgb/frame_image?display_id=display-no-image") as _:
            raise AssertionError("Expected HTTPError for missing frame image")
    except HTTPError as exc:
        assert exc.code == 404
        payload = json.loads(exc.read().decode("utf-8"))
        assert payload["error"] == "frame_image_missing"
    finally:
        hub.stop()


def test_ws_rgb_hub_http_frame_image_supports_image_path_payload(tmp_path: Path) -> None:
    pil_image = pytest.importorskip("PIL.Image")

    image_path = tmp_path / "frame.jpg"
    pil_image.new("RGB", (4, 4), color=(20, 200, 20)).save(image_path, format="JPEG")

    hub = WsRgbHubServer(host="127.0.0.1", port=0)
    hub.start()
    try:
        hub.register_display(
            DisplayRegistration(
                display_id="display-path-image",
                label="replay",
                human_player_id="player_0",
                frame_source=lambda: {
                    "frame_id": 21,
                    "_image_path_abs": str(image_path),
                    "board_text": "from file",
                },
            )
        )

        with urlopen(
            f"{hub.base_url}/ws_rgb/frame?display_id=display-path-image"
        ) as response:  # noqa: S310 - local test endpoint
            payload = json.loads(response.read().decode("utf-8"))
        assert payload["ok"] is True
        assert "_image_path_abs" not in payload["frame"]

        with urlopen(
            f"{hub.base_url}/ws_rgb/frame_image?display_id=display-path-image"
        ) as response:  # noqa: S310 - local test endpoint
            image_bytes = response.read()
            content_type = response.headers.get("Content-Type")
        assert content_type == "image/jpeg"
        assert isinstance(image_bytes, bytes)
        assert len(image_bytes) > 0
    finally:
        hub.stop()
