from __future__ import annotations

import json
from queue import Queue
from pathlib import Path
from typing import Any
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import pytest

from gage_eval.role.arena.input_mapping import BrowserKeyEvent, GameInputMapper, HumanActionEvent
from gage_eval.tools.ws_rgb_server import DisplayRegistration, WsRgbHubServer, _is_client_disconnect_error


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


def test_ws_rgb_client_disconnect_error_helper() -> None:
    assert _is_client_disconnect_error(BrokenPipeError())
    assert _is_client_disconnect_error(ConnectionResetError())
    assert _is_client_disconnect_error(OSError(32, "broken pipe"))
    assert not _is_client_disconnect_error(ValueError("unexpected"))


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
        assert displays[0]["accepts_input"] is True
        frame_payload = hub.broadcast_frame("display-1")
        assert frame_payload["ok"] is True
        assert frame_payload["frame"]["frame_id"] == 1

        response = hub.handle_input(
            display_id="display-1",
            payload={"type": "keydown", "key": "j"},
        )
        assert response["ok"] is True
        assert response["queued"] == 1
        queued_payload = json.loads(queue.get_nowait())
        assert queued_payload["player_id"] == "player_0"
        assert queued_payload["move"] == "MOVE:j"
        assert queued_payload["raw"] == "MOVE:j"
    finally:
        hub.stop()


def test_ws_rgb_hub_routes_sample_id_into_queued_payload() -> None:
    queue: Queue[str] = Queue()
    hub = WsRgbHubServer(host="127.0.0.1", port=0)
    hub.start()
    try:
        hub.register_display(
            DisplayRegistration(
                display_id="display-sample",
                label="retro",
                human_player_id="player_0",
                frame_source=lambda: {"frame_id": 1},
                input_mapper=_Mapper(),
                action_queue=queue,
                default_context={"sample_id": "sample_ws"},
            )
        )

        response = hub.handle_input(
            display_id="display-sample",
            payload={"type": "keydown", "key": "j"},
        )

        assert response["ok"] is True
        queued_payload = json.loads(queue.get_nowait())
        assert queued_payload["sample_id"] == "sample_ws"
        assert queued_payload["player_id"] == "player_0"
        assert queued_payload["move"] == "MOVE:j"
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


def test_ws_rgb_hub_reports_missing_input_mapper() -> None:
    hub = WsRgbHubServer(port=0)
    hub.start()
    try:
        hub.register_display(
            DisplayRegistration(
                display_id="display-no-mapper",
                label="text-only",
                human_player_id="player_0",
                frame_source=lambda: {"frame_id": 1},
            )
        )
        displays = hub.list_displays()
        assert displays[0]["accepts_input"] is False
        result = hub.handle_input(
            display_id="display-no-mapper",
            payload={"type": "keydown", "key": "x"},
        )
        assert result["ok"] is False
        assert result["error"] == "input_mapper_missing"
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
        assert payload["displays"][0]["accepts_input"] is True

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
        queued_payload = json.loads(queue.get_nowait())
        assert queued_payload["player_id"] == "player_0"
        assert queued_payload["move"] == "MOVE:k"
        assert queued_payload["raw"] == "MOVE:k"
    finally:
        hub.stop()


def test_ws_rgb_hub_http_session_control_endpoints() -> None:
    session = {
        "session_controlled": True,
        "phase": "in_progress",
        "replay_allowed": False,
        "can_terminate_game": True,
        "can_terminate_process": False,
    }

    def _session_state() -> dict[str, Any]:
        return dict(session)

    def _terminate_game() -> dict[str, Any]:
        session.update(
            {
                "phase": "game_ended",
                "replay_allowed": True,
                "can_terminate_game": False,
                "can_terminate_process": True,
                "reason": "manual_stop",
            }
        )
        return {"ok": True, "session": dict(session)}

    def _terminate_process(*, confirm: bool) -> dict[str, Any]:
        if not confirm:
            return {"ok": False, "error": "process_end_confirmation_required", "session": dict(session)}
        session.update(
            {
                "phase": "process_ended",
                "replay_allowed": False,
                "can_terminate_process": False,
            }
        )
        return {"ok": True, "session": dict(session)}

    hub = WsRgbHubServer(host="127.0.0.1", port=0)
    hub.start()
    try:
        hub.register_display(
            DisplayRegistration(
                display_id="display-session",
                label="retro",
                human_player_id="player_0",
                frame_source=lambda: {"frame_id": 3},
                session_state_source=_session_state,
                terminate_game=_terminate_game,
                terminate_process=_terminate_process,
            )
        )

        with urlopen(f"{hub.base_url}/ws_rgb/displays") as response:  # noqa: S310 - local test endpoint
            payload = json.loads(response.read().decode("utf-8"))
        assert payload["ok"] is True
        assert payload["displays"][0]["session"]["phase"] == "in_progress"

        terminate_game_request = Request(
            f"{hub.base_url}/ws_rgb/session",
            data=json.dumps({"display_id": "display-session", "action": "terminate_game"}).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urlopen(terminate_game_request) as response:  # noqa: S310 - local test endpoint
            payload = json.loads(response.read().decode("utf-8"))
        assert payload["ok"] is True
        assert payload["session"]["phase"] == "game_ended"

        terminate_process_request = Request(
            f"{hub.base_url}/ws_rgb/session",
            data=json.dumps({"display_id": "display-session", "action": "terminate_process"}).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urlopen(terminate_process_request) as _:
            raise AssertionError("Expected HTTPError for missing process-end confirmation")
    except HTTPError as exc:
        assert exc.code == 409
        payload = json.loads(exc.read().decode("utf-8"))
        assert payload["error"] == "process_end_confirmation_required"
    finally:
        hub.stop()


def test_ws_rgb_hub_http_session_control_confirmed_process_end() -> None:
    session = {
        "session_controlled": True,
        "phase": "game_ended",
        "replay_allowed": True,
        "can_terminate_game": False,
        "can_terminate_process": True,
    }

    def _terminate_process(*, confirm: bool) -> dict[str, Any]:
        if not confirm:
            return {"ok": False, "error": "process_end_confirmation_required", "session": dict(session)}
        session.update(
            {
                "phase": "process_ended",
                "replay_allowed": False,
                "can_terminate_process": False,
            }
        )
        return {"ok": True, "session": dict(session)}

    hub = WsRgbHubServer(host="127.0.0.1", port=0)
    hub.start()
    try:
        hub.register_display(
            DisplayRegistration(
                display_id="display-process-end",
                label="retro",
                human_player_id="player_0",
                frame_source=lambda: {"frame_id": 4},
                session_state_source=lambda: dict(session),
                terminate_process=_terminate_process,
            )
        )

        request = Request(
            f"{hub.base_url}/ws_rgb/session",
            data=json.dumps(
                {
                    "display_id": "display-process-end",
                    "action": "terminate_process",
                    "confirm": True,
                }
            ).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urlopen(request) as response:  # noqa: S310 - local test endpoint
            payload = json.loads(response.read().decode("utf-8"))
        assert payload["ok"] is True
        assert payload["session"]["phase"] == "process_ended"
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
        assert "GAGE Game Play Viewer" in html
        assert "/ws_rgb/displays" in html
        assert "/ws_rgb/frame" in html
        assert "/ws_rgb/frame_image" in html
        assert "/ws_rgb/replay_buffer" in html
        assert "/ws_rgb/session" in html
        assert "Terminate Game" in html
        assert "End Process" in html
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


def test_ws_rgb_hub_http_replay_buffer_endpoint_and_indexed_frame(tmp_path: Path) -> None:
    pil_image = pytest.importorskip("PIL.Image")
    image_path_0 = tmp_path / "frame_0.jpg"
    image_path_1 = tmp_path / "frame_1.jpg"
    pil_image.new("RGB", (4, 4), color=(200, 10, 10)).save(image_path_0, format="JPEG")
    pil_image.new("RGB", (4, 4), color=(10, 200, 10)).save(image_path_1, format="JPEG")

    replay_frames = [
        {"board_text": "f0", "_image_path_abs": str(image_path_0)},
        {"board_text": "f1", "_image_path_abs": str(image_path_1)},
    ]

    hub = WsRgbHubServer(host="127.0.0.1", port=0)
    hub.start()
    try:
        hub.register_display(
            DisplayRegistration(
                display_id="display-replay-indexed",
                label="replay",
                human_player_id="player_0",
                frame_source=lambda: {"board_text": "live"},
                frame_at=lambda index: replay_frames[max(0, min(index, len(replay_frames) - 1))],
                frame_count=lambda: len(replay_frames),
            )
        )

        with urlopen(f"{hub.base_url}/ws_rgb/displays") as response:  # noqa: S310 - local test endpoint
            displays_payload = json.loads(response.read().decode("utf-8"))
        assert displays_payload["ok"] is True
        assert displays_payload["displays"][0]["replay_seekable"] is True
        assert displays_payload["displays"][0]["replay_total"] == 2

        with urlopen(
            f"{hub.base_url}/ws_rgb/replay_buffer?display_id=display-replay-indexed"
        ) as response:  # noqa: S310 - local test endpoint
            replay_payload = json.loads(response.read().decode("utf-8"))
        assert replay_payload["ok"] is True
        assert replay_payload["total"] == 2
        assert replay_payload["loaded"] == 2
        assert replay_payload["frames"][0]["board_text"] == "f0"
        assert "_image_path_abs" not in replay_payload["frames"][0]

        with urlopen(
            f"{hub.base_url}/ws_rgb/frame?display_id=display-replay-indexed&replay_index=1"
        ) as response:  # noqa: S310 - local test endpoint
            indexed_frame_payload = json.loads(response.read().decode("utf-8"))
        assert indexed_frame_payload["ok"] is True
        assert indexed_frame_payload["frame"]["board_text"] == "f1"

        with urlopen(
            f"{hub.base_url}/ws_rgb/frame_image?display_id=display-replay-indexed&replay_index=1"
        ) as response:  # noqa: S310 - local test endpoint
            indexed_image_bytes = response.read()
            indexed_content_type = response.headers.get("Content-Type")
        assert indexed_content_type == "image/jpeg"
        assert isinstance(indexed_image_bytes, bytes)
        assert len(indexed_image_bytes) > 0
    finally:
        hub.stop()
