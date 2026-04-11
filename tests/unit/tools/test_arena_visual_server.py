from __future__ import annotations

import base64
import os
import io
import json
import socket
from pathlib import Path
import time
from urllib.error import HTTPError
from urllib.parse import urlparse
from urllib.request import Request, urlopen

import pytest

from gage_eval.game_kits.board_game.gomoku.visualization import (
    VISUALIZATION_SPEC as GOMOKU_VISUALIZATION_SPEC,
)
from gage_eval.game_kits.board_game.tictactoe.visualization import (
    VISUALIZATION_SPEC as TICTACTOE_VISUALIZATION_SPEC,
)
from gage_eval.game_kits.phase_card_game.doudizhu.visualization import (
    VISUALIZATION_SPEC as DOUDIZHU_VISUALIZATION_SPEC,
)
from gage_eval.game_kits.phase_card_game.mahjong.visualization import (
    VISUALIZATION_SPEC as MAHJONG_VISUALIZATION_SPEC,
)
from gage_eval.game_kits.real_time_game.vizdoom.visualization import (
    VISUALIZATION_SPEC as VIZDOOM_VISUALIZATION_SPEC,
)
from gage_eval.role.arena.core.game_session import _visual_payload_snapshot
from gage_eval.game_kits.board_game.gomoku.environment import GomokuArenaEnvironment
from gage_eval.role.arena.types import GameResult
from gage_eval.role.arena.runtime_services import ArenaRuntimeServiceHub
from gage_eval.role.arena.visualization.contracts import ActionIntentReceipt
from gage_eval.role.arena.visualization.contracts import ObserverRef
from gage_eval.role.arena.visualization.contracts import PlaybackState
from gage_eval.role.arena.visualization.contracts import SchedulingState
from gage_eval.role.arena.visualization.contracts import VisualSession
from gage_eval.role.arena.visualization.gateway_service import TimelinePage
from gage_eval.role.arena.visualization import http_server as http_server_module
from gage_eval.role.arena.visualization.http_server import (
    ArenaVisualHTTPServer,
    ArenaVisualRequestHandler,
    _LOW_LATENCY_STREAM_POLL_INTERVAL_S,
)
from gage_eval.role.arena.visualization import live_session as live_session_module
from gage_eval.role.arena.visualization.live_session import RecorderLiveSessionSource
from gage_eval.role.arena.visualization.recorder import ArenaVisualSessionRecorder

_SMALL_PNG_BYTES = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAQAAAAECAIAAAAmkwkpAAAAGUlEQVR4nGNkaGBgYGBg+M8ABYwMjAyMDAwAAB0vAQx0J7s8AAAAAElFTkSuQmCC"
)


def _build_valid_png_bytes() -> bytes:
    pil_image = __import__("PIL.Image", fromlist=["Image"])
    buffer = io.BytesIO()
    pil_image.new("RGB", (6, 6), (22, 108, 74)).save(buffer, format="PNG")
    return buffer.getvalue()


def _get_json(url: str) -> dict[str, object]:
    with urlopen(url) as response:  # noqa: S310 - local test endpoint
        body = response.read().decode("utf-8")
    parsed = json.loads(body)
    assert isinstance(parsed, dict)
    return parsed


def _post_json(url: str, payload: dict[str, object]) -> dict[str, object]:
    request = Request(
        url,
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urlopen(request) as response:  # noqa: S310 - local test endpoint
        body = response.read().decode("utf-8")
    parsed = json.loads(body)
    assert isinstance(parsed, dict)
    return parsed


def _post_json_status(url: str, payload: dict[str, object]) -> tuple[int, bytes]:
    request = Request(
        url,
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urlopen(request) as response:  # noqa: S310 - local test endpoint
        return response.status, response.read()


def _get_bytes(url: str) -> bytes:
    with urlopen(url) as response:  # noqa: S310 - local test endpoint
        return response.read()


def _read_stream_prefix(url: str) -> tuple[str, bytes]:
    with urlopen(url, timeout=3) as response:  # noqa: S310 - local test endpoint
        content_type = response.headers.get("Content-Type", "")
        prefix = response.readline() + response.readline() + response.readline()
    return content_type, prefix


def _read_http_error(target: str | Request) -> tuple[int, dict[str, object]]:
    try:
        with urlopen(target) as _:
            raise AssertionError("Expected HTTPError")
    except HTTPError as exc:
        payload = json.loads(exc.read().decode("utf-8"))
        assert isinstance(payload, dict)
        return exc.code, payload


def _open_websocket(url: str) -> socket.socket:
    parsed = urlparse(url)
    host = parsed.hostname or "127.0.0.1"
    port = int(parsed.port or (443 if parsed.scheme == "wss" else 80))
    path = parsed.path or "/"
    if parsed.query:
        path = f"{path}?{parsed.query}"
    key = base64.b64encode(os.urandom(16)).decode("ascii")
    request = (
        f"GET {path} HTTP/1.1\r\n"
        f"Host: {host}:{port}\r\n"
        "Upgrade: websocket\r\n"
        "Connection: Upgrade\r\n"
        f"Sec-WebSocket-Key: {key}\r\n"
        "Sec-WebSocket-Version: 13\r\n"
        "\r\n"
    ).encode("utf-8")
    client = socket.create_connection((host, port), timeout=3)
    client.sendall(request)
    response = b""
    while b"\r\n\r\n" not in response:
        chunk = client.recv(4096)
        if not chunk:
            break
        response += chunk
    assert b"101 Switching Protocols" in response
    return client


def _send_websocket_text(client: socket.socket, payload: dict[str, object]) -> None:
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    mask = os.urandom(4)
    masked = bytes(
        value ^ mask[index % 4]
        for index, value in enumerate(body)
    )
    header = bytearray([0x81])
    body_length = len(masked)
    if body_length < 126:
        header.append(0x80 | body_length)
    elif body_length < (1 << 16):
        header.append(0x80 | 126)
        header.extend(body_length.to_bytes(2, "big"))
    else:
        header.append(0x80 | 127)
        header.extend(body_length.to_bytes(8, "big"))
    client.sendall(bytes(header) + mask + masked)


def _persist_visual_session(
    tmp_path: Path,
    *,
    run_id: str = "run-1",
    session_id: str = "sample-1",
) -> Path:
    replay_path = tmp_path / "runs" / run_id / "replays" / session_id / "replay.json"
    recorder = ArenaVisualSessionRecorder(
        plugin_id="arena.visualization.pettingzoo.frame_v1",
        game_id="pettingzoo",
        scheduling_family="turn",
        session_id=session_id,
    )

    recorder.record_decision_window_open(
        ts_ms=1001,
        step=1,
        tick=0,
        player_id="player_0",
        observation={"activePlayerId": "player_0", "frameRef": "frame-open"},
    )
    recorder.record_action_intent(
        ts_ms=1002,
        step=1,
        tick=0,
        player_id="player_0",
        action={"move": "fire"},
        observation={"frameRef": "frame-open"},
    )
    recorder.record_action_committed(
        ts_ms=1003,
        step=1,
        tick=0,
        player_id="player_0",
        action={"move": "fire"},
        result={"reward": 1},
    )
    recorder.record_decision_window_close(
        ts_ms=1004,
        step=1,
        tick=1,
        player_id="player_1",
        reason="committed",
    )
    recorder.record_snapshot(
        ts_ms=1005,
        step=2,
        tick=1,
        snapshot={
            "activePlayerId": "player_1",
            "legalActions": [
                {"id": "noop", "label": "No-op"},
                {"id": "jump", "label": "Jump"},
            ],
            "frameRef": "frame-5",
            "media": {
                "primary": {
                    "mediaId": "frame-5",
                    "transport": "artifact_ref",
                    "mimeType": "image/png",
                    "url": "frames/frame-5.png",
                    "previewRef": "thumb-5",
                },
                "auxiliary": [
                    {
                        "mediaId": "frame-5-mini",
                        "transport": "artifact_ref",
                        "mimeType": "image/png",
                        "url": "frames/frame-5-mini.png",
                    }
                ],
            },
            "board": {"state": "anchored"},
        },
        label="frame_snapshot",
        anchor=True,
    )
    recorder.record_result(
        ts_ms=1006,
        step=2,
        tick=1,
        result=GameResult(
            winner="player_0",
            result="win",
            reason="terminal",
            move_count=1,
            illegal_move_count=0,
            final_board="frame-state",
            move_log=[{"player": "player_0", "move": "fire"}],
            replay_path=str(replay_path),
        ),
    )

    manifest_path = recorder.persist(replay_path).manifest_path
    frames_dir = manifest_path.parent / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)
    (frames_dir / "frame-5.png").write_bytes(b"frame-5-bytes")
    (frames_dir / "frame-5-mini.png").write_bytes(b"frame-5-mini-bytes")
    return manifest_path


def _build_live_visual_recorder(
    *,
    session_id: str = "sample-live",
) -> ArenaVisualSessionRecorder:
    recorder = ArenaVisualSessionRecorder(
        plugin_id="arena.visualization.pettingzoo.frame_v1",
        game_id="pettingzoo",
        scheduling_family="real_time_tick",
        session_id=session_id,
    )
    recorder.record_decision_window_open(
        ts_ms=2001,
        step=1,
        tick=1,
        player_id="pilot_alpha",
        observation={
            "activePlayerId": "pilot_alpha",
            "view": {
                "text": "live frame text",
                "image": {
                    "data_url": "data:image/png;base64,Zm9v",
                },
            },
            "legalActions": [{"id": "noop", "label": "No-op"}],
            "metadata": {"stream_id": "main"},
        },
    )
    recorder.record_snapshot(
        ts_ms=2002,
        step=1,
        tick=1,
        snapshot={
            "step": 1,
            "tick": 1,
            "observation": {
                "activePlayerId": "pilot_alpha",
                "view": {
                    "text": "live frame text",
                    "image": {
                        "data_url": "data:image/png;base64,Zm9v",
                    },
                },
                "legalActions": [{"id": "noop", "label": "No-op"}],
                "metadata": {"stream_id": "main"},
            },
        },
        label="live_frame_snapshot",
        anchor=True,
    )
    return recorder


def test_low_latency_http_stream_poll_interval_stays_realtime_friendly() -> None:
    assert _LOW_LATENCY_STREAM_POLL_INTERVAL_S <= (1.0 / 144.0)


def test_arena_visual_http_server_send_json_normalizes_numpy_payloads() -> None:
    numpy = pytest.importorskip("numpy")

    class _DummyServer:
        allow_origin = "*"

    class _DummyHandler:
        def __init__(self) -> None:
            self.server = _DummyServer()
            self.status = None
            self.headers: list[tuple[str, str]] = []
            self.wfile = io.BytesIO()

        def send_response(self, status) -> None:  # noqa: ANN001
            self.status = status

        def send_header(self, key: str, value: str) -> None:
            self.headers.append((key, value))

        def end_headers(self) -> None:
            return None

        def _send_cors_headers(self) -> None:
            ArenaVisualRequestHandler._send_cors_headers(self)

    handler = _DummyHandler()
    payload = {
        "frameSeq": numpy.int32(7),
        "shape": numpy.array([84, 84, 3], dtype=numpy.int32),
    }

    ArenaVisualRequestHandler._send_json(handler, payload, status=200)

    body = json.loads(handler.wfile.getvalue().decode("utf-8"))
    assert handler.status == 200
    assert body == {
        "frameSeq": 7,
        "shape": [84, 84, 3],
    }


def _build_low_latency_frame_visual_recorder(
    *,
    session_id: str = "sample-low-latency-live",
) -> ArenaVisualSessionRecorder:
    recorder = ArenaVisualSessionRecorder(
        plugin_id=VIZDOOM_VISUALIZATION_SPEC.plugin_id,
        game_id="vizdoom",
        scheduling_family="real_time_tick",
        session_id=session_id,
    )
    recorder.record_decision_window_open(
        ts_ms=3001,
        step=1,
        tick=1,
        player_id="player_0",
        observation={
            "active_player": "player_0",
            "board_text": "Low latency frame runtime",
            "view": {
                "text": "Low latency frame runtime",
            },
            "legal_actions": {
                "items": [
                    {"id": "noop", "label": "No-op"},
                    {"id": "fire", "label": "Fire"},
                ]
            },
            "metadata": {
                "stream_id": "pov",
            },
        },
    )
    recorder.record_snapshot(
        ts_ms=3002,
        step=1,
        tick=1,
        snapshot={
            "step": 1,
            "tick": 1,
            "move_count": 1,
            "stream_id": "pov",
            "observation": {
                "active_player": "player_0",
                "board_text": "Low latency frame runtime",
                "view": {
                    "text": "Low latency frame runtime",
                },
                "legal_actions": {
                    "items": [
                        {"id": "noop", "label": "No-op"},
                        {"id": "fire", "label": "Fire"},
                    ]
                },
                "metadata": {
                    "stream_id": "pov",
                },
            },
            "media": {
                "primary": {
                    "mediaId": "frame-live",
                    "transport": "http_pull",
                    "mimeType": "image/png",
                    "url": "data:image/png;base64,Zm9v",
                }
            },
            "viewport": {
                "width": 1280,
                "height": 720,
            },
        },
        label="low_latency_live_snapshot",
        anchor=True,
    )
    return recorder


def _build_repeated_live_visual_recorder(
    *,
    session_id: str = "sample-live-repeat",
) -> ArenaVisualSessionRecorder:
    recorder = ArenaVisualSessionRecorder(
        plugin_id="arena.visualization.pettingzoo.frame_v1",
        game_id="pettingzoo",
        scheduling_family="real_time_tick",
        session_id=session_id,
    )
    for tick in (1, 2):
        recorder.record_snapshot(
            ts_ms=3000 + tick,
            step=tick,
            tick=tick,
            snapshot={
                "step": tick,
                "tick": tick,
                "observation": {
                    "activePlayerId": "pilot_alpha",
                    "view": {
                        "text": f"live frame {tick}",
                        "image": {
                            "data_url": "data:image/png;base64,Zm9v",
                        },
                    },
                    "legalActions": [{"id": "noop", "label": "No-op"}],
                },
            },
            label=f"repeated_frame_{tick}",
            anchor=True,
        )
    return recorder


def _encode_rgb_frame_data_url(frame) -> str:
    pil_image = __import__("PIL.Image", fromlist=["Image"])
    image = pil_image.fromarray(frame)
    if image.mode not in {"RGB", "RGBA", "L"}:
        image = image.convert("RGB")
    elif image.mode in {"RGBA", "L"}:
        image = image.convert("RGB")
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=85, optimize=True)
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/jpeg;base64,{encoded}"


def _persist_runtime_style_board_session(
    tmp_path: Path,
    *,
    session_id: str,
    game_id: str,
    plugin_id: str,
    snapshot_body: dict[str, object],
) -> Path:
    replay_path = tmp_path / "runs" / session_id / "replays" / session_id / "replay.json"
    replay_path.parent.mkdir(parents=True, exist_ok=True)
    replay_path.write_text(
        json.dumps({"schema": "gage_replay/v1", "artifacts": {}}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    recorder = ArenaVisualSessionRecorder(
        plugin_id=plugin_id,
        game_id=game_id,
        scheduling_family="turn",
        session_id=session_id,
    )
    recorder.record_snapshot(
        ts_ms=1005,
        step=int(snapshot_body.get("step", 1) or 1),
        tick=int(snapshot_body.get("tick", 1) or 1),
        snapshot=snapshot_body,
        anchor=True,
    )
    recorder.record_result(
        ts_ms=1006,
        step=int(snapshot_body.get("step", 1) or 1),
        tick=int(snapshot_body.get("tick", 1) or 1),
        result={"result": "completed", "reason": "terminal"},
    )
    return recorder.persist(replay_path).manifest_path


def test_arena_visual_http_server_serves_session_timeline_scene_media_and_markers(tmp_path: Path) -> None:
    _persist_visual_session(tmp_path)
    submitted_actions: list[tuple[str, str | None, dict[str, object]]] = []
    submitted_chat: list[tuple[str, str | None, dict[str, object]]] = []
    submitted_control: list[tuple[str, str | None, dict[str, object]]] = []

    def _action_submitter(session_id: str, run_id: str | None, payload: dict[str, object]) -> ActionIntentReceipt:
        submitted_actions.append((session_id, run_id, payload))
        return ActionIntentReceipt(
            intent_id="intent-1",
            state="accepted",
            related_event_seq=6,
            reason="queued",
        )

    def _chat_submitter(session_id: str, run_id: str | None, payload: dict[str, object]) -> ActionIntentReceipt:
        submitted_chat.append((session_id, run_id, payload))
        return ActionIntentReceipt(
            intent_id="chat-1",
            state="accepted",
            reason="queued",
        )

    def _control_submitter(session_id: str, run_id: str | None, payload: dict[str, object]) -> ActionIntentReceipt:
        submitted_control.append((session_id, run_id, payload))
        return ActionIntentReceipt(
            intent_id="control-1",
            state="accepted",
            reason="queued",
        )

    server = ArenaVisualHTTPServer(
        host="127.0.0.1",
        port=0,
        base_dir=tmp_path,
        action_submitter=_action_submitter,
        chat_submitter=_chat_submitter,
        control_submitter=_control_submitter,
    )
    server.start()
    try:
        host, port = server.server_address
        base_url = f"http://{host}:{port}/arena_visual/sessions/sample-1"

        session_payload = _get_json(base_url)
        timeline_payload = _get_json(f"{base_url}/timeline?after_seq=2&limit=2")
        markers_payload = _get_json(f"{base_url}/markers?marker=snapshot")
        scene_payload = _get_json(f"{base_url}/scene?seq=6")
        media_payload = _get_json(f"{base_url}/media/frame-5")
        media_content = _get_bytes(f"{base_url}/media/frame-5?content=1")
        receipt_payload = _post_json(
            f"{base_url}/actions",
            {
                "playerId": "player_0",
                "action": {"move": "fire"},
            },
        )
        chat_receipt_payload = _post_json(
            f"{base_url}/chat",
            {
                "playerId": "player_0",
                "text": "hello",
            },
        )
        control_receipt_payload = _post_json(
            f"{base_url}/control",
            {
                "commandType": "pause",
            },
        )

        assert session_payload["sessionId"] == "sample-1"
        assert session_payload["pluginId"] == "arena.visualization.pettingzoo.frame_v1"
        assert session_payload["lifecycle"] == "closed"

        assert timeline_payload == {
            "sessionId": "sample-1",
            "afterSeq": 2,
            "nextAfterSeq": 4,
            "limit": 2,
            "hasMore": True,
            "events": [
                {
                    "seq": 3,
                    "tsMs": 1003,
                    "type": "action_committed",
                    "label": "action_committed",
                    "payload": {
                        "step": 1,
                        "tick": 0,
                        "playerId": "player_0",
                        "action": {"move": "fire"},
                        "traceEntry": None,
                        "result": {"reward": 1},
                    },
                },
                {
                    "seq": 4,
                    "tsMs": 1004,
                    "type": "decision_window_close",
                    "label": "decision_window_close",
                    "payload": {
                        "step": 1,
                        "tick": 1,
                        "playerId": "player_1",
                        "windowId": None,
                        "reason": "committed",
                    },
                },
            ],
        }

        assert markers_payload == {
            "sessionId": "sample-1",
            "marker": "snapshot",
            "seqs": [5],
        }

        assert scene_payload["sceneId"] == "sample-1:seq:6"
        assert scene_payload["seq"] == 6
        assert scene_payload["kind"] == "frame"
        assert scene_payload["summary"]["snapshotSeq"] == 5
        assert scene_payload["media"]["primary"]["mediaId"] == "frame-5"
        assert [ref["mediaId"] for ref in scene_payload["media"]["auxiliary"]] == ["frame-5-mini"]

        assert media_payload == {
            "mediaId": "frame-5",
            "transport": "artifact_ref",
            "mimeType": "image/png",
            "url": "frames/frame-5.png",
            "previewRef": "thumb-5",
        }
        assert media_content == b"frame-5-bytes"

        assert receipt_payload == {
            "intentId": "intent-1",
            "state": "accepted",
            "relatedEventSeq": 6,
            "reason": "queued",
        }
        assert chat_receipt_payload == {
            "intentId": "chat-1",
            "state": "accepted",
            "reason": "queued",
        }
        assert control_receipt_payload == {
            "intentId": "control-1",
            "state": "accepted",
            "reason": "queued",
        }
        assert submitted_actions == [
            (
                "sample-1",
                None,
                {
                    "playerId": "player_0",
                    "action": {"move": "fire"},
                },
            )
        ]
        assert submitted_chat == [
            (
                "sample-1",
                None,
                {
                    "playerId": "player_0",
                    "text": "hello",
                },
            )
        ]
        assert submitted_control == [
            (
                "sample-1",
                None,
                {
                    "commandType": "pause",
                },
            )
        ]
    finally:
        server.stop()


def test_arena_visual_http_server_supports_fast_realtime_action_ack(tmp_path: Path) -> None:
    _persist_visual_session(tmp_path)
    submitted_actions: list[tuple[str, str | None, dict[str, object]]] = []

    def _action_submitter(session_id: str, run_id: str | None, payload: dict[str, object]) -> ActionIntentReceipt:
        submitted_actions.append((session_id, run_id, payload))
        return ActionIntentReceipt(
            intent_id="intent-fast",
            state="accepted",
            related_event_seq=6,
            reason="queued",
        )

    server = ArenaVisualHTTPServer(
        host="127.0.0.1",
        port=0,
        base_dir=tmp_path,
        action_submitter=_action_submitter,
    )
    server.start()
    try:
        host, port = server.server_address
        status, body = _post_json_status(
            f"http://{host}:{port}/arena_visual/sessions/sample-1/actions?fast=1",
            {
                "playerId": "player_0",
                "action": {"move": "fire"},
            },
        )
    finally:
        server.stop()

    assert status == 204
    assert body == b""
    assert submitted_actions == [
        (
            "sample-1",
            None,
            {
                "playerId": "player_0",
                "action": {"move": "fire"},
            },
        )
    ]


def test_arena_visual_http_server_accepts_realtime_input_websocket_messages(
    tmp_path: Path,
) -> None:
    recorder = _build_live_visual_recorder(session_id="sample-live-ws")
    recorder.extra_capabilities["supportsRealtimeInputWebSocket"] = True
    live_source = RecorderLiveSessionSource(
        recorder=recorder,
        run_id="run-live",
        live_scene_scheme="low_latency_channel",
    )
    submitted_actions: list[tuple[str, str | None, dict[str, object]]] = []

    def submit_action(
        session_id: str,
        run_id: str | None,
        payload: dict[str, object],
    ) -> ActionIntentReceipt:
        submitted_actions.append((session_id, run_id, dict(payload)))
        return ActionIntentReceipt(
            intent_id="intent-ws-1",
            state="accepted",
            related_event_seq=2,
            reason="queued",
        )

    server = ArenaVisualHTTPServer(
        host="127.0.0.1",
        port=0,
        base_dir=tmp_path,
        action_submitter=submit_action,
    )
    server.register_live_session(live_source)
    server.start()
    try:
        host, port = server.server_address
        ws_url = (
            f"ws://{host}:{port}/arena_visual/sessions/sample-live-ws/actions/ws?run_id=run-live"
        )
        client = _open_websocket(ws_url)
        try:
            _send_websocket_text(
                client,
                {
                    "playerId": "player_0",
                    "action": {
                        "move": "right",
                        "metadata": {"input_seq": 3},
                    },
                },
            )
            deadline = time.time() + 2.0
            while not submitted_actions and time.time() < deadline:
                time.sleep(0.01)
        finally:
            client.close()

        assert submitted_actions == [
            (
                "sample-live-ws",
                "run-live",
                {
                    "playerId": "player_0",
                    "action": {
                        "move": "right",
                        "metadata": {"input_seq": 3},
                    },
                },
            )
        ]
    finally:
        server.stop()


def test_arena_visual_http_server_serves_registered_live_session_without_manifest(tmp_path: Path) -> None:
    recorder = _build_live_visual_recorder()
    live_source = RecorderLiveSessionSource(
        recorder=recorder,
        run_id="run-live",
        live_scene_scheme="http_pull",
    )
    server = ArenaVisualHTTPServer(host="127.0.0.1", port=0, base_dir=tmp_path)
    server.register_live_session(live_source)
    server.start()
    try:
        host, port = server.server_address
        session_url = f"http://{host}:{port}/arena_visual/sessions/sample-live?run_id=run-live"
        timeline_url = f"http://{host}:{port}/arena_visual/sessions/sample-live/timeline?run_id=run-live"
        scene_url = f"http://{host}:{port}/arena_visual/sessions/sample-live/scene?seq=2&run_id=run-live"

        session_payload = _get_json(session_url)
        timeline_payload = _get_json(timeline_url)
        scene_payload = _get_json(scene_url)
        primary_media = scene_payload["media"]["primary"]
        assert isinstance(primary_media, dict)
        media_id = str(primary_media["mediaId"])
        media_payload = _get_json(
            f"http://{host}:{port}/arena_visual/sessions/sample-live/media/{media_id}?run_id=run-live"
        )
        media_content = _get_bytes(
            f"http://{host}:{port}/arena_visual/sessions/sample-live/media/{media_id}?run_id=run-live&content=1"
        )

        assert session_payload["sessionId"] == "sample-live"
        assert session_payload["lifecycle"] == "live_running"
        assert timeline_payload["events"][-1]["type"] == "snapshot"
        assert scene_payload["phase"] == "live"
        assert scene_payload["kind"] == "frame"
        assert primary_media["transport"] == "http_pull"
        assert str(primary_media["url"]).startswith("data:image/png;base64,")
        assert media_payload["transport"] == "http_pull"
        assert media_content == b"foo"
    finally:
        server.stop()


def test_live_session_refresh_payload_omits_snapshot_anchor_list_for_low_latency_sessions() -> None:
    recorder = _build_live_visual_recorder(session_id="sample-live-lightweight")
    for seq in range(3, 18):
        recorder.record_snapshot(
            ts_ms=2000 + seq,
            step=seq,
            tick=seq,
            snapshot={
                "step": seq,
                "tick": seq,
                "observation": {"metadata": {"stream_id": "main"}},
            },
            label="live_frame_snapshot",
            anchor=True,
        )
    live_source = RecorderLiveSessionSource(
        recorder=recorder,
        run_id="run-live",
        live_scene_scheme="low_latency_channel",
    )

    session = live_source.load_session()
    timeline_page = live_source.page_timeline(limit=3)

    assert session.timeline["eventCount"] == 17
    assert session.timeline["tailSeq"] == 17
    assert session.timeline["snapshotAnchors"] == []
    assert [event.seq for event in timeline_page.events] == [1, 2, 3]


def test_live_session_source_caches_live_scene_and_stream_frame(monkeypatch: pytest.MonkeyPatch) -> None:
    recorder = _build_live_visual_recorder(session_id="sample-live-cache")
    live_source = RecorderLiveSessionSource(
        recorder=recorder,
        run_id="run-live",
        live_scene_scheme="low_latency_channel",
        live_frame_supplier=lambda: {
            "stream_id": "main",
            "_rgb": object(),
        },
    )

    assemble_calls = 0
    original_assemble = live_session_module.assemble_visual_scene

    def _counted_assemble(*args, **kwargs):
        nonlocal assemble_calls
        assemble_calls += 1
        return original_assemble(*args, **kwargs)

    stream_payload_calls = 0

    def _counted_stream_payload(self, media_id: str):
        nonlocal stream_payload_calls
        stream_payload_calls += 1
        return (b"frame-bytes", "image/png")

    monkeypatch.setattr(live_session_module, "assemble_visual_scene", _counted_assemble)
    monkeypatch.setattr(
        RecorderLiveSessionSource,
        "_load_live_frame_stream_payload",
        _counted_stream_payload,
    )

    first_scene = live_source.load_scene(seq=2)
    second_scene = live_source.load_scene(seq=2)

    assert first_scene is not None
    assert second_scene is not None
    assert assemble_calls == 1
    assert first_scene.to_dict() == second_scene.to_dict()

    media_id = str(first_scene.media.primary.media_id)
    assert live_source.load_stream_frame(media_id) == (b"frame-bytes", "image/png")
    assert live_source.load_stream_frame(media_id) == (b"frame-bytes", "image/png")
    assert stream_payload_calls == 1


def test_arena_visual_http_server_streams_live_session_updates(tmp_path: Path) -> None:
    recorder = _build_live_visual_recorder(session_id="sample-live-events")
    live_source = RecorderLiveSessionSource(
        recorder=recorder,
        run_id="run-live",
        live_scene_scheme="low_latency_channel",
    )
    server = ArenaVisualHTTPServer(host="127.0.0.1", port=0, base_dir=tmp_path)
    server.register_live_session(live_source)
    server.start()
    try:
        host, port = server.server_address
        request = Request(
            f"http://{host}:{port}/arena_visual/sessions/sample-live-events/events?run_id=run-live&after_seq=0",
            headers={"Accept": "text/event-stream"},
            method="GET",
        )
        with urlopen(request, timeout=3) as response:  # noqa: S310 - local test endpoint
            content_type = response.headers.get("Content-Type", "")
            prefix = b"".join(response.readline() for _ in range(6)).decode("utf-8")

        assert content_type.startswith("text/event-stream")
        assert "event: delta" in prefix
        assert '"sessionId": "sample-live-events"' in prefix
        assert '"timeline"' in prefix
    finally:
        server.stop()


def test_arena_visual_http_server_skips_session_reload_when_live_revision_is_unchanged(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(http_server_module, "_LIVE_UPDATE_STREAM_WAIT_TIMEOUT_S", 0.01)

    class StableRevisionLiveSource:
        session_id = "sample-stable-revision"
        run_id = "run-live"

        def __init__(self) -> None:
            self.load_session_calls = 0
            self.timeline_calls = 0

        def load_session(self, *, observer: ObserverRef | None = None) -> VisualSession:
            self.load_session_calls += 1
            return VisualSession(
                session_id=self.session_id,
                game_id="pettingzoo",
                plugin_id="arena.visualization.pettingzoo.frame_v1",
                lifecycle="live_running",
                playback=PlaybackState(mode="live_tail", cursor_event_seq=0),
                observer=observer or ObserverRef(observer_kind="spectator"),
                scheduling=SchedulingState(family="real_time_tick", phase="recording"),
                capabilities={"supportsLiveUpdateStream": True},
                timeline={"eventCount": 0},
            )

        def load_live_header(self) -> dict[str, object]:
            return {"lifecycle": "live_running", "cursorEventSeq": 0, "tailSeq": 0}

        def current_live_revision(self) -> int:
            return 1

        def wait_for_live_revision(self, *, after_revision: int, timeout_s: float | None = None) -> int:
            return 1

        def page_timeline(self, *, after_seq: int | None = None, limit: int = 50) -> TimelinePage:
            self.timeline_calls += 1
            return TimelinePage(
                events=(),
                after_seq=after_seq,
                next_after_seq=after_seq,
                limit=limit,
                has_more=False,
            )

        def load_scene(self, *, seq: int, observer: ObserverRef | None = None):
            return None

        def lookup_marker(self, marker: str) -> tuple[int, ...]:
            return ()

        def lookup_media(self, media_id: str):
            return None

        def load_media_content(self, media_id: str):
            return None

        def load_stream_frame(self, media_id: str):
            return None

        def apply_control_command(self, command) -> int:
            return 0

    live_source = StableRevisionLiveSource()
    server = ArenaVisualHTTPServer(host="127.0.0.1", port=0, base_dir=tmp_path)
    server.register_live_session(live_source)
    server.start()
    try:
        host, port = server.server_address
        request = Request(
            f"http://{host}:{port}/arena_visual/sessions/sample-stable-revision/events?run_id=run-live",
            headers={"Accept": "text/event-stream"},
            method="GET",
        )
        with urlopen(request, timeout=3) as response:  # noqa: S310 - local test endpoint
            lines: list[str] = []
            deadline = time.monotonic() + 1.0
            while time.monotonic() < deadline:
                line = response.readline().decode("utf-8")
                if not line:
                    break
                lines.append(line)
                if line == ": keepalive\n":
                    break

        assert "event: delta\n" in lines
        assert ": keepalive\n" in lines
        assert live_source.load_session_calls == 1
    finally:
        server.stop()


def test_arena_visual_http_server_streams_session_delta_without_timeline_page(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(http_server_module, "_LIVE_UPDATE_STREAM_WAIT_TIMEOUT_S", 0.01)

    class SessionOnlyLiveSource:
        session_id = "sample-session-only"
        run_id = "run-live"

        def load_session(self, *, observer: ObserverRef | None = None) -> VisualSession:
            return VisualSession(
                session_id=self.session_id,
                game_id="pettingzoo",
                plugin_id="arena.visualization.pettingzoo.frame_v1",
                lifecycle="closed",
                playback=PlaybackState(mode="live_tail", cursor_event_seq=0),
                observer=observer or ObserverRef(observer_kind="spectator"),
                scheduling=SchedulingState(family="real_time_tick", phase="recording"),
                capabilities={"supportsLiveUpdateStream": True},
                timeline={"eventCount": 0},
            )

        def load_live_header(self) -> dict[str, object]:
            return {"lifecycle": "closed", "cursorEventSeq": 0, "tailSeq": 0}

        def current_live_revision(self) -> int:
            return 1

        def wait_for_live_revision(self, *, after_revision: int, timeout_s: float | None = None) -> int:
            return after_revision

        def page_timeline(self, *, after_seq: int | None = None, limit: int = 50) -> TimelinePage | None:
            return None

        def load_scene(self, *, seq: int, observer: ObserverRef | None = None):
            return None

        def lookup_marker(self, marker: str) -> tuple[int, ...]:
            return ()

        def lookup_media(self, media_id: str):
            return None

        def load_media_content(self, media_id: str):
            return None

        def load_stream_frame(self, media_id: str):
            return None

        def apply_control_command(self, command) -> int:
            return 0

    server = ArenaVisualHTTPServer(host="127.0.0.1", port=0, base_dir=tmp_path)
    server.register_live_session(SessionOnlyLiveSource())
    server.start()
    try:
        host, port = server.server_address
        request = Request(
            f"http://{host}:{port}/arena_visual/sessions/sample-session-only/events?run_id=run-live",
            headers={"Accept": "text/event-stream"},
            method="GET",
        )
        with urlopen(request, timeout=3) as response:  # noqa: S310 - local test endpoint
            payload = response.read().decode("utf-8")

        assert "event: delta\n" in payload
        assert '"timeline": null' in payload
    finally:
        server.stop()


def test_arena_visual_http_server_applies_replay_control_to_registered_live_session(
    tmp_path: Path,
) -> None:
    recorder = _build_live_visual_recorder(session_id="sample-live-control")
    live_source = RecorderLiveSessionSource(
        recorder=recorder,
        run_id="run-live",
        live_scene_scheme="http_pull",
    )
    service_hub = ArenaRuntimeServiceHub(adapter_id="arena")
    server = ArenaVisualHTTPServer(
        host="127.0.0.1",
        port=0,
        base_dir=tmp_path,
        control_submitter=service_hub.submit_control_command,
    )
    service_hub.ensure_visualizer(lambda: server)
    server.register_live_session(live_source)
    server.start()
    try:
        host, port = server.server_address
        session_url = (
            f"http://{host}:{port}/arena_visual/sessions/sample-live-control?run_id=run-live"
        )
        control_url = (
            f"http://{host}:{port}/arena_visual/sessions/sample-live-control/control?run_id=run-live"
        )

        before_payload = _get_json(session_url)
        control_receipt = _post_json(control_url, {"commandType": "replay"})
        after_payload = _get_json(session_url)

        assert before_payload["playback"]["mode"] == "live_tail"
        assert before_payload["playback"]["cursorEventSeq"] == 2
        assert control_receipt == {
            "intentId": "sample-live-control:intent-1",
            "state": "accepted",
            "relatedEventSeq": 1,
            "reason": "playback_applied",
        }
        assert after_payload["playback"]["mode"] == "replay_playing"
        assert after_payload["playback"]["cursorEventSeq"] == 1
    finally:
        server.stop()


def test_arena_visual_http_server_resolves_visualization_spec_per_recorded_session(tmp_path: Path) -> None:
    _persist_runtime_style_board_session(
        tmp_path,
        session_id="gomoku-runtime",
        game_id="gomoku",
        plugin_id=GOMOKU_VISUALIZATION_SPEC.plugin_id,
        snapshot_body={
            "step": 5,
            "tick": 5,
            "playerId": None,
            "observation": {
                "board_text": "   A B C\n 3 . . .\n 2 . W .\n 1 B . .",
                "legal_moves": ["B1", "C1", "A3"],
                "active_player": "White",
                "last_move": "B2",
                "metadata": {
                    "board_size": 3,
                    "move_count": 2,
                    "player_id": "Black",
                    "player_ids": ["Black", "White"],
                    "player_names": {"Black": "Black", "White": "White"},
                    "coord_scheme": "A1",
                    "winning_line": ["A1", "B2"],
                },
                "legal_actions": {"items": ["B1", "C1", "A3"]},
                "context": {"mode": "turn", "step": 5},
            },
            "arenaTrace": None,
            "result": None,
        },
    )
    _persist_runtime_style_board_session(
        tmp_path,
        session_id="tictactoe-runtime",
        game_id="tictactoe",
        plugin_id=TICTACTOE_VISUALIZATION_SPEC.plugin_id,
        snapshot_body={
            "step": 3,
            "tick": 3,
            "playerId": None,
            "observation": {
                "board_text": "   1 2 3\n 3 . . .\n 2 . O .\n 1 X . .",
                "legal_moves": ["1,2", "1,3", "3,1"],
                "active_player": "player_1",
                "last_move": "2,2",
                "metadata": {
                    "board_size": 3,
                    "move_count": 2,
                    "player_id": "player_0",
                    "player_ids": ["player_0", "player_1"],
                    "player_names": {"player_0": "Alpha", "player_1": "Beta"},
                    "coord_scheme": "ROW_COL",
                    "winning_line": ["1,1", "2,2", "3,3"],
                },
                "legal_actions": {"items": ["1,2", "1,3", "3,1"]},
                "context": {"mode": "turn", "step": 3},
            },
            "arenaTrace": None,
            "result": None,
        },
    )

    server = ArenaVisualHTTPServer(host="127.0.0.1", port=0, base_dir=tmp_path)
    server.start()
    try:
        host, port = server.server_address
        gomoku_scene = _get_json(f"http://{host}:{port}/arena_visual/sessions/gomoku-runtime/scene?seq=2")
        tictactoe_scene = _get_json(f"http://{host}:{port}/arena_visual/sessions/tictactoe-runtime/scene?seq=2")

        gomoku_cells = {cell["coord"]: cell for cell in gomoku_scene["body"]["board"]["cells"]}
        assert gomoku_cells["A1"]["occupant"] == "B"
        assert gomoku_cells["A1"]["playerId"] == "Black"
        assert gomoku_scene["body"]["players"] == [
            {"playerId": "Black", "playerName": "Black", "token": "B"},
            {"playerId": "White", "playerName": "White", "token": "W"},
        ]

        tictactoe_cells = {cell["coord"]: cell for cell in tictactoe_scene["body"]["board"]["cells"]}
        assert tictactoe_cells["1,1"]["occupant"] == "X"
        assert tictactoe_cells["1,1"]["playerId"] == "player_0"
        assert tictactoe_scene["body"]["players"] == [
            {"playerId": "player_0", "playerName": "Alpha", "token": "X"},
            {"playerId": "player_1", "playerName": "Beta", "token": "O"},
        ]
    finally:
        server.stop()


def test_arena_visual_http_server_serves_registered_binary_stream_live_session_without_manifest(
    tmp_path: Path,
) -> None:
    recorder = _build_live_visual_recorder()
    live_source = RecorderLiveSessionSource(
        recorder=recorder,
        run_id="run-live",
        live_scene_scheme="binary_stream",
    )
    server = ArenaVisualHTTPServer(host="127.0.0.1", port=0, base_dir=tmp_path)
    server.register_live_session(live_source)
    server.start()
    try:
        host, port = server.server_address
        scene_url = f"http://{host}:{port}/arena_visual/sessions/sample-live/scene?seq=2&run_id=run-live"

        scene_payload = _get_json(scene_url)
        primary_media = scene_payload["media"]["primary"]
        assert isinstance(primary_media, dict)
        media_id = str(primary_media["mediaId"])
        media_payload = _get_json(
            f"http://{host}:{port}/arena_visual/sessions/sample-live/media/{media_id}?run_id=run-live"
        )
        media_content = _get_bytes(
            f"http://{host}:{port}/arena_visual/sessions/sample-live/media/{media_id}?run_id=run-live&content=1"
        )

        assert primary_media["transport"] == "binary_stream"
        assert "url" not in primary_media
        assert media_payload["transport"] == "binary_stream"
        assert "url" not in media_payload
        assert media_content == b"foo"
    finally:
        server.stop()


def test_arena_visual_http_server_serves_registered_low_latency_live_stream_without_manifest(
    tmp_path: Path,
) -> None:
    recorder = _build_live_visual_recorder()
    live_source = RecorderLiveSessionSource(
        recorder=recorder,
        run_id="run-live",
        live_scene_scheme="low_latency_channel",
    )
    server = ArenaVisualHTTPServer(host="127.0.0.1", port=0, base_dir=tmp_path)
    server.register_live_session(live_source)
    server.start()
    try:
        host, port = server.server_address
        scene_url = f"http://{host}:{port}/arena_visual/sessions/sample-live/scene?seq=2&run_id=run-live"

        scene_payload = _get_json(scene_url)
        primary_media = scene_payload["media"]["primary"]
        assert isinstance(primary_media, dict)
        media_id = str(primary_media["mediaId"])
        media_payload = _get_json(
            f"http://{host}:{port}/arena_visual/sessions/sample-live/media/{media_id}?run_id=run-live"
        )
        content_request = Request(
            f"http://{host}:{port}/arena_visual/sessions/sample-live/media/{media_id}"
            "?run_id=run-live&content=1"
        )
        stream_url = str(primary_media["url"])
        if stream_url.startswith("/"):
            stream_url = f"http://{host}:{port}{stream_url}"
        content_type, prefix = _read_stream_prefix(stream_url)
        with urlopen(content_request) as response:  # noqa: S310 - local test endpoint
            single_frame_content_type = response.headers.get("Content-Type", "")
            single_frame_payload = response.read()

        assert primary_media["transport"] == "low_latency_channel"
        assert media_payload["transport"] == "low_latency_channel"
        assert str(primary_media["url"]).endswith(
            f"/arena_visual/sessions/sample-live/media/{media_id}/stream?run_id=run-live"
        )
        assert content_type.startswith("multipart/x-mixed-replace")
        assert b"--frame" in prefix
        assert b"Content-Type:" in prefix
        assert single_frame_content_type.startswith("image/png")
        assert single_frame_payload == b"foo"
    finally:
        server.stop()


def test_arena_visual_http_server_serves_registered_low_latency_frame_stream_without_manifest(
    tmp_path: Path,
) -> None:
    recorder = _build_low_latency_frame_visual_recorder()
    live_source = RecorderLiveSessionSource(
        recorder=recorder,
        run_id="run-low-latency-live",
        live_scene_scheme="low_latency_channel",
    )
    server = ArenaVisualHTTPServer(host="127.0.0.1", port=0, base_dir=tmp_path)
    server.register_live_session(live_source)
    server.start()
    try:
        host, port = server.server_address
        scene_url = (
            f"http://{host}:{port}/arena_visual/sessions/sample-low-latency-live/scene"
            f"?seq=2&run_id=run-low-latency-live"
        )

        scene_payload = _get_json(scene_url)
        assert scene_payload["kind"] == "frame"
        primary_media = scene_payload["media"]["primary"]
        assert isinstance(primary_media, dict)
        media_id = str(primary_media["mediaId"])
        media_payload = _get_json(
            "http://"
            f"{host}:{port}/arena_visual/sessions/sample-low-latency-live/media/{media_id}"
            "?run_id=run-low-latency-live"
        )
        stream_url = str(primary_media["url"])
        if stream_url.startswith("/"):
            stream_url = f"http://{host}:{port}{stream_url}"
        content_type, prefix = _read_stream_prefix(stream_url)

        assert primary_media["transport"] == "low_latency_channel"
        assert media_payload["transport"] == "low_latency_channel"
        assert str(primary_media["url"]).endswith(
            "/arena_visual/sessions/sample-low-latency-live/media/"
            f"{media_id}/stream?run_id=run-low-latency-live"
        )
        assert content_type.startswith("multipart/x-mixed-replace")
        assert b"--frame" in prefix
        assert b"Content-Type:" in prefix
    finally:
        server.stop()


def test_low_latency_live_source_prefers_latest_environment_frame_supplier() -> None:
    recorder = _build_low_latency_frame_visual_recorder()

    def load_latest_frame() -> dict[str, object]:
        return {
            "stream_id": "pov",
            "media": {
                "primary": {
                    "mediaId": "frame-live",
                    "transport": "http_pull",
                    "mimeType": "image/png",
                    "url": "data:image/png;base64,YmFy",
                }
            },
        }

    live_source = RecorderLiveSessionSource(
        recorder=recorder,
        run_id="run-low-latency-live",
        live_scene_scheme="low_latency_channel",
        live_frame_supplier=load_latest_frame,
    )

    scene = live_source.load_scene(seq=2)

    assert scene is not None
    assert scene.media is not None
    assert scene.media.primary is not None
    assert scene.media.primary.media_id == "live-channel-pov"
    assert live_source.load_stream_frame(scene.media.primary.media_id) == (b"bar", "image/png")


def test_http_pull_live_source_prefers_current_rgb_frame_over_snapshot_anchor() -> None:
    numpy = pytest.importorskip("numpy")
    recorder = ArenaVisualSessionRecorder(
        plugin_id="arena.visualization.pettingzoo.frame_v1",
        game_id="pettingzoo",
        scheduling_family="real_time_tick",
        session_id="sample-http-pull-live",
    )
    snapshot_url = f"data:image/png;base64,{base64.b64encode(_SMALL_PNG_BYTES).decode('ascii')}"
    recorder.record_snapshot(
        ts_ms=4101,
        step=1,
        tick=1,
        snapshot={
            "step": 1,
            "tick": 1,
            "observation": {
                "activePlayerId": "pilot_alpha",
                "board_text": "snapshot frame",
                "view": {
                    "text": "snapshot frame",
                    "image": {
                        "data_url": snapshot_url,
                    },
                },
                "metadata": {"stream_id": "main"},
            },
        },
        label="snapshot_frame",
        anchor=True,
    )

    live_rgb = numpy.zeros((4, 4, 3), dtype=numpy.uint8)
    live_rgb[:, :, 0] = 180
    live_rgb[:, :, 1] = 32
    expected_live_url = _encode_rgb_frame_data_url(live_rgb)
    live_frame_payload = {"_rgb": live_rgb, "_live_frame_version": 1}

    live_source = RecorderLiveSessionSource(
        recorder=recorder,
        run_id="run-http-pull-live",
        live_scene_scheme="http_pull",
        live_frame_supplier=lambda: live_frame_payload,
    )

    scene = live_source.load_scene(seq=1)

    assert scene is not None
    assert scene.media is not None
    assert scene.media.primary is not None
    assert scene.media.primary.transport == "http_pull"
    assert scene.media.primary.url == expected_live_url
    assert (
        scene.body["snapshot"]["observation"]["view"]["image"]["data_url"]
        == snapshot_url
    )


def test_live_table_scene_overlays_latest_tail_frame_chat_log() -> None:
    recorder = ArenaVisualSessionRecorder(
        plugin_id=DOUDIZHU_VISUALIZATION_SPEC.plugin_id,
        game_id="doudizhu",
        scheduling_family="turn",
        session_id="doudizhu-live-tail",
    )
    recorder.record_decision_window_open(
        ts_ms=2001,
        step=0,
        tick=0,
        player_id="player_0",
        observation={
            "board_text": "board",
            "public_state": {
                "landlord_id": "player_0",
                "num_cards_left": {"player_0": 20, "player_1": 17, "player_2": 17},
                "played_cards": [
                    {"player_id": "player_0", "cards": []},
                    {"player_id": "player_1", "cards": []},
                    {"player_id": "player_2", "cards": []},
                ],
                "seen_cards": ["3", "J", "A"],
                "trace": [],
            },
            "private_state": {"self_id": "player_0", "current_hand": ["3", "4"]},
            "ui_state": {
                "roles": {
                    "player_0": "landlord",
                    "player_1": "peasant",
                    "player_2": "peasant",
                },
                "seat_order": {
                    "bottom": "player_0",
                    "left": "player_1",
                    "right": "player_2",
                },
                "hands": [["C3", "D4"], [], []],
                "seen_cards": ["D3", "HJ", "SA"],
                "latest_actions": [[], [], []],
                "move_history": [],
                "chat_log": [],
            },
            "chat_log": [],
            "player_ids": ["player_0", "player_1", "player_2"],
            "player_names": {
                "player_0": "player_0",
                "player_1": "player_1",
                "player_2": "player_2",
            },
            "legal_actions": {"items": ["pass", "3"]},
        },
    )
    latest_frame_payload = {
        "_live_frame_version": 2,
        "active_player_id": "player_0",
        "player_ids": ["player_0", "player_1", "player_2"],
        "player_names": {
            "player_0": "player_0",
            "player_1": "player_1",
            "player_2": "player_2",
        },
        "public_state": {
            "landlord_id": "player_0",
            "num_cards_left": {"player_0": 20, "player_1": 17, "player_2": 17},
            "played_cards": [
                {"player_id": "player_0", "cards": []},
                {"player_id": "player_1", "cards": []},
                {"player_id": "player_2", "cards": []},
            ],
            "seen_cards": ["3", "J", "A"],
            "trace": [],
        },
        "private_state": {"self_id": "player_0", "current_hand": ["3", "4"]},
        "ui_state": {
            "roles": {
                "player_0": "landlord",
                "player_1": "peasant",
                "player_2": "peasant",
            },
            "seat_order": {
                "bottom": "player_0",
                "left": "player_1",
                "right": "player_2",
            },
            "hands": [["C3", "ST", "BJ"], [], []],
            "seen_cards": ["D3", "HJ", "SA"],
            "latest_actions": [[], [], []],
            "move_history": [],
            "chat_log": [{"player_id": "player_0", "text": "bubble now"}],
        },
        "chat_log": [{"player_id": "player_0", "text": "bubble now"}],
        "legal_moves": ["pass", "3"],
        "move_count": 0,
        "last_move": None,
    }
    live_source = RecorderLiveSessionSource(
        recorder=recorder,
        run_id="run-doudizhu-live",
        visualization_spec=DOUDIZHU_VISUALIZATION_SPEC,
        live_frame_supplier=lambda: latest_frame_payload,
    )

    scene = live_source.load_scene(
        seq=1,
        observer=ObserverRef(observer_id="player_0", observer_kind="player"),
    )

    assert scene is not None
    assert scene.kind == "table"
    assert scene.body["panels"]["chatLog"] == [{"playerId": "player_0", "text": "bubble now"}]
    bottom_seat = next(
        seat for seat in scene.body["table"]["seats"] if seat["playerId"] == "player_0"
    )
    assert bottom_seat["hand"]["cards"] == ["C3", "ST", "BJ"]


def test_live_table_scene_uses_observer_specific_mahjong_frame_for_player_view() -> None:
    recorder = ArenaVisualSessionRecorder(
        plugin_id=MAHJONG_VISUALIZATION_SPEC.plugin_id,
        game_id="mahjong",
        scheduling_family="turn",
        session_id="mahjong-live-tail",
    )
    recorder.record_snapshot(
        ts_ms=3001,
        step=0,
        tick=0,
        snapshot={"public_state": {"discards": []}},
        label="snapshot",
        anchor=True,
    )

    south_frame_payload = {
        "_live_frame_version": 4,
        "active_player_id": "south",
        "observer_player_id": "south",
        "player_ids": ["east", "south", "west", "north"],
        "player_names": {
            "east": "East",
            "south": "South",
            "west": "West",
            "north": "North",
        },
        "public_state": {
            "discards": ["B1"],
            "melds": {},
            "num_cards_left": {
                "east": 13,
                "south": 14,
                "west": 13,
                "north": 13,
            },
        },
        "private_state": {
            "self_id": "south",
            "hand": ["C1", "C2", "C3", "C4"],
            "draw_tile": "C4",
        },
        "legal_moves": ["C1"],
        "move_count": 1,
        "last_move": "B1",
    }
    east_frame_payload = {
        **south_frame_payload,
        "observer_player_id": "east",
        "private_state": {
            "self_id": "east",
            "hand": ["B2", "B3", "B4", "Red"],
            "draw_tile": None,
        },
        "legal_moves": ["Pass", "Pong"],
    }
    observer_calls: list[tuple[str, str]] = []

    live_source = RecorderLiveSessionSource(
        recorder=recorder,
        run_id="run-mahjong-live",
        visualization_spec=MAHJONG_VISUALIZATION_SPEC,
        live_frame_supplier=lambda: south_frame_payload,
        observer_live_frame_supplier=lambda observer: (
            observer_calls.append((observer.observer_kind, observer.observer_id)),
            east_frame_payload,
        )[1],
    )

    scene = live_source.load_scene(
        seq=1,
        observer=ObserverRef(observer_id="east", observer_kind="player"),
    )

    assert scene is not None
    assert observer_calls == [("player", "east")]
    assert scene.body["status"]["activePlayerId"] == "south"
    assert scene.body["status"]["observerPlayerId"] == "east"
    assert scene.body["status"]["privateViewPlayerId"] == "east"
    east_seat = next(seat for seat in scene.body["table"]["seats"] if seat["playerId"] == "east")
    south_seat = next(seat for seat in scene.body["table"]["seats"] if seat["playerId"] == "south")
    assert east_seat["isObserver"] is True
    assert east_seat["hand"]["isVisible"] is True
    assert east_seat["hand"]["cards"] == ["B2", "B3", "B4", "Red"]
    assert south_seat["hand"]["isVisible"] is False


def test_live_source_wait_for_revision_detects_latest_frame_version_updates() -> None:
    recorder = _build_low_latency_frame_visual_recorder()
    payload = {"_live_frame_version": 1, "stream_id": "pov"}
    live_source = RecorderLiveSessionSource(
        recorder=recorder,
        run_id="run-live-frame-revision",
        live_scene_scheme="low_latency_channel",
        live_frame_supplier=lambda: payload,
    )

    first_revision = live_source.current_live_revision()
    payload["_live_frame_version"] = 2

    assert live_source.wait_for_live_revision(after_revision=first_revision, timeout_s=0.2) > first_revision


def test_live_source_wait_for_revision_uses_recorder_condition_without_sleep(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    recorder = _build_live_visual_recorder(session_id="sample-wait-condition")
    live_source = RecorderLiveSessionSource(
        recorder=recorder,
        run_id="run-live-wait-condition",
    )
    revision = live_source.current_live_revision()

    def fail_sleep(_: float) -> None:
        raise AssertionError("wait_for_live_revision should use the recorder condition")

    monkeypatch.setattr(live_session_module.time, "sleep", fail_sleep)

    assert live_source.wait_for_live_revision(after_revision=revision, timeout_s=0.001) == revision


def test_low_latency_live_source_synthesizes_stream_media_without_inline_snapshot_image() -> None:
    recorder = ArenaVisualSessionRecorder(
        plugin_id="arena.visualization.retro.frame_v1",
        game_id="retro_mario",
        scheduling_family="real_time_tick",
        session_id="sample-synthetic-live",
    )
    recorder.record_snapshot(
        ts_ms=4101,
        step=1,
        tick=1,
        snapshot={
            "step": 1,
            "tick": 1,
            "observation": {
                "board_text": "live frame text",
                "view": {"text": "live frame text"},
                "metadata": {"stream_id": "main"},
            },
        },
        label="live_frame_snapshot",
        anchor=True,
    )

    def load_latest_frame() -> dict[str, object]:
        return {
            "stream_id": "main",
            "media": {
                "primary": {
                    "mediaId": "retro-frame-live",
                    "transport": "http_pull",
                    "mimeType": "image/png",
                    "url": "data:image/png;base64,YmFy",
                }
            },
        }

    live_source = RecorderLiveSessionSource(
        recorder=recorder,
        run_id="run-retro-live",
        live_scene_scheme="low_latency_channel",
        live_frame_supplier=load_latest_frame,
    )

    scene = live_source.load_scene(seq=1)

    assert scene is not None
    assert scene.media is not None
    assert scene.media.primary is not None
    assert scene.media.primary.media_id == "live-channel-main"
    assert live_source.lookup_media(scene.media.primary.media_id) is not None
    assert live_source.load_stream_frame(scene.media.primary.media_id) == (b"bar", "image/png")


def test_low_latency_live_source_streams_latest_rgb_frame_without_inline_snapshot_image() -> None:
    numpy = pytest.importorskip("numpy")
    recorder = ArenaVisualSessionRecorder(
        plugin_id="arena.visualization.retro.frame_v1",
        game_id="retro_mario",
        scheduling_family="real_time_tick",
        session_id="sample-rgb-live",
    )
    recorder.record_snapshot(
        ts_ms=5101,
        step=1,
        tick=1,
        snapshot={
            "step": 1,
            "tick": 1,
            "observation": {
                "board_text": "live rgb frame",
                "view": {"text": "live rgb frame"},
                "metadata": {"stream_id": "main"},
            },
        },
        label="live_rgb_frame_snapshot",
        anchor=True,
    )

    rgb_frame = numpy.zeros((4, 4, 3), dtype=numpy.uint8)
    rgb_frame[:, :, 1] = 180

    def load_latest_frame() -> dict[str, object]:
        return {
            "stream_id": "main",
            "_rgb": rgb_frame,
        }

    live_source = RecorderLiveSessionSource(
        recorder=recorder,
        run_id="run-retro-rgb-live",
        live_scene_scheme="low_latency_channel",
        live_frame_supplier=load_latest_frame,
    )

    scene = live_source.load_scene(seq=1)

    assert scene is not None
    assert scene.media is not None
    assert scene.media.primary is not None
    content, mime_type = live_source.load_stream_frame(scene.media.primary.media_id) or (None, None)

    assert mime_type == "image/jpeg"
    assert isinstance(content, bytes)
    assert len(content) > 0


def test_low_latency_live_source_preserves_png_frames_losslessly() -> None:
    recorder = _build_low_latency_frame_visual_recorder()
    png_data_url = f"data:image/png;base64,{base64.b64encode(_SMALL_PNG_BYTES).decode('ascii')}"

    def load_latest_frame() -> dict[str, object]:
        return {
            "stream_id": "pov",
            "media": {
                "primary": {
                    "mediaId": "frame-live",
                    "transport": "http_pull",
                    "mimeType": "image/png",
                    "url": png_data_url,
                }
            },
        }

    live_source = RecorderLiveSessionSource(
        recorder=recorder,
        run_id="run-low-latency-live",
        live_scene_scheme="low_latency_channel",
        live_frame_supplier=load_latest_frame,
    )

    scene = live_source.load_scene(seq=2)

    assert scene is not None
    assert scene.media is not None
    assert scene.media.primary is not None
    assert live_source.load_stream_frame(scene.media.primary.media_id) == (
        _SMALL_PNG_BYTES,
        "image/png",
    )


def test_low_latency_http_stream_keeps_png_part_type_for_source_frames(
    tmp_path: Path,
) -> None:
    recorder = _build_low_latency_frame_visual_recorder()
    png_bytes = _build_valid_png_bytes()
    png_data_url = f"data:image/png;base64,{base64.b64encode(png_bytes).decode('ascii')}"

    def load_latest_frame() -> dict[str, object]:
        return {
            "stream_id": "pov",
            "media": {
                "primary": {
                    "mediaId": "frame-live",
                    "transport": "http_pull",
                    "mimeType": "image/png",
                    "url": png_data_url,
                }
            },
        }

    live_source = RecorderLiveSessionSource(
        recorder=recorder,
        run_id="run-low-latency-live",
        live_scene_scheme="low_latency_channel",
        live_frame_supplier=load_latest_frame,
    )
    server = ArenaVisualHTTPServer(host="127.0.0.1", port=0, base_dir=tmp_path)
    server.register_live_session(live_source)
    server.start()
    try:
        host, port = server.server_address
        stream_url = (
            f"http://{host}:{port}/arena_visual/sessions/sample-low-latency-live/media/"
            "live-channel-pov/stream?run_id=run-low-latency-live"
        )

        content_type, prefix = _read_stream_prefix(stream_url)

        assert content_type.startswith("multipart/x-mixed-replace")
        assert b"Content-Type: image/png" in prefix
    finally:
        server.stop()


def test_low_latency_frame_encoder_keeps_png_when_pillow_is_available(
    monkeypatch,
) -> None:
    pil_image = __import__("PIL.Image", fromlist=["Image"])
    monkeypatch.setattr(live_session_module, "Image", pil_image)
    png_bytes = _build_valid_png_bytes()

    content, mime_type = live_session_module._encode_low_latency_frame(
        content=png_bytes,
        mime_type="image/png",
    )

    assert content == png_bytes
    assert mime_type == "image/png"


def test_binary_stream_live_scene_versions_media_id_per_scene_seq() -> None:
    recorder = _build_repeated_live_visual_recorder()
    live_source = RecorderLiveSessionSource(
        recorder=recorder,
        run_id="run-live",
        live_scene_scheme="binary_stream",
    )

    first_scene = live_source.load_scene(seq=1)
    second_scene = live_source.load_scene(seq=2)

    assert first_scene is not None
    assert second_scene is not None
    assert first_scene.media is not None
    assert second_scene.media is not None
    assert first_scene.media.primary is not None
    assert second_scene.media.primary is not None
    assert first_scene.media.primary.transport == "binary_stream"
    assert second_scene.media.primary.transport == "binary_stream"
    assert first_scene.media.primary.media_id != second_scene.media.primary.media_id
    assert live_source.load_media_content(first_scene.media.primary.media_id) == (b"foo", "image/png")
    assert live_source.load_media_content(second_scene.media.primary.media_id) == (b"foo", "image/png")


def test_live_gomoku_scene_preserves_full_15x15_board_and_legal_actions() -> None:
    recorder = ArenaVisualSessionRecorder(
        plugin_id=GOMOKU_VISUALIZATION_SPEC.plugin_id,
        game_id="gomoku",
        scheduling_family="turn",
        session_id="gomoku-live",
    )
    observation = GomokuArenaEnvironment(board_size=15, win_len=5, obs_image=True).observe("Black")

    recorder.record_decision_window_open(
        ts_ms=2001,
        step=1,
        tick=0,
        player_id="Black",
        observation=_visual_payload_snapshot(observation),
    )

    live_source = RecorderLiveSessionSource(
        recorder=recorder,
        run_id="run-live",
        visualization_spec=GOMOKU_VISUALIZATION_SPEC,
    )

    scene = live_source.load_scene(seq=1)

    assert scene is not None
    assert scene.body["board"]["size"] == 15
    assert len(scene.body["board"]["cells"]) == 225
    assert len(scene.legal_actions) == 225
    assert scene.legal_actions[0]["coord"] == "A1"
    assert scene.legal_actions[-1]["coord"] == "O15"


def test_arena_visual_http_server_serves_frontend_shell_and_assets(tmp_path: Path) -> None:
    app_dir = tmp_path / "app"
    assets_dir = app_dir / "assets"
    assets_dir.mkdir(parents=True)
    (app_dir / "index.html").write_text(
        "<!doctype html><html><head><title>Arena Visual</title></head><body>shell</body></html>",
        encoding="utf-8",
    )
    (assets_dir / "app.js").write_text("console.log('arena-visual');", encoding="utf-8")

    server = ArenaVisualHTTPServer(
        host="127.0.0.1",
        port=0,
        base_dir=tmp_path,
        app_dir=app_dir,
    )
    server.start()
    try:
        host, port = server.server_address
        root_html = _get_bytes(f"http://{host}:{port}/").decode("utf-8")
        session_html = _get_bytes(f"http://{host}:{port}/sessions/sample-1?run_id=run-9").decode("utf-8")
        asset_bytes = _get_bytes(f"http://{host}:{port}/assets/app.js")

        assert "<title>Arena Visual</title>" in root_html
        assert "<title>Arena Visual</title>" in session_html
        assert asset_bytes == b"console.log('arena-visual');"
        assert (
            server.build_viewer_url("sample-1", run_id="run-9")
            == f"http://{host}:{port}/sessions/sample-1?run_id=run-9"
        )
    finally:
        server.stop()


def test_arena_visual_request_handler_suppresses_routine_browser_poll_logs(monkeypatch) -> None:
    debug_calls: list[str] = []
    handler = object.__new__(ArenaVisualRequestHandler)
    handler.command = "GET"
    handler.path = "/arena_visual/sessions/sample-live/timeline?after_seq=1&limit=50&run_id=run-live"

    monkeypatch.setattr(
        "gage_eval.role.arena.visualization.http_server.logger.debug",
        lambda message, payload: debug_calls.append(message.format(payload)),
    )

    ArenaVisualRequestHandler.log_message(handler, '"%s" %s %s', "GET /arena_visual/sessions/sample-live/timeline HTTP/1.1", "200", "-")

    assert debug_calls == []


def test_arena_visual_request_handler_suppresses_stale_browser_poll_404_logs(monkeypatch) -> None:
    debug_calls: list[str] = []
    handler = object.__new__(ArenaVisualRequestHandler)
    handler.command = "GET"
    handler.path = "/arena_visual/sessions/sample-live?run_id=run-live"

    monkeypatch.setattr(
        "gage_eval.role.arena.visualization.http_server.logger.debug",
        lambda message, payload: debug_calls.append(message.format(payload)),
    )

    ArenaVisualRequestHandler.log_message(handler, '"%s" %s %s', "GET /arena_visual/sessions/sample-live HTTP/1.1", "404", "-")

    assert debug_calls == []


def test_arena_visual_request_handler_keeps_server_error_access_logs(monkeypatch) -> None:
    debug_calls: list[str] = []
    handler = object.__new__(ArenaVisualRequestHandler)
    handler.command = "GET"
    handler.path = "/arena_visual/sessions/sample-live?run_id=run-live"

    monkeypatch.setattr(
        "gage_eval.role.arena.visualization.http_server.logger.debug",
        lambda message, payload: debug_calls.append(message.format(payload)),
    )

    ArenaVisualRequestHandler.log_message(handler, '"%s" %s %s', "GET /arena_visual/sessions/sample-live HTTP/1.1", "500", "-")

    assert debug_calls == ['ArenaVisualServer "GET /arena_visual/sessions/sample-live HTTP/1.1" 500 -']


def test_arena_visual_http_server_reports_action_submitter_not_configured(tmp_path: Path) -> None:
    _persist_visual_session(tmp_path)
    server = ArenaVisualHTTPServer(host="127.0.0.1", port=0, base_dir=tmp_path)
    server.start()
    try:
        host, port = server.server_address
        code, payload = _read_http_error(
            Request(
            f"http://{host}:{port}/arena_visual/sessions/sample-1/actions",
            data=json.dumps({"action": {"move": "fire"}}, ensure_ascii=False).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        )
        assert code == 501
        assert payload == {"error": "action_submitter_not_configured"}
    finally:
        server.stop()


def test_arena_visual_http_server_reports_chat_submitter_not_configured(tmp_path: Path) -> None:
    _persist_visual_session(tmp_path)
    server = ArenaVisualHTTPServer(host="127.0.0.1", port=0, base_dir=tmp_path)
    server.start()
    try:
        host, port = server.server_address
        code, payload = _read_http_error(
            Request(
                f"http://{host}:{port}/arena_visual/sessions/sample-1/chat",
                data=json.dumps({"playerId": "player_0", "text": "hello"}, ensure_ascii=False).encode("utf-8"),
                headers={"Content-Type": "application/json"},
                method="POST",
            )
        )
        assert code == 501
        assert payload == {"error": "chat_submitter_not_configured"}
    finally:
        server.stop()


def test_arena_visual_http_server_reports_control_submitter_not_configured(tmp_path: Path) -> None:
    _persist_visual_session(tmp_path)
    server = ArenaVisualHTTPServer(host="127.0.0.1", port=0, base_dir=tmp_path)
    server.start()
    try:
        host, port = server.server_address
        code, payload = _read_http_error(
            Request(
                f"http://{host}:{port}/arena_visual/sessions/sample-1/control",
                data=json.dumps({"commandType": "pause"}, ensure_ascii=False).encode("utf-8"),
                headers={"Content-Type": "application/json"},
                method="POST",
            )
        )
        assert code == 501
        assert payload == {"error": "control_submitter_not_configured"}
    finally:
        server.stop()


def test_arena_visual_http_server_uses_run_id_to_disambiguate_duplicate_sessions(tmp_path: Path) -> None:
    _persist_visual_session(tmp_path, run_id="run-a", session_id="shared-sample")
    _persist_visual_session(tmp_path, run_id="run-b", session_id="shared-sample")
    server = ArenaVisualHTTPServer(host="127.0.0.1", port=0, base_dir=tmp_path)
    server.start()
    try:
        host, port = server.server_address
        code, payload = _read_http_error(
            f"http://{host}:{port}/arena_visual/sessions/shared-sample"
        )
        session_payload = _get_json(
            f"http://{host}:{port}/arena_visual/sessions/shared-sample?run_id=run-b"
        )

        assert code == 409
        assert payload == {
            "error": "session_ambiguous",
            "sessionId": "shared-sample",
        }
        assert session_payload["sessionId"] == "shared-sample"
        assert session_payload["timeline"]["eventCount"] == 6
    finally:
        server.stop()


def test_arena_visual_http_server_does_not_keep_stale_unqualified_resolution_cache(tmp_path: Path) -> None:
    _persist_visual_session(tmp_path, run_id="run-a", session_id="shared-sample")
    server = ArenaVisualHTTPServer(host="127.0.0.1", port=0, base_dir=tmp_path)
    server.start()
    try:
        host, port = server.server_address
        first_payload = _get_json(
            f"http://{host}:{port}/arena_visual/sessions/shared-sample"
        )
        _persist_visual_session(tmp_path, run_id="run-b", session_id="shared-sample")
        code, payload = _read_http_error(
            f"http://{host}:{port}/arena_visual/sessions/shared-sample"
        )

        assert first_payload["sessionId"] == "shared-sample"
        assert code == 409
        assert payload == {
            "error": "session_ambiguous",
            "sessionId": "shared-sample",
        }
    finally:
        server.stop()


def test_arena_visual_http_server_rejects_invalid_params_with_stable_error_codes(tmp_path: Path) -> None:
    _persist_visual_session(tmp_path)
    server = ArenaVisualHTTPServer(host="127.0.0.1", port=0, base_dir=tmp_path)
    server.start()
    try:
        host, port = server.server_address
        timeline_code, timeline_payload = _read_http_error(
            f"http://{host}:{port}/arena_visual/sessions/sample-1/timeline?limit=oops"
        )
        scene_code, scene_payload = _read_http_error(
            f"http://{host}:{port}/arena_visual/sessions/sample-1/scene"
        )

        assert timeline_code == 400
        assert timeline_payload == {"error": "invalid_limit"}
        assert scene_code == 400
        assert scene_payload == {"error": "missing_seq"}
    finally:
        server.stop()


def test_arena_visual_http_server_rejects_invalid_session_and_run_tokens(tmp_path: Path) -> None:
    _persist_visual_session(tmp_path)
    server = ArenaVisualHTTPServer(host="127.0.0.1", port=0, base_dir=tmp_path)
    server.start()
    try:
        host, port = server.server_address
        session_code, session_payload = _read_http_error(
            f"http://{host}:{port}/arena_visual/sessions/%2E%2E"
        )
        run_code, run_payload = _read_http_error(
            f"http://{host}:{port}/arena_visual/sessions/sample-1?run_id=../run-a"
        )

        assert session_code == 400
        assert session_payload == {"error": "invalid_session_id"}
        assert run_code == 400
        assert run_payload == {"error": "invalid_run_id"}
    finally:
        server.stop()


def test_arena_visual_http_server_rejects_invalid_json_on_action_submission(tmp_path: Path) -> None:
    _persist_visual_session(tmp_path)

    def _submitter(session_id: str, run_id: str | None, payload: dict[str, object]) -> ActionIntentReceipt:
        _ = session_id, run_id, payload
        raise AssertionError("submitter should not be called for invalid JSON")

    server = ArenaVisualHTTPServer(
        host="127.0.0.1",
        port=0,
        base_dir=tmp_path,
        action_submitter=_submitter,
    )
    server.start()
    try:
        host, port = server.server_address
        request = Request(
            f"http://{host}:{port}/arena_visual/sessions/sample-1/actions",
            data=b"{bad json",
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        code, payload = _read_http_error(request)

        assert code == 400
        assert payload == {"error": "invalid_json"}
    finally:
        server.stop()


def test_arena_visual_http_server_allows_action_callback_without_persisted_session(tmp_path: Path) -> None:
    submitted: list[tuple[str, str | None, dict[str, object]]] = []

    def _submitter(session_id: str, run_id: str | None, payload: dict[str, object]) -> ActionIntentReceipt:
        submitted.append((session_id, run_id, payload))
        return ActionIntentReceipt(intent_id="intent-live", state="pending")

    server = ArenaVisualHTTPServer(
        host="127.0.0.1",
        port=0,
        base_dir=tmp_path,
        action_submitter=_submitter,
    )
    server.start()
    try:
        host, port = server.server_address
        receipt = _post_json(
            f"http://{host}:{port}/arena_visual/sessions/live-session/actions?run_id=run-live",
            {"playerId": "human", "action": {"move": "B2"}},
        )

        assert receipt == {
            "intentId": "intent-live",
            "state": "pending",
        }
        assert submitted == [
            (
                "live-session",
                "run-live",
                {"playerId": "human", "action": {"move": "B2"}},
            )
        ]
    finally:
        server.stop()
