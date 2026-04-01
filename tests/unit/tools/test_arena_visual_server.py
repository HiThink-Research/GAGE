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
from gage_eval.game_kits.real_time_game.openra.visualization import (
    VISUALIZATION_SPEC as OPENRA_VISUALIZATION_SPEC,
)
from gage_eval.role.arena.core.game_session import _visual_payload_snapshot
from gage_eval.game_kits.board_game.gomoku.environment import GomokuArenaEnvironment
from gage_eval.role.arena.types import GameResult
from gage_eval.role.arena.runtime_services import ArenaRuntimeServiceHub
from gage_eval.role.arena.visualization.contracts import ActionIntentReceipt
from gage_eval.role.arena.visualization.http_server import (
    ArenaVisualHTTPServer,
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


def _build_openra_live_visual_recorder(
    *,
    session_id: str = "sample-openra-live",
) -> ArenaVisualSessionRecorder:
    recorder = ArenaVisualSessionRecorder(
        plugin_id=OPENRA_VISUALIZATION_SPEC.plugin_id,
        game_id="openra",
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
            "board_text": "Native OpenRA runtime",
            "view": {
                "text": "Native OpenRA runtime",
            },
            "legal_actions": {
                "items": [
                    {"id": "noop", "label": "No-op"},
                    {"id": "bridge_input", "label": "Native input"},
                ]
            },
            "metadata": {
                "stream_id": "main",
                "map_id": "ra_skirmish_1v1",
                "map": {
                    "id": "ra_skirmish_1v1",
                    "mod_id": "ra",
                    "title": "Marigold Town",
                    "preview_source": "native_runtime",
                    "image_size": {"width": 1280, "height": 720},
                },
                "selection": {
                    "unit_ids": [],
                    "primary_unit_id": None,
                },
                "economy": {
                    "credits": 0,
                    "income_per_minute": 0,
                    "power": {"produced": 0, "used": 0},
                },
                "objectives": [],
                "units": [],
                "production": {"queues": []},
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
            "stream_id": "main",
            "observation": {
                "active_player": "player_0",
                "board_text": "Native OpenRA runtime",
                "view": {
                    "text": "Native OpenRA runtime",
                },
                "legal_actions": {
                    "items": [
                        {"id": "noop", "label": "No-op"},
                        {"id": "bridge_input", "label": "Native input"},
                    ]
                },
                "metadata": {
                    "stream_id": "main",
                    "map_id": "ra_skirmish_1v1",
                    "map": {
                        "id": "ra_skirmish_1v1",
                        "mod_id": "ra",
                        "title": "Marigold Town",
                        "preview_source": "native_runtime",
                        "image_size": {"width": 1280, "height": 720},
                    },
                    "selection": {
                        "unit_ids": [],
                        "primary_unit_id": None,
                    },
                    "economy": {
                        "credits": 0,
                        "income_per_minute": 0,
                        "power": {"produced": 0, "used": 0},
                    },
                    "objectives": [],
                    "units": [],
                    "production": {"queues": []},
                },
            },
            "media": {
                "primary": {
                    "mediaId": "openra-frame-live",
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
        label="openra_live_snapshot",
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


def test_arena_visual_http_server_serves_registered_low_latency_openra_rts_stream_without_manifest(
    tmp_path: Path,
) -> None:
    recorder = _build_openra_live_visual_recorder()
    live_source = RecorderLiveSessionSource(
        recorder=recorder,
        run_id="run-openra-live",
        live_scene_scheme="low_latency_channel",
    )
    server = ArenaVisualHTTPServer(host="127.0.0.1", port=0, base_dir=tmp_path)
    server.register_live_session(live_source)
    server.start()
    try:
        host, port = server.server_address
        scene_url = (
            f"http://{host}:{port}/arena_visual/sessions/sample-openra-live/scene"
            f"?seq=2&run_id=run-openra-live"
        )

        scene_payload = _get_json(scene_url)
        assert scene_payload["kind"] == "rts"
        primary_media = scene_payload["media"]["primary"]
        assert isinstance(primary_media, dict)
        media_id = str(primary_media["mediaId"])
        media_payload = _get_json(
            "http://"
            f"{host}:{port}/arena_visual/sessions/sample-openra-live/media/{media_id}"
            "?run_id=run-openra-live"
        )
        stream_url = str(primary_media["url"])
        if stream_url.startswith("/"):
            stream_url = f"http://{host}:{port}{stream_url}"
        content_type, prefix = _read_stream_prefix(stream_url)

        assert primary_media["transport"] == "low_latency_channel"
        assert media_payload["transport"] == "low_latency_channel"
        assert str(primary_media["url"]).endswith(
            "/arena_visual/sessions/sample-openra-live/media/"
            f"{media_id}/stream?run_id=run-openra-live"
        )
        assert content_type.startswith("multipart/x-mixed-replace")
        assert b"--frame" in prefix
        assert b"Content-Type:" in prefix
    finally:
        server.stop()


def test_openra_low_latency_live_source_prefers_latest_environment_frame_supplier() -> None:
    recorder = _build_openra_live_visual_recorder()

    def load_latest_frame() -> dict[str, object]:
        return {
            "stream_id": "main",
            "media": {
                "primary": {
                    "mediaId": "openra-frame-live",
                    "transport": "http_pull",
                    "mimeType": "image/png",
                    "url": "data:image/png;base64,YmFy",
                }
            },
        }

    live_source = RecorderLiveSessionSource(
        recorder=recorder,
        run_id="run-openra-live",
        live_scene_scheme="low_latency_channel",
        live_frame_supplier=load_latest_frame,
    )

    scene = live_source.load_scene(seq=2)

    assert scene is not None
    assert scene.media is not None
    assert scene.media.primary is not None
    assert scene.media.primary.media_id == "live-channel-main"
    assert live_source.load_stream_frame(scene.media.primary.media_id) == (b"bar", "image/png")


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


def test_openra_low_latency_live_source_preserves_native_png_frames_losslessly() -> None:
    recorder = _build_openra_live_visual_recorder()
    png_data_url = f"data:image/png;base64,{base64.b64encode(_SMALL_PNG_BYTES).decode('ascii')}"

    def load_latest_frame() -> dict[str, object]:
        return {
            "stream_id": "main",
            "media": {
                "primary": {
                    "mediaId": "openra-frame-live",
                    "transport": "http_pull",
                    "mimeType": "image/png",
                    "url": png_data_url,
                }
            },
        }

    live_source = RecorderLiveSessionSource(
        recorder=recorder,
        run_id="run-openra-live",
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


def test_openra_low_latency_http_stream_keeps_png_part_type_for_native_frames(
    tmp_path: Path,
) -> None:
    recorder = _build_openra_live_visual_recorder()
    png_bytes = _build_valid_png_bytes()
    png_data_url = f"data:image/png;base64,{base64.b64encode(png_bytes).decode('ascii')}"

    def load_latest_frame() -> dict[str, object]:
        return {
            "stream_id": "main",
            "media": {
                "primary": {
                    "mediaId": "openra-frame-live",
                    "transport": "http_pull",
                    "mimeType": "image/png",
                    "url": png_data_url,
                }
            },
        }

    live_source = RecorderLiveSessionSource(
        recorder=recorder,
        run_id="run-openra-live",
        live_scene_scheme="low_latency_channel",
        live_frame_supplier=load_latest_frame,
    )
    server = ArenaVisualHTTPServer(host="127.0.0.1", port=0, base_dir=tmp_path)
    server.register_live_session(live_source)
    server.start()
    try:
        host, port = server.server_address
        stream_url = (
            f"http://{host}:{port}/arena_visual/sessions/sample-openra-live/media/"
            "live-channel-main/stream?run_id=run-openra-live"
        )

        content_type, prefix = _read_stream_prefix(stream_url)

        assert content_type.startswith("multipart/x-mixed-replace")
        assert b"Content-Type: image/png" in prefix
    finally:
        server.stop()


def test_openra_low_latency_frame_encoder_keeps_png_when_pillow_is_available(
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
