from __future__ import annotations

import json
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import Request, urlopen

from gage_eval.role.arena.types import GameResult
from gage_eval.role.arena.visualization.contracts import ActionIntentReceipt
from gage_eval.role.arena.visualization.http_server import ArenaVisualHTTPServer
from gage_eval.role.arena.visualization.recorder import ArenaVisualSessionRecorder


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


def _get_bytes(url: str) -> bytes:
    with urlopen(url) as response:  # noqa: S310 - local test endpoint
        return response.read()


def _read_http_error(target: str | Request) -> tuple[int, dict[str, object]]:
    try:
        with urlopen(target) as _:
            raise AssertionError("Expected HTTPError")
    except HTTPError as exc:
        payload = json.loads(exc.read().decode("utf-8"))
        assert isinstance(payload, dict)
        return exc.code, payload


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
