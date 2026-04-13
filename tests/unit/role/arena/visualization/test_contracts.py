from __future__ import annotations

import pytest

from gage_eval.role.arena.visualization.contracts import (
    ActionIntentReceipt,
    ChatMessage,
    ControlCommand,
    MediaSourceRef,
    ObserverRef,
    PlaybackState,
    SchedulingState,
    SeekSnapshotRecord,
    TimelineEvent,
    VisualScene,
    VisualSceneMedia,
    VisualSession,
)


def test_visual_session_round_trip_and_defaults() -> None:
    session = VisualSession(
        session_id="session-1",
        game_id="game-1",
        plugin_id="plugin-1",
        lifecycle="live_running",
        playback=PlaybackState(
            mode="replay_playing",
            cursor_ts=1700000012345,
            cursor_event_seq=7,
            speed=1.25,
            can_seek=True,
        ),
        observer=ObserverRef(
            observer_id="observer-1",
            observer_kind="player",
        ),
        scheduling=SchedulingState(
            family="turn",
            phase="advancing",
            accepts_human_intent=True,
            active_actor_id="actor-1",
            window_id="window-9",
        ),
        capabilities={
            "supportsReplay": True,
            "supportsSeek": True,
            "observerModes": ["player", "camera"],
        },
        summary={"turn": 5, "winner": "p1"},
        timeline={
            "cursorTs": 1700000012000,
            "cursorEventSeq": 1,
            "eventCount": 3,
            "headSeq": 1,
            "tailSeq": 9,
            "markers": {"snapshot": 1, "result": 1},
        },
    )

    payload = session.to_dict()

    assert payload == {
        "sessionId": "session-1",
        "gameId": "game-1",
        "pluginId": "plugin-1",
        "lifecycle": "live_running",
        "playback": {
            "mode": "replay_playing",
            "cursorTs": 1700000012345,
            "cursorEventSeq": 7,
            "speed": 1.25,
            "canSeek": True,
        },
        "observer": {
            "observerId": "observer-1",
            "observerKind": "player",
        },
        "scheduling": {
            "family": "turn",
            "phase": "advancing",
            "acceptsHumanIntent": True,
            "activeActorId": "actor-1",
            "windowId": "window-9",
        },
        "capabilities": {
            "supportsReplay": True,
            "supportsSeek": True,
            "observerModes": ["player", "camera"],
        },
        "summary": {"turn": 5, "winner": "p1"},
        "timeline": {
            "cursorTs": 1700000012000,
            "cursorEventSeq": 1,
            "eventCount": 3,
            "headSeq": 1,
            "tailSeq": 9,
            "markers": {"snapshot": 1, "result": 1},
        },
    }
    assert VisualSession.from_dict(payload) == session

    default_session = VisualSession(
        session_id="session-2",
        game_id="game-2",
        plugin_id="plugin-2",
    )
    assert default_session.lifecycle == "initializing"
    assert default_session.playback == PlaybackState()
    assert default_session.observer == ObserverRef()
    assert default_session.scheduling == SchedulingState()
    assert default_session.capabilities == {}
    assert default_session.summary == {}
    assert default_session.timeline == {}
    assert default_session.to_dict() == {
        "sessionId": "session-2",
        "gameId": "game-2",
        "pluginId": "plugin-2",
        "lifecycle": "initializing",
        "playback": {
            "mode": "live_tail",
            "cursorTs": 0,
            "cursorEventSeq": 0,
            "speed": 1.0,
            "canSeek": True,
        },
        "observer": {
            "observerId": "",
            "observerKind": "spectator",
        },
        "scheduling": {
            "family": "turn",
            "phase": "idle",
            "acceptsHumanIntent": False,
        },
        "capabilities": {},
        "summary": {},
        "timeline": {},
    }


def test_timeline_event_round_trip_and_shape() -> None:
    event = TimelineEvent(
        seq=7,
        ts_ms=1700000012345,
        type="system_marker",
        label="Pause requested",
        actor_id="observer-1",
        ref_snapshot_seq=6,
        detail="manual pause",
        severity="warn",
        tags=("ui", "pause"),
        payload={"step": "pause", "source": "ui"},
    )

    payload = event.to_dict()

    assert payload == {
        "seq": 7,
        "tsMs": 1700000012345,
        "type": "system_marker",
        "label": "Pause requested",
        "actorId": "observer-1",
        "refSnapshotSeq": 6,
        "detail": "manual pause",
        "severity": "warn",
        "tags": ["ui", "pause"],
        "payload": {"step": "pause", "source": "ui"},
    }
    assert TimelineEvent.from_dict(payload) == event

    default_event = TimelineEvent(seq=8, ts_ms=1700000013000, type="snapshot", label="snapshot")
    assert default_event.to_dict() == {
        "seq": 8,
        "tsMs": 1700000013000,
        "type": "snapshot",
        "label": "snapshot",
    }


def test_media_source_ref_round_trip_and_defaults() -> None:
    ref = MediaSourceRef(
        media_id="media-1",
        transport="http_pull",
        mime_type="image/png",
        url="https://example.invalid/frame.png",
        preview_ref="preview-1",
    )

    payload = ref.to_dict()

    assert payload == {
        "mediaId": "media-1",
        "transport": "http_pull",
        "mimeType": "image/png",
        "url": "https://example.invalid/frame.png",
        "previewRef": "preview-1",
    }
    assert MediaSourceRef.from_dict(payload) == ref

    local_ref = MediaSourceRef(media_id="media-2", transport="artifact_ref")
    assert local_ref.to_dict() == {
        "mediaId": "media-2",
        "transport": "artifact_ref",
    }


def test_control_command_round_trip_for_seek() -> None:
    command = ControlCommand(
        command_type="seek_seq",
        target_seq=42,
        issued_by=ObserverRef(observer_id="ops", observer_kind="global"),
    )

    payload = command.to_dict()

    assert payload == {
        "commandType": "seek_seq",
        "targetSeq": 42,
        "issuedBy": {
            "observerId": "ops",
            "observerKind": "global",
        },
    }
    assert ControlCommand.from_dict(payload) == command


def test_control_command_round_trip_for_restart() -> None:
    command = ControlCommand(
        command_type="restart",
        issued_by=ObserverRef(observer_id="player_0", observer_kind="player"),
    )

    payload = command.to_dict()

    assert payload == {
        "commandType": "restart",
        "issuedBy": {
            "observerId": "player_0",
            "observerKind": "player",
        },
    }
    assert ControlCommand.from_dict(payload) == command


@pytest.mark.parametrize(
    ("payload", "expected_payload"),
    [
        (
            {"commandType": "follow_tail"},
            {"commandType": "follow_tail"},
        ),
        (
            {"commandType": "pause"},
            {"commandType": "pause"},
        ),
        (
            {"commandType": "replay"},
            {"commandType": "replay"},
        ),
        (
            {"commandType": "seek_end"},
            {"commandType": "seek_end"},
        ),
        (
            {"commandType": "step", "stepDelta": 1},
            {"commandType": "step", "stepDelta": 1},
        ),
        (
            {"commandType": "set_speed", "speed": 2.5},
            {"commandType": "set_speed", "speed": 2.5},
        ),
        (
            {"commandType": "back_to_tail"},
            {"commandType": "back_to_tail"},
        ),
    ],
)
def test_control_command_round_trip_for_phase2_vocabulary(
    payload: dict[str, object],
    expected_payload: dict[str, object],
) -> None:
    command = ControlCommand.from_dict(payload)
    serialized = command.to_dict()

    assert serialized == expected_payload
    assert ControlCommand.from_dict(serialized) == command


@pytest.mark.parametrize(
    "payload",
    [
        {"commandType": "seek_seq"},
        {"commandType": "step"},
        {"commandType": "step", "stepDelta": 0},
        {"commandType": "step", "stepDelta": 2},
        {"commandType": "set_speed"},
        {"commandType": "set_speed", "speed": 0},
        {"commandType": "set_speed", "speed": -1.5},
    ],
)
def test_control_command_invalid_payload_validation(payload: dict[str, object]) -> None:
    with pytest.raises(ValueError):
        ControlCommand.from_dict(payload)


@pytest.mark.parametrize(
    "payload",
    [
        {"commandType": "pause", "targetSeq": 3},
        {"commandType": "pause", "stepDelta": 1},
        {"commandType": "pause", "speed": 1.25},
        {"commandType": "follow_tail", "targetSeq": 3},
        {"commandType": "follow_tail", "stepDelta": 1},
        {"commandType": "follow_tail", "speed": 1.25},
        {"commandType": "replay", "targetSeq": 3},
        {"commandType": "replay", "stepDelta": 1},
        {"commandType": "replay", "speed": 1.25},
        {"commandType": "seek_end", "targetSeq": 3},
        {"commandType": "seek_end", "stepDelta": 1},
        {"commandType": "seek_end", "speed": 1.25},
        {"commandType": "seek_seq", "targetSeq": 42, "stepDelta": 1},
        {"commandType": "seek_seq", "targetSeq": 42, "speed": 1.25},
        {"commandType": "step", "stepDelta": 1, "targetSeq": 42},
        {"commandType": "step", "stepDelta": 1, "speed": 1.25},
        {"commandType": "set_speed", "speed": 1.25, "targetSeq": 42},
        {"commandType": "set_speed", "speed": 1.25, "stepDelta": 1},
        {"commandType": "back_to_tail", "targetSeq": 3},
        {"commandType": "back_to_tail", "stepDelta": 1},
        {"commandType": "back_to_tail", "speed": 1.25},
    ],
)
def test_control_command_rejects_incompatible_extra_fields(payload: dict[str, object]) -> None:
    with pytest.raises(ValueError):
        ControlCommand.from_dict(payload)


def test_chat_message_round_trip() -> None:
    message = ChatMessage(
        player_id="p0",
        text="hold position",
        channel="table",
    )

    payload = message.to_dict()

    assert payload == {
        "playerId": "p0",
        "text": "hold position",
        "channel": "table",
    }
    assert ChatMessage.from_dict(payload) == message


@pytest.mark.parametrize("snapshot_mode", ["full", "media_ref"])
def test_seek_snapshot_record_round_trip(snapshot_mode: str) -> None:
    record = SeekSnapshotRecord(
        seq=15,
        ts_ms=1234,
        snapshot_mode=snapshot_mode,
        snapshot_ref="snapshots/seq-000015.json",
    )

    payload = record.to_dict()

    assert payload == {
        "seq": 15,
        "tsMs": 1234,
        "snapshotMode": snapshot_mode,
        "snapshotRef": "snapshots/seq-000015.json",
    }
    assert SeekSnapshotRecord.from_dict(payload) == record


def test_visual_scene_serialization_includes_nested_media_refs() -> None:
    scene = VisualScene(
        scene_id="scene-3",
        game_id="game-3",
        plugin_id="plugin-3",
        kind="frame",
        ts_ms=1700000015555,
        seq=3,
        phase="live",
        active_player_id="player-0",
        legal_actions=(
            {"id": "noop", "label": "No-op"},
            {"id": "advance", "label": "Advance"},
        ),
        summary={"turn": 9, "score": {"p0": 12, "p1": 8}},
        body={"frameRef": "media-1", "note": "live capture"},
        media=VisualSceneMedia(
            primary=MediaSourceRef(
                media_id="media-1",
                transport="binary_stream",
                mime_type="image/jpeg",
                preview_ref="preview-1",
            ),
            auxiliary=(
                MediaSourceRef(
                    media_id="media-2",
                    transport="artifact_ref",
                    mime_type="image/jpeg",
                    url="https://example.invalid/frame-3.jpg",
                ),
            ),
        ),
        overlays=(
            {"kind": "cursor", "x": 14, "y": 22},
        ),
    )

    payload = scene.to_dict()

    assert payload == {
        "sceneId": "scene-3",
        "gameId": "game-3",
        "pluginId": "plugin-3",
        "kind": "frame",
        "tsMs": 1700000015555,
        "seq": 3,
        "phase": "live",
        "activePlayerId": "player-0",
        "legalActions": [
            {"id": "noop", "label": "No-op"},
            {"id": "advance", "label": "Advance"},
        ],
        "summary": {"turn": 9, "score": {"p0": 12, "p1": 8}},
        "body": {"frameRef": "media-1", "note": "live capture"},
        "media": {
            "primary": {
                "mediaId": "media-1",
                "transport": "binary_stream",
                "mimeType": "image/jpeg",
                "previewRef": "preview-1",
            },
            "auxiliary": [
                {
                    "mediaId": "media-2",
                    "transport": "artifact_ref",
                    "mimeType": "image/jpeg",
                    "url": "https://example.invalid/frame-3.jpg",
                },
            ],
        },
        "overlays": [{"kind": "cursor", "x": 14, "y": 22}],
    }
    assert VisualScene.from_dict(payload) == scene

    bare_scene = VisualScene(
        scene_id="scene-4",
        game_id="game-4",
        plugin_id="plugin-4",
        kind="board",
        ts_ms=1700000016666,
        seq=4,
        phase="replay",
        active_player_id="player-1",
    )
    assert bare_scene.to_dict() == {
        "sceneId": "scene-4",
        "gameId": "game-4",
        "pluginId": "plugin-4",
        "kind": "board",
        "tsMs": 1700000016666,
        "seq": 4,
        "phase": "replay",
        "activePlayerId": "player-1",
        "legalActions": [],
        "summary": {},
        "body": {},
    }


def test_action_intent_receipt_round_trip_and_status_default() -> None:
    receipt = ActionIntentReceipt(
        intent_id="intent-1",
        state="accepted",
        related_event_seq=42,
        reason="validated",
    )

    payload = receipt.to_dict()

    assert payload == {
        "intentId": "intent-1",
        "state": "accepted",
        "relatedEventSeq": 42,
        "reason": "validated",
    }
    assert ActionIntentReceipt.from_dict(payload) == receipt

    default_receipt = ActionIntentReceipt(intent_id="intent-2")
    assert default_receipt.state == "pending"
    assert default_receipt.related_event_seq is None
    assert default_receipt.reason is None
    assert default_receipt.to_dict() == {
        "intentId": "intent-2",
        "state": "pending",
    }


def test_boundary_validation_rejects_invalid_enum_and_bool_payloads() -> None:
    with pytest.raises(ValueError):
        VisualSession(
            session_id="session-x",
            game_id="game-x",
            plugin_id="plugin-x",
            lifecycle="bad_state",
        )

    with pytest.raises(ValueError):
        MediaSourceRef(media_id="media-x", transport="invalid_transport")

    with pytest.raises(ValueError):
        TimelineEvent(seq=1, ts_ms=1, type="bad_type", label="x")

    with pytest.raises(TypeError):
        PlaybackState.from_dict({"canSeek": "false"})

    with pytest.raises(TypeError):
        SchedulingState.from_dict({"acceptsHumanIntent": "true"})

    with pytest.raises(TypeError):
        VisualSession.from_dict(
            {
                "sessionId": "session-x",
                "gameId": "game-x",
                "pluginId": "plugin-x",
                "playback": {"canSeek": "false"},
                "observer": {},
                "scheduling": {},
                "capabilities": {},
                "summary": {},
                "timeline": {},
            }
        )
