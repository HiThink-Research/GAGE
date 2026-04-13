from __future__ import annotations

import base64
import json
from pathlib import Path

import pytest

from gage_eval.game_kits.board_game.gomoku.visualization import (
    VISUALIZATION_SPEC as GOMOKU_VISUALIZATION_SPEC,
)
from gage_eval.game_kits.phase_card_game.doudizhu.visualization import (
    VISUALIZATION_SPEC as DOUDIZHU_VISUALIZATION_SPEC,
)
from gage_eval.game_kits.real_time_game.retro_platformer.visualization import (
    VISUALIZATION_SPEC as RETRO_VISUALIZATION_SPEC,
)
from gage_eval.game_kits.contracts import GameVisualizationSpec
from gage_eval.role.arena.types import GameResult
from gage_eval.role.arena.visualization.contracts import ObserverRef, VisualSession
from gage_eval.role.arena.visualization.gateway_service import ArenaVisualGatewayQueryService
from gage_eval.role.arena.visualization.recorder import ArenaVisualSessionRecorder


def _build_visualization_spec() -> GameVisualizationSpec:
    return GameVisualizationSpec(
        spec_id="arena/visualization/pettingzoo_frame_v1",
        plugin_id="arena.visualization.pettingzoo.frame_v1",
        visual_kind="frame",
    )


def _persist_visual_session(tmp_path: Path) -> Path:
    replay_path = tmp_path / "runs" / "run-1" / "replays" / "sample-1" / "replay.json"
    recorder = ArenaVisualSessionRecorder(
        plugin_id="arena.visualization.pettingzoo.frame_v1",
        game_id="pettingzoo",
        scheduling_family="turn",
        session_id="sample-1",
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

    return recorder.persist(replay_path).manifest_path


def _persist_table_visual_session(tmp_path: Path) -> Path:
    session_dir = tmp_path / "runs" / "run-table" / "replays" / "table-sample" / "arena_visual_session" / "v1"
    session_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = session_dir / "manifest.json"
    index_path = session_dir / "index.json"
    timeline_path = session_dir / "timeline.jsonl"
    snapshot_dir = session_dir / "snapshots"
    snapshot_dir.mkdir(exist_ok=True)
    snapshot_path = snapshot_dir / "seq-7.json"

    manifest_payload = {
        "visualSession": VisualSession(
            session_id="table-sample",
            game_id="doudizhu",
            plugin_id=DOUDIZHU_VISUALIZATION_SPEC.plugin_id,
            lifecycle="closed",
            observer=ObserverRef(observer_id="host", observer_kind="global"),
            timeline={"eventCount": 1, "tailSeq": 7},
        ).to_dict(),
        "artifacts": {
            "index_ref": "index.json",
            "timeline_ref": "timeline.jsonl",
        },
        "timeline": {
            "indexRef": "index.json",
            "timelineRef": "timeline.jsonl",
        },
    }
    index_payload = {
        "snapshotAnchors": [
            {
                "seq": 7,
                "snapshotRef": "snapshots/seq-7.json",
            }
        ],
        "markers": {
            "snapshot": [7],
        },
    }
    event_payload = {
        "seq": 7,
        "tsMs": 1007,
        "type": "snapshot",
        "label": "snapshot",
    }
    snapshot_payload = {
        "body": {
            "active_player_id": "player_0",
            "observer_player_id": "player_2",
            "player_ids": ["player_0", "player_1", "player_2"],
            "player_names": {
                "player_0": "Player 0",
                "player_1": "Player 1",
                "player_2": "Player 2",
            },
            "public_state": {
                "landlord_id": "player_0",
                "num_cards_left": {"player_0": 2, "player_1": 2, "player_2": 2},
                "played_cards": [
                    {"player_id": "player_0", "cards": ["3"]},
                    {"player_id": "player_1", "cards": []},
                    {"player_id": "player_2", "cards": []},
                ],
                "seen_cards": ["3"],
                "trace": [{"player": 0, "action": "3"}],
            },
            "private_state": {
                "self_id": "player_0",
                "current_hand": ["3", "4"],
                "current_hand_text": "3, 4",
            },
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
            },
            "legal_moves": ["pass", "4"],
        }
    }

    manifest_path.write_text(json.dumps(manifest_payload, ensure_ascii=False, indent=2), encoding="utf-8")
    index_path.write_text(json.dumps(index_payload, ensure_ascii=False, indent=2), encoding="utf-8")
    timeline_path.write_text(json.dumps(event_payload, ensure_ascii=False) + "\n", encoding="utf-8")
    snapshot_path.write_text(json.dumps(snapshot_payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return manifest_path


def _persist_visual_session_with_dedicated_seek_snapshots(
    tmp_path: Path,
    *,
    seek_snapshot_seqs: tuple[int, ...] = (5, 8),
    event_ref_snapshot_seq: int | None = None,
) -> Path:
    session_dir = tmp_path / "runs" / "run-seek" / "replays" / "seek-sample" / "arena_visual_session" / "v1"
    session_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = session_dir / "manifest.json"
    index_path = session_dir / "index.json"
    timeline_path = session_dir / "timeline.jsonl"
    seek_snapshots_path = session_dir / "seek_snapshots.json"
    snapshot_dir = session_dir / "snapshots"
    snapshot_dir.mkdir(exist_ok=True)
    snapshot_path_5 = snapshot_dir / "seq-5.json"
    snapshot_path_8 = snapshot_dir / "seq-8.json"

    manifest_payload = {
        "visualSession": VisualSession(
            session_id="seek-sample",
            game_id="gomoku",
            plugin_id=GOMOKU_VISUALIZATION_SPEC.plugin_id,
            lifecycle="closed",
            observer=ObserverRef(observer_id="", observer_kind="spectator"),
            timeline={"eventCount": 3, "tailSeq": 8},
        ).to_dict(),
        "artifacts": {
            "index_ref": "index.json",
            "timeline_ref": "timeline.jsonl",
            "seek_snapshots_ref": "seek_snapshots.json",
        },
        "timeline": {
            "indexRef": "index.json",
            "timelineRef": "timeline.jsonl",
            "snapshotAnchors": [
                {
                    "seq": 5,
                    "label": "legacy-anchor-5",
                    "snapshotRef": "snapshots/seq-5.json",
                },
                {
                    "seq": 8,
                    "label": "legacy-anchor-8",
                    "snapshotRef": "snapshots/seq-8.json",
                }
            ],
        },
    }
    index_payload = {
        "snapshotAnchors": [
            {
                "seq": 5,
                "label": "legacy-anchor-5",
                "snapshotRef": "snapshots/seq-5.json",
            },
            {
                "seq": 8,
                "label": "legacy-anchor-8",
                "snapshotRef": "snapshots/seq-8.json",
            }
        ],
        "markers": {
            "snapshot": [5, 8],
            "result": [7],
        },
    }
    seek_snapshot_payload = {
        "seekSnapshots": [
            payload
            for payload in (
                {
                    "seq": 5,
                    "tsMs": 1005,
                    "snapshotMode": "full",
                    "snapshotRef": "snapshots/seq-5.json",
                },
                {
                    "seq": 8,
                    "tsMs": 1008,
                    "snapshotMode": "full",
                    "snapshotRef": "snapshots/seq-8.json",
                },
            )
            if int(payload["seq"]) in set(seek_snapshot_seqs)
        ]
    }
    events = [
        {
            "seq": 5,
            "tsMs": 1005,
            "type": "snapshot",
            "label": "snapshot",
        },
        {
            "seq": 7,
            "tsMs": 1007,
            "type": "result",
            "label": "result",
        },
        {
            "seq": 8,
            "tsMs": 1008,
            "type": "snapshot",
            "label": "snapshot",
            "refSnapshotSeq": event_ref_snapshot_seq,
        },
    ]
    snapshot_path_5.write_text(
        json.dumps({"body": {"board": {"state": "older-anchor"}, "step": 5}}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    snapshot_path_8.write_text(
        json.dumps({"body": {"board": {"state": "dedicated-anchor"}, "step": 8}}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    manifest_path.write_text(json.dumps(manifest_payload, ensure_ascii=False, indent=2), encoding="utf-8")
    index_path.write_text(json.dumps(index_payload, ensure_ascii=False, indent=2), encoding="utf-8")
    seek_snapshots_path.write_text(json.dumps(seek_snapshot_payload, ensure_ascii=False, indent=2), encoding="utf-8")
    timeline_path.write_text(
        "\n".join(json.dumps(event, ensure_ascii=False) for event in events) + "\n",
        encoding="utf-8",
    )
    return manifest_path


def _persist_runtime_style_recorded_session(
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


def test_gateway_service_loads_visual_session_manifest(tmp_path: Path) -> None:
    manifest_path = _persist_visual_session(tmp_path)
    service = ArenaVisualGatewayQueryService(visualization_spec=_build_visualization_spec())

    session = service.load_session(manifest_path)

    assert session.session_id == "sample-1"
    assert session.plugin_id == "arena.visualization.pettingzoo.frame_v1"
    assert session.lifecycle == "closed"
    assert session.timeline["eventCount"] == 6
    assert session.timeline["tailSeq"] == 6


def test_gateway_service_applies_observer_override_to_session_and_scene(tmp_path: Path) -> None:
    manifest_path = _persist_table_visual_session(tmp_path)
    service = ArenaVisualGatewayQueryService(visualization_spec=DOUDIZHU_VISUALIZATION_SPEC)

    spectator_scene = service.load_scene(
        manifest_path,
        seq=7,
        observer=ObserverRef(observer_id="", observer_kind="spectator"),
    )
    player_session = service.load_session(
        manifest_path,
        observer=ObserverRef(observer_id="player_0", observer_kind="player"),
    )
    player_scene = service.load_scene(
        manifest_path,
        seq=7,
        observer=ObserverRef(observer_id="player_0", observer_kind="player"),
    )

    assert spectator_scene is not None
    spectator_seats = {seat["playerId"]: seat for seat in spectator_scene.body["table"]["seats"]}
    assert spectator_scene.body["status"]["observerPlayerId"] is None
    assert spectator_seats["player_0"]["hand"] == {
        "isVisible": False,
        "cards": [],
        "maskedCount": 2,
    }

    assert player_session.observer == ObserverRef(observer_id="player_0", observer_kind="player")
    assert player_scene is not None
    player_seats = {seat["playerId"]: seat for seat in player_scene.body["table"]["seats"]}
    assert player_scene.body["status"]["observerPlayerId"] == "player_0"
    assert player_seats["player_0"]["hand"] == {
        "isVisible": True,
        "cards": ["3", "4"],
        "maskedCount": 0,
    }


def test_gateway_service_projects_runtime_style_board_snapshot_from_recorder(tmp_path: Path) -> None:
    manifest_path = _persist_runtime_style_recorded_session(
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
                "view": {
                    "text": "   A B C\n 3 . . .\n 2 . W .\n 1 B . .",
                    "image": {
                        "data_url": "data:image/png;base64,ZmFrZQ==",
                        "shape": [64, 64, 3],
                    },
                },
                "legal_actions": {"items": ["B1", "C1", "A3"]},
                "context": {"mode": "turn", "step": 5},
            },
            "arenaTrace": None,
            "result": None,
        },
    )
    service = ArenaVisualGatewayQueryService(visualization_spec=GOMOKU_VISUALIZATION_SPEC)

    scene = service.load_scene(manifest_path, seq=2)

    assert scene is not None
    assert scene.body["board"]["size"] == 3
    assert len(scene.body["board"]["cells"]) == 9
    assert any(cell["occupant"] == "B" for cell in scene.body["board"]["cells"])
    assert {action["coord"] for action in scene.legal_actions} == {"B1", "C1", "A3"}


def test_gateway_service_projects_runtime_style_table_snapshot_from_recorder(tmp_path: Path) -> None:
    manifest_path = _persist_runtime_style_recorded_session(
        tmp_path,
        session_id="doudizhu-runtime",
        game_id="doudizhu",
        plugin_id=DOUDIZHU_VISUALIZATION_SPEC.plugin_id,
        snapshot_body={
            "step": 4,
            "tick": 4,
            "playerId": None,
            "observation": {
                "board_text": "Public State: ...",
                "legal_moves": ["pass", "4"],
                "active_player": "player_0",
                "metadata": {
                    "public_state": {
                        "landlord_id": "player_0",
                        "num_cards_left": {"player_0": 2, "player_1": 2, "player_2": 2},
                        "played_cards": [
                            {"player_id": "player_0", "cards": ["3"]},
                            {"player_id": "player_1", "cards": []},
                            {"player_id": "player_2", "cards": []},
                        ],
                        "seen_cards": ["3"],
                    },
                    "private_state": {
                        "self_id": "player_0",
                        "current_hand": ["3", "4"],
                    },
                    "player_id": "player_0",
                    "active_player_id": "player_0",
                    "chat_log": [{"player_id": "player_1", "text": "watch this"}],
                },
                "view": {"text": "Public State: ..."},
                "legal_actions": {"items": ["pass", "4"]},
                "context": {"mode": "turn"},
            },
            "arenaTrace": None,
            "result": None,
        },
    )
    service = ArenaVisualGatewayQueryService(visualization_spec=DOUDIZHU_VISUALIZATION_SPEC)

    spectator_scene = service.load_scene(manifest_path, seq=2)
    player_scene = service.load_scene(
        manifest_path,
        seq=2,
        observer=ObserverRef(observer_id="player_0", observer_kind="player"),
    )

    assert spectator_scene is not None
    assert len(spectator_scene.body["table"]["seats"]) == 3
    assert spectator_scene.body["table"]["center"]["cards"] == ["3"]
    spectator_seats = {seat["playerId"]: seat for seat in spectator_scene.body["table"]["seats"]}
    assert spectator_scene.body["status"]["privateViewPlayerId"] == "player_0"
    assert spectator_seats["player_0"]["hand"] == {
        "isVisible": False,
        "cards": [],
        "maskedCount": 2,
    }

    assert player_scene is not None
    player_seats = {seat["playerId"]: seat for seat in player_scene.body["table"]["seats"]}
    assert player_scene.body["status"]["observerPlayerId"] == "player_0"
    assert player_seats["player_0"]["hand"] == {
        "isVisible": True,
        "cards": ["3", "4"],
        "maskedCount": 0,
    }


def test_gateway_service_projects_runtime_style_frame_snapshot_from_recorder(tmp_path: Path) -> None:
    inline_frame_data_url = "data:image/png;base64," + ("QUJD" * 3000)
    manifest_path = _persist_runtime_style_recorded_session(
        tmp_path,
        session_id="retro-runtime",
        game_id="retro_platformer",
        plugin_id=RETRO_VISUALIZATION_SPEC.plugin_id,
        snapshot_body={
            "step": 4,
            "tick": 4,
            "playerId": None,
            "observation": {
                "board_text": "Mode: tick\nDecision step: 3\nTick: 3",
                "legal_moves": ["noop", "right", "jump"],
                "active_player": "mario",
                "last_move": "right",
                "metadata": {
                    "game_type": "SuperMarioBros3-Nes-v0",
                    "player_id": "mario",
                    "reward": 12.5,
                },
                "view": {
                    "text": "Mode: tick\nDecision step: 3\nTick: 3",
                    "image": {
                        "data_url": inline_frame_data_url,
                        "shape": [240, 256, 3],
                    },
                },
                "legal_actions": {
                    "items": [
                        {"id": "noop", "label": "noop"},
                        {"id": "right", "label": "right", "hold_ticks": 6},
                        {"id": "jump", "label": "jump", "hold_ticks": 4},
                    ]
                },
                "context": {"mode": "tick", "step": 4, "tick": 4},
            },
            "arenaTrace": None,
            "result": None,
        },
    )
    service = ArenaVisualGatewayQueryService(visualization_spec=RETRO_VISUALIZATION_SPEC)

    scene = service.load_scene(manifest_path, seq=2)

    assert scene is not None
    assert scene.body["frame"]["viewport"] == {"width": 256, "height": 240}
    assert scene.media is not None
    assert scene.media.primary is not None
    assert scene.media.primary.url == inline_frame_data_url
    assert base64.b64decode(scene.media.primary.url.split(",", 1)[1])[:6] == b"ABCABC"
    assert any(action["id"] == "right" and action["hold_ticks"] == 6 for action in scene.legal_actions)


def test_gateway_service_pages_timeline_with_stable_after_seq_semantics(tmp_path: Path) -> None:
    manifest_path = _persist_visual_session(tmp_path)
    service = ArenaVisualGatewayQueryService(visualization_spec=_build_visualization_spec())

    first_page = service.page_timeline(manifest_path, limit=2)
    second_page = service.page_timeline(manifest_path, after_seq=2, limit=2)
    tail_page = service.page_timeline(manifest_path, after_seq=4, limit=10)

    assert [event.seq for event in first_page.events] == [1, 2]
    assert first_page.next_after_seq == 2
    assert first_page.has_more is True

    assert [event.seq for event in second_page.events] == [3, 4]
    assert second_page.next_after_seq == 4
    assert second_page.has_more is True

    assert [event.seq for event in tail_page.events] == [5, 6]
    assert tail_page.next_after_seq == 6
    assert tail_page.has_more is False


def test_gateway_service_loads_scene_by_seq_using_snapshot_anchor_when_available(tmp_path: Path) -> None:
    manifest_path = _persist_visual_session(tmp_path)
    service = ArenaVisualGatewayQueryService(visualization_spec=_build_visualization_spec())

    scene = service.load_scene(manifest_path, seq=6)

    assert scene is not None
    assert scene.kind == "frame"
    assert scene.phase == "replay"
    assert scene.seq == 6
    assert scene.active_player_id == "player_1"
    assert scene.summary["snapshotSeq"] == 5
    assert scene.summary["eventType"] == "result"
    assert scene.body["snapshot"]["board"]["state"] == "anchored"
    assert scene.body["snapshot"]["frameRef"] == "frame-5"
    assert scene.media is not None
    assert scene.media.primary is not None
    assert scene.media.primary.media_id == "frame-5"
    assert [ref.media_id for ref in scene.media.auxiliary] == ["frame-5-mini"]


def test_gateway_service_loads_scene_by_seq_using_seek_snapshot_index(tmp_path: Path) -> None:
    manifest_path = _persist_visual_session_with_dedicated_seek_snapshots(tmp_path)
    service = ArenaVisualGatewayQueryService(visualization_spec=GOMOKU_VISUALIZATION_SPEC)

    scene = service.load_scene(manifest_path, seq=8)

    assert scene is not None
    assert scene.phase == "replay"
    assert scene.seq == 8
    assert scene.summary["snapshotSeq"] == 8
    assert scene.summary["snapshotLabel"] == "legacy-anchor-8"
    assert scene.summary["eventType"] == "snapshot"


def test_gateway_service_falls_back_to_legacy_snapshot_anchor_when_seek_snapshot_file_is_missing(
    tmp_path: Path,
) -> None:
    manifest_path = _persist_visual_session_with_dedicated_seek_snapshots(tmp_path)
    (manifest_path.parent / "seek_snapshots.json").unlink()
    service = ArenaVisualGatewayQueryService(visualization_spec=GOMOKU_VISUALIZATION_SPEC)

    scene = service.load_scene(manifest_path, seq=8)

    assert scene is not None
    assert scene.summary["snapshotSeq"] == 8
    assert scene.summary["snapshotLabel"] == "legacy-anchor-8"


def test_gateway_service_falls_back_to_legacy_snapshot_anchor_when_explicit_ref_snapshot_seq_is_missing_from_seek_index(
    tmp_path: Path,
) -> None:
    manifest_path = _persist_visual_session_with_dedicated_seek_snapshots(
        tmp_path,
        seek_snapshot_seqs=(5,),
        event_ref_snapshot_seq=8,
    )
    service = ArenaVisualGatewayQueryService(visualization_spec=GOMOKU_VISUALIZATION_SPEC)

    scene = service.load_scene(manifest_path, seq=8)

    assert scene is not None
    assert scene.summary["snapshotSeq"] == 8
    assert scene.summary["snapshotLabel"] == "legacy-anchor-8"


def test_gateway_service_looks_up_marker_sequences_from_index(tmp_path: Path) -> None:
    manifest_path = _persist_visual_session(tmp_path)
    service = ArenaVisualGatewayQueryService(visualization_spec=_build_visualization_spec())

    assert service.lookup_marker(manifest_path, "snapshot") == (5,)
    assert service.lookup_marker(manifest_path, "result") == (6,)
    assert service.lookup_marker(manifest_path, "missing") == ()


def test_gateway_service_looks_up_media_refs_from_assembled_scenes(tmp_path: Path) -> None:
    manifest_path = _persist_visual_session(tmp_path)
    service = ArenaVisualGatewayQueryService(visualization_spec=_build_visualization_spec())

    media_ref = service.lookup_media(manifest_path, "frame-5")

    assert media_ref is not None
    assert media_ref.media_id == "frame-5"
    assert media_ref.transport == "artifact_ref"
    assert media_ref.mime_type == "image/png"
    assert media_ref.url == "frames/frame-5.png"
    assert media_ref.preview_ref == "thumb-5"


def test_gateway_service_lookup_media_does_not_depend_on_timeline_rescans(tmp_path: Path) -> None:
    class _NoFindEventService(ArenaVisualGatewayQueryService):
        def _find_event(self, events, *, seq):  # type: ignore[override]
            raise AssertionError("lookup_media should not need timeline rescans")

    manifest_path = _persist_visual_session(tmp_path)
    service = _NoFindEventService(visualization_spec=_build_visualization_spec())

    media_ref = service.lookup_media(manifest_path, "frame-5")

    assert media_ref is not None
    assert media_ref.media_id == "frame-5"


def test_gateway_service_fails_loudly_when_manifest_is_missing_index_ref(tmp_path: Path) -> None:
    manifest_path = _persist_visual_session(tmp_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    del manifest["artifacts"]["index_ref"]
    del manifest["timeline"]["indexRef"]
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    service = ArenaVisualGatewayQueryService(visualization_spec=_build_visualization_spec())

    with pytest.raises(ValueError, match="indexRef"):
        service.load_session(manifest_path)


def test_gateway_service_fails_loudly_when_manifest_is_missing_timeline_ref(tmp_path: Path) -> None:
    manifest_path = _persist_visual_session(tmp_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    del manifest["artifacts"]["timeline_ref"]
    del manifest["timeline"]["timelineRef"]
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    service = ArenaVisualGatewayQueryService(visualization_spec=_build_visualization_spec())

    with pytest.raises(ValueError, match="timelineRef"):
        service.page_timeline(manifest_path)


def test_gateway_service_fails_loudly_when_snapshot_anchor_is_missing_snapshot_ref(tmp_path: Path) -> None:
    manifest_path = _persist_visual_session(tmp_path)
    index_path = manifest_path.parent / "index.json"
    index_payload = json.loads(index_path.read_text(encoding="utf-8"))
    del index_payload["snapshotAnchors"][0]["snapshotRef"]
    index_path.write_text(json.dumps(index_payload, ensure_ascii=False, indent=2), encoding="utf-8")
    service = ArenaVisualGatewayQueryService(visualization_spec=_build_visualization_spec())

    with pytest.raises(ValueError, match="snapshotRef"):
        service.load_scene(manifest_path, seq=6)
