from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import pytest

from gage_eval.game_kits.phase_card_game.doudizhu.visualization import (
    VISUALIZATION_SPEC as DOUDIZHU_VISUALIZATION_SPEC,
)
from gage_eval.role.arena.core.game_session import GameSession
from gage_eval.role.arena.core.invocation import GameArenaInvocationContext
from gage_eval.role.arena.core.types import ArenaSample
from gage_eval.role.arena.core.players import PlayerBindingSpec
from gage_eval.role.arena.output.writer import ArenaOutputWriter
from gage_eval.role.arena.human_input_protocol import (
    SampleActionRouter,
)
from gage_eval.role.arena.runtime_services import ArenaRuntimeServiceHub
from gage_eval.role.arena.types import ArenaAction, GameResult
from gage_eval.role.arena.visualization.contracts import ActionIntentReceipt, ObserverRef, VisualSession
from gage_eval.role.arena.visualization.gateway_service import ArenaVisualGatewayQueryService
from gage_eval.role.arena.visualization.http_server import ArenaVisualHTTPServer
from gage_eval.role.arena.visualization.recorder import ArenaVisualSessionRecorder
from gage_eval.role.adapters.arena import ArenaRoleAdapter
from gage_eval.role.arena.player_drivers.registry import PlayerDriverRegistry
from gage_eval.tools.action_server import ActionQueueServer
from gage_eval.role.arena.replay_schema_writer import update_replay_manifest_visual_session_ref


class _DummyPlayer:
    player_id = "player_0"
    display_name = "Player 0"
    seat = "seat-0"
    player_kind = "dummy"
    metadata: dict[str, object] = {}

    def next_action(self, observation) -> ArenaAction:
        return ArenaAction(player=self.player_id, move="A1", raw="A1")


class _DummyEnv:
    def __init__(self, replay_path: Path) -> None:
        self._replay_path = str(replay_path)
        self._active_player = "player_0"
        self._terminal = False
        self.apply_calls: list[str] = []

    def get_active_player(self) -> str:
        return self._active_player

    def observe(self, player_id: str) -> dict[str, object]:
        return {
            "board_text": "demo-board",
            "active_player_id": player_id,
            "legalActions": [
                {"id": "A1", "label": "Action A1"},
                {"id": "PASS", "label": "Pass"},
            ],
            "media": {
                "primary": {
                    "mediaId": "frame-1",
                    "transport": "artifact_ref",
                    "mimeType": "image/png",
                    "url": "frames/frame-1.png",
                    "previewRef": "thumb-1",
                }
            },
        }

    def apply(self, action: ArenaAction):
        self.apply_calls.append(action.move)
        self._terminal = True
        return None

    def is_terminal(self) -> bool:
        return self._terminal

    def build_result(self, *, result: str, reason: str | None):
        return GameResult(
            winner=self._active_player,
            result=result,
            reason=reason,
            move_count=1,
            illegal_move_count=0,
            final_board="demo-board",
            move_log=[{"player": self._active_player, "move": "A1"}],
            replay_path=self._replay_path,
        )


class _DummyEnvWithoutReplayPath:
    def __init__(self) -> None:
        self._active_player = "player_0"
        self._terminal = False
        self.apply_calls: list[str] = []

    def get_active_player(self) -> str:
        return self._active_player

    def observe(self, player_id: str) -> dict[str, object]:
        return {
            "board_text": "demo-board",
            "active_player_id": player_id,
            "legalActions": [
                {"id": "A1", "label": "Action A1"},
                {"id": "PASS", "label": "Pass"},
            ],
        }

    def apply(self, action: ArenaAction):
        self.apply_calls.append(action.move)
        self._terminal = True
        return None

    def is_terminal(self) -> bool:
        return self._terminal

    def build_result(self, *, result: str, reason: str | None):
        return GameResult(
            winner=self._active_player,
            result=result,
            reason=reason,
            move_count=1,
            illegal_move_count=0,
            final_board="demo-board",
            move_log=[{"player": self._active_player, "move": "A1"}],
        )


class _DummyScheduler:
    family = "turn"
    defaults = {"max_steps": 1}


class _DummyVisualizationSpec:
    def __init__(
        self,
        plugin_id: str,
        game_id: str | None = None,
        *,
        supported_modes: tuple[str, ...] = (),
    ) -> None:
        self.plugin_id = plugin_id
        self.game_id = game_id
        self.observer_schema = {"supported_modes": list(supported_modes)}


class _DummyResolved:
    def __init__(self, env_factory, *, visualization_spec) -> None:
        self.observation_workflow = None
        self.support_workflow = None
        self.scheduler = _DummyScheduler()
        self.env_spec = type("EnvSpec", (), {"defaults": {"env_factory": env_factory}, "env_id": "dummy"})()
        self.visualization_spec = visualization_spec
        self.player_bindings = ()
        self.player_driver_registry = None
        self.game_kit = type("GameKit", (), {"kit_id": "gomoku"})()


class _HumanEnv:
    def __init__(self) -> None:
        self._active_player = "Human"
        self._terminal = False
        self.applied_actions: list[str] = []

    def get_active_player(self) -> str:
        return self._active_player

    def observe(self, player_id: str) -> dict[str, object]:
        return {
            "player_id": player_id,
            "legal_moves": ["A1", "PASS"],
        }

    def apply(self, action: ArenaAction):
        self.applied_actions.append(action.move)
        self._terminal = True
        return None

    def is_terminal(self) -> bool:
        return self._terminal

    def build_result(self, *, result: str, reason: str | None):
        return GameResult(
            winner=self._active_player,
            result=result,
            reason=reason,
            move_count=len(self.applied_actions),
            illegal_move_count=0,
            final_board="human-board",
            move_log=[{"player": self._active_player, "move": move} for move in self.applied_actions],
        )


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


def _read_http_error(target: str | Request) -> tuple[int, dict[str, object]]:
    try:
        with urlopen(target) as _:
            raise AssertionError("Expected HTTPError")
    except HTTPError as exc:
        payload = json.loads(exc.read().decode("utf-8"))
        assert isinstance(payload, dict)
        return exc.code, payload


def _materialize_visual_http_session(tmp_path: Path, *, run_id: str, sample_id: str) -> None:
    replay_path = tmp_path / "runs" / run_id / "replays" / sample_id / "replay.json"
    recorder = ArenaVisualSessionRecorder(
        plugin_id="arena.visualization.pettingzoo.frame_v1",
        game_id="gomoku",
        scheduling_family="turn",
        session_id=sample_id,
        observer_modes=("player", "global"),
    )
    session = GameSession(
        sample=ArenaSample(game_kit="gomoku", env="gomoku-standard", scheduler="turn/default"),
        environment=_DummyEnv(replay_path),
        player_specs=(_DummyPlayer(),),
        visual_recorder=recorder,
    )

    replay_path.parent.mkdir(parents=True, exist_ok=True)
    replay_path.write_text(
        json.dumps({"schema": "gage_replay/v1", "artifacts": {}}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    observation = session.observe()
    action = session.decide_current_player(observation)
    session.apply(action)
    session.advance()
    session.finalize()
    ArenaOutputWriter().finalize(session)


def _assert_capabilities_include(
    capabilities: dict[str, object],
    *,
    observer_modes: list[str],
) -> None:
    assert capabilities["supportsReplay"] is True
    assert capabilities["supportsTimeline"] is True
    assert capabilities["supportsSeek"] is True
    assert capabilities["observerModes"] == observer_modes


def _materialize_table_visual_http_session(tmp_path: Path, *, run_id: str, sample_id: str) -> None:
    session_dir = tmp_path / "runs" / run_id / "replays" / sample_id / "arena_visual_session" / "v1"
    session_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = session_dir / "manifest.json"
    index_path = session_dir / "index.json"
    timeline_path = session_dir / "timeline.jsonl"
    snapshot_dir = session_dir / "snapshots"
    snapshot_dir.mkdir(exist_ok=True)
    snapshot_path = snapshot_dir / "seq-7.json"

    manifest_payload = {
        "visualSession": VisualSession(
            session_id=sample_id,
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
        ]
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
                "played_cards": [],
                "seen_cards": ["3"],
            },
            "private_state": {
                "self_id": "player_0",
                "current_hand": ["3", "4"],
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
        }
    }

    manifest_path.write_text(json.dumps(manifest_payload, ensure_ascii=False, indent=2), encoding="utf-8")
    index_path.write_text(json.dumps(index_payload, ensure_ascii=False, indent=2), encoding="utf-8")
    timeline_path.write_text(json.dumps(event_payload, ensure_ascii=False) + "\n", encoding="utf-8")
    snapshot_path.write_text(json.dumps(snapshot_payload, ensure_ascii=False, indent=2), encoding="utf-8")


def test_arena_visual_gateway_records_and_persists_sidecar_artifacts(tmp_path: Path) -> None:
    replay_path = tmp_path / "runs" / "run-1" / "replays" / "sample-1" / "replay.json"
    recorder = ArenaVisualSessionRecorder(
        plugin_id="arena-role",
        game_id="gomoku",
        scheduling_family="turn",
        session_id="sample-1",
    )
    session = GameSession(
        sample=ArenaSample(game_kit="gomoku", env="gomoku-standard", scheduler="turn/default"),
        environment=_DummyEnv(replay_path),
        player_specs=(_DummyPlayer(),),
        visual_recorder=recorder,
    )

    replay_path.parent.mkdir(parents=True, exist_ok=True)
    replay_path.write_text(
        json.dumps({"schema": "gage_replay/v1", "artifacts": {}}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    observation = session.observe()
    action = session.decide_current_player(observation)
    session.apply(action)
    session.advance()
    session.finalize()

    output = ArenaOutputWriter().finalize(session)
    serialized = ArenaRoleAdapter._serialize_gamearena_value(output)
    artifacts = serialized["artifacts"]

    assert artifacts["replay_ref"].endswith("replay.json")
    assert artifacts["visual_session_ref"].endswith("arena_visual_session/v1/manifest.json")

    manifest_path = Path(artifacts["visual_session_ref"])
    timeline_path = manifest_path.parent / "timeline.jsonl"
    snapshot_dir = manifest_path.parent / "snapshots"
    index_path = manifest_path.parent / "index.json"
    seek_snapshots_path = manifest_path.parent / "seek_snapshots.json"

    assert manifest_path.exists()
    assert timeline_path.exists()
    assert index_path.exists()
    assert seek_snapshots_path.exists()
    assert any(snapshot_dir.iterdir())

    events = [json.loads(line) for line in timeline_path.read_text(encoding="utf-8").splitlines()]
    assert [event["type"] for event in events] == [
        "decision_window_open",
        "action_intent",
        "action_committed",
        "decision_window_close",
        "snapshot",
        "result",
    ]
    assert [int(event["seq"]) for event in events] == [1, 2, 3, 4, 5, 6]
    seek_snapshots = json.loads(seek_snapshots_path.read_text(encoding="utf-8"))
    assert seek_snapshots["seekSnapshots"][0]["snapshotMode"] == "full"
    replay_manifest = json.loads(Path(artifacts["replay_ref"]).read_text(encoding="utf-8"))
    assert replay_manifest["artifacts"]["visual_session_ref"].endswith(
        "arena_visual_session/v1/manifest.json"
    )


def test_arena_visual_gateway_materializes_replay_manifest_when_result_replay_path_is_missing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("GAGE_EVAL_SAVE_DIR", str(tmp_path / "runs"))
    sample_id = "sample-inferred"
    run_id = "run-inferred"
    recorder = ArenaVisualSessionRecorder(
        plugin_id="arena.visualization.tictactoe.board_v1",
        game_id="tictactoe",
        scheduling_family="turn",
        session_id=sample_id,
    )
    session = GameSession(
        sample=ArenaSample(game_kit="tictactoe", env="tictactoe_standard", scheduler="turn/default"),
        environment=_DummyEnvWithoutReplayPath(),
        player_specs=(_DummyPlayer(),),
        invocation_context=GameArenaInvocationContext(
            adapter_id="tictactoe_visual_gateway",
            sample_id=sample_id,
            trace=SimpleNamespace(run_id=run_id),
            sample_payload={"id": sample_id},
        ),
        visual_recorder=recorder,
    )

    observation = session.observe()
    action = session.decide_current_player(observation)
    session.apply(action)
    session.advance()
    session.finalize()

    output = ArenaOutputWriter().finalize(session)
    serialized = ArenaRoleAdapter._serialize_gamearena_value(output)
    artifacts = serialized["artifacts"]
    replay_path = Path(artifacts["replay_ref"])

    assert replay_path.exists()
    assert replay_path.name == "replay.json"
    assert replay_path.parent.parent.name == "replays"
    assert serialized["result"]["replay_path"] == str(replay_path)

    replay_manifest = json.loads(replay_path.read_text(encoding="utf-8"))
    assert replay_manifest["schema"] == "gage_replay/v1"
    assert replay_manifest["artifacts"]["visual_session_ref"].endswith(
        "arena_visual_session/v1/manifest.json"
    )


def test_arena_visual_gateway_prefers_visualization_spec_identity_for_recorder(tmp_path: Path) -> None:
    replay_path = tmp_path / "runs" / "run-2" / "replays" / "sample-2" / "replay.json"

    def _env_factory(**kwargs):
        del kwargs
        return _DummyEnv(replay_path)

    resolved = _DummyResolved(
        _env_factory,
        visualization_spec=_DummyVisualizationSpec(
            plugin_id="viz-plugin",
            game_id="ignored",
            supported_modes=("player", "camera"),
        ),
    )
    session = GameSession.from_resolved(
        ArenaSample(game_kit="gomoku", env="gomoku-standard", scheduler="turn/default"),
        resolved,
        resources=None,
    )

    assert session.visual_recorder is not None
    assert session.visual_recorder.plugin_id == "viz-plugin"
    assert session.visual_recorder.game_id == "gomoku"
    assert session.visual_recorder.observer_modes == ("player", "camera")
    assert session.visual_recorder.build_visual_session().capabilities["observerModes"] == [
        "player",
        "camera",
    ]


def test_arena_visual_gateway_persists_observer_modes_from_resolved_runtime_into_query_and_http_read_paths(
    tmp_path: Path,
) -> None:
    replay_path = tmp_path / "runs" / "run-wired" / "replays" / "sample-wired" / "replay.json"

    def _env_factory(**kwargs):
        del kwargs
        return _DummyEnv(replay_path)

    resolved = _DummyResolved(
        _env_factory,
        visualization_spec=_DummyVisualizationSpec(
            plugin_id="arena.visualization.vizdoom.frame_v1",
            game_id="ignored",
            supported_modes=("player", "camera"),
        ),
    )
    session = GameSession.from_resolved(
        ArenaSample(game_kit="gomoku", env="gomoku-standard", scheduler="turn/default"),
        resolved,
        resources=None,
        invocation_context=GameArenaInvocationContext(
            adapter_id="arena",
            sample_id="sample-wired",
        ),
    )
    session.player_specs = (_DummyPlayer(),)

    replay_path.parent.mkdir(parents=True, exist_ok=True)
    replay_path.write_text(
        json.dumps({"schema": "gage_replay/v1", "artifacts": {}}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    observation = session.observe()
    action = session.decide_current_player(observation)
    session.apply(action)
    session.advance()
    session.finalize()

    output = ArenaOutputWriter().finalize(session)
    serialized = ArenaRoleAdapter._serialize_gamearena_value(output)
    manifest_path = Path(serialized["artifacts"]["visual_session_ref"])
    seek_snapshots_path = manifest_path.parent / "seek_snapshots.json"

    query_service = ArenaVisualGatewayQueryService()
    visual_session = query_service.load_session(manifest_path)

    _assert_capabilities_include(
        visual_session.capabilities,
        observer_modes=["player", "camera"],
    )
    seek_snapshot_payload = json.loads(seek_snapshots_path.read_text(encoding="utf-8"))
    assert seek_snapshot_payload["seekSnapshots"][0]["snapshotMode"] == "media_ref"

    server = ArenaVisualHTTPServer(
        host="127.0.0.1",
        port=0,
        base_dir=tmp_path,
    )
    server.start()
    try:
        host, port = server.server_address
        session_payload = _get_json(
            f"http://{host}:{port}/arena_visual/sessions/sample-wired"
        )

        _assert_capabilities_include(
            session_payload["capabilities"],
            observer_modes=["player", "camera"],
        )
    finally:
        server.stop()


def test_update_replay_manifest_visual_session_ref_is_non_fatal_when_missing(tmp_path: Path) -> None:
    replay_path = tmp_path / "missing" / "replay.json"
    assert update_replay_manifest_visual_session_ref(
        replay_path=replay_path,
        visual_session_ref="arena_visual_session/v1/manifest.json",
    ) is False


def test_arena_visual_gateway_http_server_reads_generated_artifacts(tmp_path: Path) -> None:
    _materialize_visual_http_session(tmp_path, run_id="run-http", sample_id="sample-http")

    submitted_actions: list[tuple[str, str | None, dict[str, object]]] = []
    submitted_chat: list[tuple[str, str | None, dict[str, object]]] = []
    submitted_control: list[tuple[str, str | None, dict[str, object]]] = []

    def _action_submitter(session_id: str, run_id: str | None, payload: dict[str, object]) -> ActionIntentReceipt:
        submitted_actions.append((session_id, run_id, payload))
        return ActionIntentReceipt(
            intent_id="intent-http-1",
            state="accepted",
            related_event_seq=6,
            reason="queued",
        )

    def _chat_submitter(session_id: str, run_id: str | None, payload: dict[str, object]) -> ActionIntentReceipt:
        submitted_chat.append((session_id, run_id, payload))
        return ActionIntentReceipt(
            intent_id="chat-http-1",
            state="accepted",
            reason="queued",
        )

    def _control_submitter(session_id: str, run_id: str | None, payload: dict[str, object]) -> ActionIntentReceipt:
        submitted_control.append((session_id, run_id, payload))
        return ActionIntentReceipt(
            intent_id="control-http-1",
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
        base_url = f"http://{host}:{port}/arena_visual/sessions/sample-http"

        session_payload = _get_json(base_url)
        timeline_payload = _get_json(f"{base_url}/timeline?limit=3")
        scene_payload = _get_json(f"{base_url}/scene?seq=6")
        media_payload = _get_json(f"{base_url}/media/frame-1")
        receipt_payload = _post_json(
            f"{base_url}/actions",
            {"playerId": "player_0", "action": {"move": "A1"}},
        )
        chat_receipt_payload = _post_json(
            f"{base_url}/chat",
            {"playerId": "player_0", "text": "hello"},
        )
        control_receipt_payload = _post_json(
            f"{base_url}/control",
            {"commandType": "pause"},
        )

        assert session_payload["sessionId"] == "sample-http"
        assert session_payload["timeline"]["eventCount"] == 6
        assert session_payload["playback"]["canSeek"] is True
        _assert_capabilities_include(
            session_payload["capabilities"],
            observer_modes=["player", "global"],
        )

        assert timeline_payload["sessionId"] == "sample-http"
        assert timeline_payload["limit"] == 3
        assert timeline_payload["hasMore"] is True
        assert [event["type"] for event in timeline_payload["events"]] == [
            "decision_window_open",
            "action_intent",
            "action_committed",
        ]

        assert scene_payload["sceneId"] == "sample-http:seq:6"
        assert scene_payload["summary"]["snapshotSeq"] == 5
        assert scene_payload["media"]["primary"]["mediaId"] == "frame-1"

        assert media_payload == {
            "mediaId": "frame-1",
            "transport": "artifact_ref",
            "mimeType": "image/png",
            "url": "frames/frame-1.png",
            "previewRef": "thumb-1",
        }

        assert receipt_payload == {
            "intentId": "intent-http-1",
            "state": "accepted",
            "relatedEventSeq": 6,
            "reason": "queued",
        }
        assert chat_receipt_payload == {
            "intentId": "chat-http-1",
            "state": "accepted",
            "reason": "queued",
        }
        assert control_receipt_payload == {
            "intentId": "control-http-1",
            "state": "accepted",
            "reason": "queued",
        }
        assert submitted_actions == [
            (
                "sample-http",
                None,
                {"playerId": "player_0", "action": {"move": "A1"}},
            )
        ]
        assert submitted_chat == [
            (
                "sample-http",
                None,
                {"playerId": "player_0", "text": "hello"},
            )
        ]
        assert submitted_control == [
            (
                "sample-http",
                None,
                {"commandType": "pause"},
            )
        ]
    finally:
        server.stop()


def test_arena_visual_gateway_http_server_accepts_observer_override_on_session_and_scene_reads(
    tmp_path: Path,
) -> None:
    _materialize_table_visual_http_session(tmp_path, run_id="run-http", sample_id="sample-http")

    server = ArenaVisualHTTPServer(
        host="127.0.0.1",
        port=0,
        base_dir=tmp_path,
    )
    server.start()
    try:
        host, port = server.server_address
        base_url = f"http://{host}:{port}/arena_visual/sessions/sample-http"

        session_payload = _get_json(f"{base_url}?observer_kind=player&observer_id=player_0")
        scene_payload = _get_json(f"{base_url}/scene?seq=7&observer_kind=player&observer_id=player_0")
        code, error_payload = _read_http_error(f"{base_url}/scene?seq=7&observer_kind=wizard")

        assert session_payload["observer"] == {
            "observerId": "player_0",
            "observerKind": "player",
        }
        assert scene_payload["body"]["status"]["observerPlayerId"] == "player_0"
        seats = {seat["playerId"]: seat for seat in scene_payload["body"]["table"]["seats"]}
        assert seats["player_0"]["hand"] == {
            "isVisible": True,
            "cards": ["3", "4"],
            "maskedCount": 0,
        }
        assert code == 400
        assert error_payload == {"error": "invalid_observer_kind"}
    finally:
        server.stop()


def test_arena_visual_gateway_routes_actions_into_runtime_queue_and_driver_consumes_them(
    tmp_path: Path,
    monkeypatch,
) -> None:
    runtime_hub = ArenaRuntimeServiceHub(adapter_id="arena")

    def _env_factory(*, sample, resolved, resources, player_specs, invocation_context=None):
        _ = sample, resolved, resources, player_specs, invocation_context
        return _HumanEnv()

    resolved = SimpleNamespace(
        game_kit=SimpleNamespace(kit_id="gomoku", seat_spec={}, defaults={}),
        env_spec=SimpleNamespace(
            env_id="human-env",
            defaults={"env_factory": _env_factory},
        ),
        scheduler=_DummyScheduler(),
        resource_spec={},
        player_bindings=(
            PlayerBindingSpec(
                seat="human-seat",
                player_id="Human",
                player_kind="human",
                driver_id="player_driver/human_local_input",
            ),
        ),
        player_driver_registry=PlayerDriverRegistry(),
        observation_workflow=None,
        support_workflow=None,
        visualization_spec=_DummyVisualizationSpec(plugin_id="arena.visualization.gomoku.board_v1"),
    )
    session = GameSession.from_resolved(
        ArenaSample(game_kit="gomoku", env="human-env", scheduler="turn/default"),
        resolved,
        resources={},
        invocation_context=GameArenaInvocationContext(
            adapter_id="arena",
            sample_id="sample-human-1",
            human_input_config={"enabled": True, "host": "127.0.0.1", "port": 0},
            runtime_service_hub=runtime_hub,
        ),
    )

    server = ArenaVisualHTTPServer(
        host="127.0.0.1",
        port=0,
        base_dir=tmp_path,
        action_submitter=runtime_hub.submit_action_intent,
        chat_submitter=runtime_hub.submit_chat_message,
        control_submitter=runtime_hub.submit_control_command,
    )
    server.start()
    monkeypatch.setattr(
        "builtins.input",
        lambda *args, **kwargs: pytest.fail("interactive fallback should not be used"),
    )
    try:
        host, port = server.server_address
        receipt_payload = _post_json(
            f"http://{host}:{port}/arena_visual/sessions/sample-human-1/actions",
            {
                "playerId": "Human",
                "action": {
                    "move": "A1",
                    "hold_ticks": 3,
                    "metadata": {"source": "frontend"},
                },
            },
        )
        chat_receipt_payload = _post_json(
            f"http://{host}:{port}/arena_visual/sessions/sample-human-1/chat",
            {
                "playerId": "Human",
                "text": "hello from human",
            },
        )
        control_receipt_payload = _post_json(
            f"http://{host}:{port}/arena_visual/sessions/sample-human-1/control",
            {
                "commandType": "pause",
            },
        )

        assert receipt_payload == {
            "intentId": "sample-human-1:intent-1",
            "state": "accepted",
            "reason": "queued",
        }
        assert chat_receipt_payload == {
            "intentId": "sample-human-1:intent-2",
            "state": "accepted",
            "reason": "queued",
        }
        assert control_receipt_payload == {
            "intentId": "sample-human-1:intent-3",
            "state": "rejected",
            "reason": "control_queue_not_available",
        }
        action_server = runtime_hub.peek_action_server()
        assert action_server is not None
        assert action_server.has_action_routes() is True
        action_router, error = action_server.resolve_action_queue("sample-human-1")
        assert error is None
        assert isinstance(action_router, SampleActionRouter)
        queued_payload = json.loads(action_router.queue_for("Human").get(timeout=1.0))
        assert queued_payload["sample_id"] == "sample-human-1"
        assert queued_payload["player_id"] == "Human"
        assert queued_payload["action"] == "A1"
        assert queued_payload["move"] == "A1"
        assert queued_payload["source"] == "arena_visual_gateway"
        assert queued_payload["metadata"] == {
            "source": "frontend",
            "hold_ticks": 3,
        }
        assert json.loads(queued_payload["raw"]) == {
            "action": "A1",
            "source": "frontend",
            "hold_ticks": 3,
        }
        queued_chat_payload = action_server.chat_queue.get(timeout=1.0)
        assert queued_chat_payload == {
            "player_id": "Human",
            "text": "hello from human",
            "channel": "table",
        }
        action_router.queue_for("Human").put(json.dumps(queued_payload, ensure_ascii=False))

        observation = session.observe()
        action = session.decide_current_player(observation)
        assert action.move == "A1"
        assert json.loads(action.raw) == {
            "action": "A1",
            "source": "frontend",
            "hold_ticks": 3,
        }
        assert action.metadata["hold_ticks"] == 3
        assert action.metadata["source"] == "frontend"
        session.apply(action)
        session.finalize()

        assert action_server.has_action_routes() is False
        rejected_receipt = runtime_hub.submit_action_intent(
            "sample-human-1",
            None,
            {"playerId": "Human", "action": {"move": "A1"}},
        )
        assert rejected_receipt.state == "rejected"
        assert rejected_receipt.reason == "sample_route_not_found"
    finally:
        server.stop()
        action_server = runtime_hub.peek_action_server()
        if action_server is not None:
            action_server.stop()


def test_arena_visual_gateway_http_server_disambiguates_duplicate_sample_ids_by_run_id(tmp_path: Path) -> None:
    _materialize_visual_http_session(tmp_path, run_id="run-a", sample_id="shared-sample")
    _materialize_visual_http_session(tmp_path, run_id="run-b", sample_id="shared-sample")

    submitted_actions: list[tuple[str, str | None, dict[str, object]]] = []
    submitted_chat: list[tuple[str, str | None, dict[str, object]]] = []
    submitted_control: list[tuple[str, str | None, dict[str, object]]] = []

    def _action_submitter(session_id: str, run_id: str | None, payload: dict[str, object]) -> ActionIntentReceipt:
        submitted_actions.append((session_id, run_id, payload))
        return ActionIntentReceipt(intent_id="intent-run-b", state="accepted")

    def _chat_submitter(session_id: str, run_id: str | None, payload: dict[str, object]) -> ActionIntentReceipt:
        submitted_chat.append((session_id, run_id, payload))
        return ActionIntentReceipt(intent_id="chat-run-b", state="accepted")

    def _control_submitter(session_id: str, run_id: str | None, payload: dict[str, object]) -> ActionIntentReceipt:
        submitted_control.append((session_id, run_id, payload))
        return ActionIntentReceipt(intent_id="control-run-b", state="accepted")

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
        code, payload = _read_http_error(
            f"http://{host}:{port}/arena_visual/sessions/shared-sample"
        )
        session_payload = _get_json(
            f"http://{host}:{port}/arena_visual/sessions/shared-sample?run_id=run-b"
        )
        receipt_payload = _post_json(
            f"http://{host}:{port}/arena_visual/sessions/shared-sample/actions?run_id=run-b",
            {"playerId": "player_0", "action": {"move": "A1"}},
        )
        chat_receipt_payload = _post_json(
            f"http://{host}:{port}/arena_visual/sessions/shared-sample/chat?run_id=run-b",
            {"playerId": "player_0", "text": "hello"},
        )
        control_receipt_payload = _post_json(
            f"http://{host}:{port}/arena_visual/sessions/shared-sample/control?run_id=run-b",
            {"commandType": "pause"},
        )

        assert code == 409
        assert payload == {
            "error": "session_ambiguous",
            "sessionId": "shared-sample",
        }
        assert session_payload["sessionId"] == "shared-sample"
        assert session_payload["timeline"]["eventCount"] == 6
        assert receipt_payload == {
            "intentId": "intent-run-b",
            "state": "accepted",
        }
        assert chat_receipt_payload == {
            "intentId": "chat-run-b",
            "state": "accepted",
        }
        assert control_receipt_payload == {
            "intentId": "control-run-b",
            "state": "accepted",
        }
        assert submitted_actions == [
            (
                "shared-sample",
                "run-b",
                {"playerId": "player_0", "action": {"move": "A1"}},
            )
        ]
        assert submitted_chat == [
            (
                "shared-sample",
                "run-b",
                {"playerId": "player_0", "text": "hello"},
            )
        ]
        assert submitted_control == [
            (
                "shared-sample",
                "run-b",
                {"commandType": "pause"},
            )
        ]
    finally:
        server.stop()
