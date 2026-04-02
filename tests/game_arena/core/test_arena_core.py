from __future__ import annotations

import gage_eval.role.arena.schedulers.real_time_tick as real_time_tick_module
import threading
import time
from types import SimpleNamespace

import pytest

from gage_eval.role.arena.core.arena_core import GameArenaCore
from gage_eval.role.arena.core.invocation import GameArenaInvocationContext
from gage_eval.role.arena.core.game_session import (
    GameSession,
    _build_action_server,
    _resolve_max_steps,
)
from gage_eval.role.arena.core.types import ArenaSample
from gage_eval.role.arena.human_input_protocol import LatestActionMailbox, SampleActionRouter
from gage_eval.role.arena.schedulers.turn import TurnScheduler
from gage_eval.role.arena.schedulers.real_time_tick import RealTimeTickScheduler
from gage_eval.role.arena.output.writer import ArenaOutputWriter
from gage_eval.role.arena.player_drivers.registry import PlayerDriverRegistry
from gage_eval.role.arena.resources.control import ArenaResourceControl
from gage_eval.role.arena.resources.specs import ArenaResources
from gage_eval.role.arena.resources.visualization import (
    VisualizationPhase,
    VisualizationSession,
)
from gage_eval.role.arena.support.context import SupportContext
from gage_eval.role.arena.support.hooks import SupportHook
from gage_eval.role.arena.support.units.action_shaping import (
    ContinuousActionShapingUnit,
)
from gage_eval.role.arena.support.workflow import GameSupportWorkflow
from gage_eval.role.arena.types import ArenaAction
from gage_eval.role.arena.types import GameResult
from gage_eval.role.arena.core.players import PlayerBindingSpec
from gage_eval.role.arena.visualization.contracts import ControlCommand
from gage_eval.role.arena.visualization.recorder import ArenaVisualSessionRecorder
from gage_eval.game_kits.real_time_game.vizdoom.envs.duel_map01 import (
    DuelMap01Environment,
)
from gage_eval.game_kits.contracts import (
    HumanRealtimeInputProfile,
    RealtimeHumanControlProfile,
    ResolvedRuntimeProfile,
)


def test_game_arena_core_releases_resources_on_scheduler_error(
    fake_resolver,
    fake_resource_control,
    fake_output_writer,
) -> None:
    core = GameArenaCore(
        resolver=fake_resolver(raise_in_scheduler=True),
        resource_control=fake_resource_control,
        output_writer=fake_output_writer,
    )

    with pytest.raises(RuntimeError):
        core.run_sample(ArenaSample(game_kit="gomoku", env=None))

    assert fake_resource_control.release_calls == 1


def test_game_arena_core_releases_resources_on_session_construction_error(
    fake_resolver,
    fake_resource_control,
    fake_output_writer,
    monkeypatch,
) -> None:
    core = GameArenaCore(
        resolver=fake_resolver(),
        resource_control=fake_resource_control,
        output_writer=fake_output_writer,
    )

    def _raise_on_session_build(_cls, sample, resolved, resources):
        raise RuntimeError("session build failed")

    monkeypatch.setattr(
        GameSession,
        "from_resolved",
        classmethod(_raise_on_session_build),
    )

    with pytest.raises(RuntimeError, match="session build failed"):
        core.run_sample(ArenaSample(game_kit="gomoku", env=None))

    assert fake_resource_control.release_calls == 1


def test_arena_output_writer_returns_frozen_snapshot() -> None:
    session = GameSession(sample=ArenaSample(game_kit="gomoku", env=None))
    session.arena_trace.append({"event": "tick", "value": 1})

    output = ArenaOutputWriter().finalize(session)

    assert output.arena_trace[0] == {"event": "tick", "value": 1}
    assert output.arena_trace[0] is not session.arena_trace[0]

    session.arena_trace[0]["value"] = 2
    assert output.arena_trace[0]["value"] == 1

    with pytest.raises(TypeError):
        output.arena_trace[0]["value"] = 3


def test_game_arena_core_returns_output_after_release_failures() -> None:
    class _FailingRuntimeHandle:
        def close(self) -> None:
            raise RuntimeError("close failed")

        def terminate(self) -> None:
            return None

        def reap(self) -> None:
            return None

    class _TerminalEnvironment:
        def is_terminal(self) -> bool:
            return True

        def build_result(self, *, result: str, reason: str | None):
            return GameResult(
                winner="alpha",
                result=result,
                reason=reason,
                move_count=0,
                illegal_move_count=0,
                final_board="board",
                move_log=[],
            )

    class _Resolver:
        def resolve(self, sample):
            del sample
            return SimpleNamespace(
                game_kit=SimpleNamespace(kit_id="gomoku", seat_spec={}, defaults={}),
                env_spec=SimpleNamespace(
                    env_id="gomoku_standard",
                    defaults={
                        "env_factory": (
                            lambda *, sample, resolved, resources, player_specs: _TerminalEnvironment()
                        )
                    },
                ),
                scheduler=SimpleNamespace(run=lambda session: None, defaults={}),
                resource_spec={"env_id": "gomoku_standard"},
                player_bindings=(),
                player_driver_registry=PlayerDriverRegistry(),
                observation_workflow=None,
                support_workflow=None,
                visualization_spec=None,
            )

    class _ResourceControl:
        def allocate(self, resource_spec):
            del resource_spec
            return ArenaResources(game_runtime=_FailingRuntimeHandle())

        def release(self, resources):
            return ArenaResourceControl().release(resources)

    core = GameArenaCore(
        resolver=_Resolver(),
        resource_control=_ResourceControl(),
        output_writer=ArenaOutputWriter(),
    )

    output = core.run_sample(ArenaSample(game_kit="gomoku", env="gomoku_standard"))

    assert output.game_context["resource_lifecycle_phase"] == "release_failed"
    assert output.game_context["resource_errors"][0]["error_code"] == "resource_lifecycle_error"
    assert output.game_context["resource_errors"][0]["operation"] == "close"


def test_game_session_capture_output_tick_is_available_for_record_cadence() -> None:
    session = GameSession(sample=ArenaSample(game_kit="gomoku", env=None))

    session.capture_output_tick()

    assert session.tick == 0
    assert session.step == 0
    assert session.arena_trace == []


def test_game_session_resolves_max_decisions_before_default_max_steps() -> None:
    sample = ArenaSample(
        game_kit="openra",
        env="ra_skirmish_1v1",
        runtime_overrides={"max_decisions": 7200},
    )
    resolved = SimpleNamespace(scheduler=SimpleNamespace(defaults={"max_steps": 256}))

    assert _resolve_max_steps(sample=sample, resolved=resolved) == 7200


def test_game_session_executes_support_hooks_across_main_chain() -> None:
    hook_trace: list[tuple[str, object]] = []

    class _RecordingUnit:
        def __init__(self, label: str, *, rewrite: object | None = None) -> None:
            self.label = label
            self.rewrite = rewrite

        def invoke(self, context: SupportContext) -> SupportContext:
            hook_trace.append((self.label, context.payload.get("action")))
            if self.rewrite is not None:
                context.payload["action"] = self.rewrite
            return context

    class FakeEnvironment:
        def __init__(self) -> None:
            self.applied_actions: list[object] = []
            self._terminal = False

        def get_active_player(self) -> str:
            return "alpha"

        def observe(self, player):
            return {"player": player}

        def apply(self, action):
            self.applied_actions.append(action.move)
            self._terminal = True
            return GameResult(
                winner="alpha",
                result="completed",
                reason="applied",
                move_count=len(self.applied_actions),
                illegal_move_count=0,
                final_board="board",
                move_log=[{"move": action.move}],
            )

        def is_terminal(self) -> bool:
            return self._terminal

        def build_result(self, *, result: str, reason: str | None):
            return GameResult(
                winner="alpha",
                result=result,
                reason=reason,
                move_count=len(self.applied_actions),
                illegal_move_count=0,
                final_board="board",
                move_log=[{"move": move} for move in self.applied_actions],
            )

    class FakePlayer:
        def __init__(self, player_id: str, move: object) -> None:
            self.player_id = player_id
            self._move = move

        def next_action(self, observation) -> ArenaAction:
            del observation
            return ArenaAction(player=self.player_id, move=self._move, raw=self._move)

    workflow = GameSupportWorkflow(
        workflow_id="support-chain",
        units_by_hook={
            SupportHook.AFTER_OBSERVE: [_RecordingUnit("after_observe")],
            SupportHook.BEFORE_DECIDE: [_RecordingUnit("before_decide")],
            SupportHook.AFTER_DECIDE: [_RecordingUnit("after_decide")],
            SupportHook.BEFORE_APPLY: [_RecordingUnit("before_apply")],
            SupportHook.AFTER_APPLY: [_RecordingUnit("after_apply")],
            SupportHook.ON_FINALIZE: [_RecordingUnit("on_finalize")],
        },
    )

    session = GameSession(
        sample=ArenaSample(game_kit="gomoku", env="gomoku_standard"),
        environment=FakeEnvironment(),
        player_specs=(FakePlayer("alpha", [0.5]),),
        support_workflow=workflow,
    )

    observation = session.observe()
    action = session.decide_current_player(observation)
    session.apply(action)
    session.finalize()

    assert hook_trace == [
        ("after_observe", None),
        ("before_decide", None),
        ("after_decide", [0.5]),
        ("before_apply", [0.5]),
        ("after_apply", [0.5]),
        ("on_finalize", None),
    ]


def test_game_session_restart_resets_live_round_state() -> None:
    class FakeEnvironment:
        def __init__(self) -> None:
            self._terminal = True
            self.reset_calls = 0

        def reset(self):
            self.reset_calls += 1
            self._terminal = False
            return {"board_text": "fresh"}

        def is_terminal(self) -> bool:
            return self._terminal

        def get_active_player(self) -> str:
            return "player_0"

        def build_result(self, *, result: str, reason: str | None):
            return GameResult(
                winner="player_0",
                result=result,
                reason=reason,
                move_count=0,
                illegal_move_count=0,
                final_board="board",
                move_log=[],
            )

        def export_frame(self):
            return {
                "active_player_id": "player_0",
                "board_text": "fresh",
                "legal_moves": ["noop"],
                "tick": 0,
                "step": 0,
            }

    environment = FakeEnvironment()
    recorder = ArenaVisualSessionRecorder(
        plugin_id="arena.visualization.retro_platformer.frame_v1",
        game_id="retro_platformer",
        scheduling_family="real_time_tick",
        session_id="sample-live",
    )
    recorder.record_result(
        ts_ms=1011,
        step=12,
        tick=12,
        result=GameResult(
            winner="player_0",
            result="terminated",
            reason="dead",
            move_count=12,
            illegal_move_count=0,
            final_board="dead",
            move_log=[],
        ),
    )
    session = GameSession(
        sample=ArenaSample(game_kit="retro_mario", env="retro_mario"),
        environment=environment,
        player_specs=(
            SimpleNamespace(
                player_id="player_0",
                display_name="player_0",
                seat="seat_0",
                player_kind="human",
                metadata={},
            ),
        ),
        visual_recorder=recorder,
    )
    session.tick = 12
    session.step = 12
    session.final_result = GameResult(
        winner="player_0",
        result="terminated",
        reason="dead",
        move_count=12,
        illegal_move_count=0,
        final_board="dead",
        move_log=[],
    )
    session.arena_trace.append({"step": 12})

    session.request_restart()

    assert session.should_stop() is False
    assert environment.reset_calls == 1
    assert session.final_result is None
    assert session.tick == 0
    assert session.step == 0
    assert session.arena_trace == []
    assert session.visual_recorder is not None
    visual_session = session.visual_recorder.build_visual_session()
    assert visual_session.lifecycle == "live_running"
    assert visual_session.playback.mode == "live_tail"
    assert session.visual_recorder.export_live_state().timeline_events[-1].type == "snapshot"


def test_game_session_marks_pure_human_realtime_visual_capabilities() -> None:
    recorder = ArenaVisualSessionRecorder(
        plugin_id="arena.visualization.retro_platformer.frame_v1",
        game_id="retro_platformer",
        scheduling_family="real_time_tick",
        session_id="sample-live-capabilities",
    )
    session = GameSession(
        sample=ArenaSample(game_kit="retro_platformer", env="retro_mario"),
        environment=SimpleNamespace(reset=lambda: None, is_terminal=lambda: False),
        player_specs=(
            SimpleNamespace(
                player_id="player_0",
                display_name="player_0",
                seat="player_0",
                player_kind="human",
                metadata={},
            ),
        ),
        visual_recorder=recorder,
        runtime_profile=ResolvedRuntimeProfile(
            scheduler_binding="real_time_tick/default",
            scheduler_family="real_time_tick",
            tick_interval_ms=16,
            pure_human_realtime=True,
            scheduler_owns_realtime_clock=True,
            supports_low_latency_realtime_input=True,
            supports_realtime_input_websocket=True,
            human_realtime_inputs=(
                HumanRealtimeInputProfile(
                    player_id="player_0",
                    semantics="continuous_state",
                    tick_interval_ms=16,
                ),
            ),
        ),
    )

    session._visualization_mode = "arena_visual"
    session._visualization_live_scene_scheme = "low_latency_channel"
    session._configure_visual_capabilities()

    visual_session = recorder.build_visual_session()
    assert visual_session.capabilities["supportsLowLatencyRealtimeInput"] is True
    assert visual_session.capabilities["supportsRealtimeInputWebSocket"] is True
    assert visual_session.capabilities["supportsLiveUpdateStream"] is True
    assert visual_session.capabilities["supportsRestart"] is True


def test_game_session_visual_capabilities_require_resolved_runtime_profile_flags() -> None:
    recorder = ArenaVisualSessionRecorder(
        plugin_id="arena.visualization.retro_platformer.frame_v1",
        game_id="retro_platformer",
        scheduling_family="real_time_tick",
        session_id="sample-live-capabilities-negative",
    )
    session = GameSession(
        sample=ArenaSample(game_kit="retro_platformer", env="retro_mario"),
        environment=SimpleNamespace(reset=lambda: None, is_terminal=lambda: False),
        player_specs=(
            SimpleNamespace(
                player_id="player_0",
                display_name="player_0",
                seat="player_0",
                player_kind="human",
                metadata={},
            ),
        ),
        visual_recorder=recorder,
        runtime_profile=ResolvedRuntimeProfile(
            scheduler_binding="real_time_tick/default",
            scheduler_family="real_time_tick",
            tick_interval_ms=16,
            pure_human_realtime=False,
            scheduler_owns_realtime_clock=False,
            supports_low_latency_realtime_input=False,
            supports_realtime_input_websocket=False,
            human_realtime_inputs=(
                HumanRealtimeInputProfile(
                    player_id="player_0",
                    semantics="continuous_state",
                    tick_interval_ms=16,
                ),
            ),
        ),
    )

    session._visualization_mode = "arena_visual"
    session._visualization_live_scene_scheme = "http_pull"
    session._configure_visual_capabilities()

    visual_session = recorder.build_visual_session()
    assert "supportsLowLatencyRealtimeInput" not in visual_session.capabilities
    assert "supportsRealtimeInputWebSocket" not in visual_session.capabilities
    assert "supportsLiveUpdateStream" not in visual_session.capabilities
    assert "supportsRestart" not in visual_session.capabilities


def test_game_session_wires_human_action_router_before_binding_and_cleans_it_up() -> None:
    class _StubActionServer:
        def __init__(self) -> None:
            self.register_calls: list[tuple[str, object]] = []
            self.unregister_calls: list[str] = []

        def register_action_queue(self, sample_id: str, action_router: object) -> None:
            self.register_calls.append((sample_id, action_router))

        def unregister_action_queue(self, sample_id: str) -> None:
            self.unregister_calls.append(sample_id)

    class _StubRuntimeServiceHub:
        def __init__(self) -> None:
            self.action_server = _StubActionServer()
            self.bind_calls: list[tuple[str, object, object]] = []
            self.clear_calls: list[tuple[str, object]] = []

        def ensure_action_server(self, factory):
            _ = factory
            return self.action_server

        def bind_sample_routes(
            self,
            *,
            sample_id: str,
            action_server: object | None = None,
            action_router: object | None = None,
            visualizer: object | None = None,
        ) -> None:
            _ = visualizer
            self.bind_calls.append((sample_id, action_server, action_router))
            if action_server is not None and action_router is not None:
                action_server.register_action_queue(sample_id, action_router)

        def clear_sample_routes(
            self,
            *,
            sample_id: str,
            action_server: object | None = None,
            visualizer: object | None = None,
        ) -> None:
            _ = visualizer
            self.clear_calls.append((sample_id, action_server))
            if action_server is not None:
                action_server.unregister_action_queue(sample_id)

    class FakeEnvironment:
        def __init__(self) -> None:
            self._terminal = False

        def get_active_player(self) -> str:
            return "Human"

        def observe(self, player_id: str) -> dict[str, object]:
            return {"player_id": player_id, "legal_moves": ["A1"]}

        def apply(self, action):
            self._terminal = True
            return GameResult(
                winner=action.player,
                result="completed",
                reason="applied",
                move_count=1,
                illegal_move_count=0,
                final_board="board",
                move_log=[{"move": action.move}],
            )

        def is_terminal(self) -> bool:
            return self._terminal

    runtime_hub = _StubRuntimeServiceHub()
    seen_invocation_contexts: list[GameArenaInvocationContext | None] = []

    def _env_factory(*, sample, resolved, resources, player_specs, invocation_context=None):
        _ = sample, resolved, resources, player_specs
        seen_invocation_contexts.append(invocation_context)
        return FakeEnvironment()

    resolved = SimpleNamespace(
        game_kit=SimpleNamespace(kit_id="gomoku", seat_spec={}, defaults={}),
        env_spec=SimpleNamespace(
            env_id="gomoku_standard",
            defaults={"env_factory": _env_factory},
        ),
        scheduler=TurnScheduler(binding_id="turn/default"),
        resource_spec={},
        player_bindings=(
            PlayerBindingSpec(
                seat="human-seat",
                player_id="Human",
                player_kind="human",
                driver_id="player_driver/human_local_input",
                driver_params={},
            ),
        ),
        player_driver_registry=PlayerDriverRegistry(),
        observation_workflow=None,
        support_workflow=None,
        runtime_profile=ResolvedRuntimeProfile(
            scheduler_binding="real_time_tick/default",
            scheduler_family="real_time_tick",
            tick_interval_ms=16,
            pure_human_realtime=True,
            scheduler_owns_realtime_clock=True,
            supports_low_latency_realtime_input=True,
            supports_realtime_input_websocket=True,
            human_realtime_inputs=(
                HumanRealtimeInputProfile(
                    player_id="Human",
                    semantics="continuous_state",
                    tick_interval_ms=16,
                ),
            ),
        ),
        visualization_spec=SimpleNamespace(
            plugin_id="arena.visualization.gomoku.board_v1",
            observer_schema={"supported_modes": ["player", "global"]},
        ),
    )

    session = GameSession.from_resolved(
        ArenaSample(game_kit="gomoku", env="gomoku-standard"),
        resolved,
        resources={},
        invocation_context=GameArenaInvocationContext(
            adapter_id="arena",
            sample_id="sample-human-1",
            human_input_config={"enabled": True, "host": "127.0.0.1", "port": 0},
            runtime_service_hub=runtime_hub,
        ),
    )

    assert runtime_hub.bind_calls and runtime_hub.action_server.register_calls
    sample_id, action_server, action_router = runtime_hub.bind_calls[0]
    assert sample_id == "sample-human-1"
    assert isinstance(action_router, SampleActionRouter)
    assert isinstance(action_router.queue_for("Human"), LatestActionMailbox)
    assert action_server is runtime_hub.action_server

    assert seen_invocation_contexts[0] is not None
    assert seen_invocation_contexts[0].player_action_queues["Human"] is action_router.queue_for("Human")
    assert session.invocation_context is not None
    assert session.invocation_context.player_action_queues["Human"] is action_router.queue_for("Human")
    assert session.visual_recorder is not None
    assert session.visual_recorder.observer_kind == "player"
    assert session.visual_recorder.observer_id == "Human"

    session.finalize()

    assert runtime_hub.clear_calls == [("sample-human-1", runtime_hub.action_server)]
    assert runtime_hub.action_server.unregister_calls == ["sample-human-1"]


def test_build_action_server_respects_explicit_ephemeral_port() -> None:
    server = _build_action_server(
        GameArenaInvocationContext(
            adapter_id="arena",
            human_input_config={"enabled": True, "host": "127.0.0.1", "port": 0},
        )
    )

    try:
        assert server._port == 0
        assert server._server.server_address[1] > 0
    finally:
        server.stop()


def test_game_session_support_hooks_can_rewrite_action_through_real_runtime_path() -> None:
    action_trace: list[tuple[str, object]] = []

    class _RewriteAndRecordUnit:
        def __init__(self, label: str, *, rewrite: object | None = None) -> None:
            self.label = label
            self.rewrite = rewrite

        def invoke(self, context: SupportContext) -> SupportContext:
            action_trace.append((self.label, context.payload.get("action")))
            if self.rewrite is not None:
                context.payload["action"] = self.rewrite
            return context

    class FakeEnvironment:
        def __init__(self) -> None:
            self.applied_actions: list[object] = []
            self._terminal = False

        def get_active_player(self) -> str:
            return "alpha"

        def observe(self, player):
            return {"player": player}

        def apply(self, action):
            self.applied_actions.append(action.move)
            self._terminal = True
            return GameResult(
                winner="alpha",
                result="completed",
                reason="applied",
                move_count=len(self.applied_actions),
                illegal_move_count=0,
                final_board="board",
                move_log=[{"move": action.move}],
            )

        def is_terminal(self) -> bool:
            return self._terminal

        def build_result(self, *, result: str, reason: str | None):
            return GameResult(
                winner="alpha",
                result=result,
                reason=reason,
                move_count=len(self.applied_actions),
                illegal_move_count=0,
                final_board="board",
                move_log=[{"move": move} for move in self.applied_actions],
            )

    resolved = SimpleNamespace(
        game_kit=SimpleNamespace(kit_id="gomoku", seat_spec={}, defaults={}),
        env_spec=SimpleNamespace(
            env_id="gomoku_standard",
            defaults={"env_factory": lambda *, sample, resolved, resources, player_specs: FakeEnvironment()},
        ),
        scheduler=TurnScheduler(binding_id="turn/default"),
        resource_spec={},
        player_bindings=(
            PlayerBindingSpec(
                seat="alpha",
                player_id="alpha",
                player_kind="dummy",
                driver_id="player_driver/dummy",
                actions=([5.0, -5.0],),
            ),
        ),
        player_driver_registry=PlayerDriverRegistry(),
        observation_workflow=None,
        support_workflow=GameSupportWorkflow(
            workflow_id="arena/default",
            units_by_hook={
                SupportHook.AFTER_DECIDE: [
                    _RewriteAndRecordUnit("after_decide", rewrite=[2.0, -2.0])
                ],
                SupportHook.BEFORE_APPLY: [
                    _RewriteAndRecordUnit("before_apply"),
                    ContinuousActionShapingUnit(low=-1.0, high=1.0),
                ],
                SupportHook.AFTER_APPLY: [_RewriteAndRecordUnit("after_apply")],
                SupportHook.ON_FINALIZE: [_RewriteAndRecordUnit("on_finalize")],
            },
        ),
    )

    session = GameSession.from_resolved(
        ArenaSample(game_kit="gomoku", env="gomoku_standard"),
        resolved,
        resources={},
    )

    TurnScheduler(binding_id="turn/default").run(session)
    session.finalize()

    assert session.environment.applied_actions == [[1.0, -1.0]]
    assert action_trace == [
        ("after_decide", [5.0, -5.0]),
        ("before_apply", [2.0, -2.0]),
        ("after_apply", [1.0, -1.0]),
        ("on_finalize", None),
    ]


def test_game_session_registers_ws_rgb_live_viewer_when_visualizer_enabled(
    monkeypatch,
) -> None:
    browser_calls: list[str] = []

    class _StubWsHub:
        def __init__(self) -> None:
            self.started = False
            self.registrations: list[object] = []
            self.viewer_url = "http://127.0.0.1:5810/ws_rgb/viewer"

        def start(self) -> None:
            self.started = True

        def register_display(self, registration: object) -> None:
            self.registrations.append(registration)

    class _StubRuntimeServiceHub:
        def __init__(self) -> None:
            self.ws_hub = _StubWsHub()
            self.display_id: str | None = None

        def ensure_ws_rgb_hub(self, factory):
            return self.ws_hub

        def register_display(self, *, display_id: str, hub: object, registration: object) -> None:
            self.display_id = display_id
            hub.register_display(registration)

    class FakeEnvironment:
        def get_active_player(self) -> str:
            return "alpha"

        def get_last_frame(self) -> dict[str, object]:
            return {"board_text": "frame-1", "move_count": 0}

        def observe(self, player):
            return {
                "player": player,
                "legal_moves": ["A1"],
                "active_player": "alpha",
            }

        def apply(self, action):
            del action
            return None

        def is_terminal(self) -> bool:
            return False

    runtime_hub = _StubRuntimeServiceHub()
    monkeypatch.setattr(
        "gage_eval.role.arena.core.game_session.webbrowser.open",
        lambda url: browser_calls.append(url) or True,
    )

    resolved = SimpleNamespace(
        game_kit=SimpleNamespace(kit_id="gomoku", seat_spec={}, defaults={}),
        env_spec=SimpleNamespace(
            env_id="gomoku_standard",
            defaults={
                "env_factory": (
                    lambda *, sample, resolved, resources, player_specs: FakeEnvironment()
                )
            },
        ),
        scheduler=TurnScheduler(binding_id="turn/default"),
        resource_spec={},
        player_bindings=(),
        player_driver_registry=PlayerDriverRegistry(),
        observation_workflow=None,
        support_workflow=None,
    )

    session = GameSession.from_resolved(
        ArenaSample(game_kit="gomoku", env="gomoku_standard"),
        resolved,
        resources={},
        invocation_context=GameArenaInvocationContext(
            adapter_id="arena",
            sample_id="sample-live-1",
            visualizer_config={"enabled": True, "launch_browser": True, "port": 5810},
            runtime_service_hub=runtime_hub,
        ),
    )

    assert session is not None
    assert runtime_hub.display_id == "arena:sample-live-1:gomoku_standard"
    assert runtime_hub.ws_hub.started is True
    assert len(runtime_hub.ws_hub.registrations) == 1
    registration = runtime_hub.ws_hub.registrations[0]
    assert registration.display_id == "arena:sample-live-1:gomoku_standard"
    assert registration.frame_source()["board_text"] == "frame-1"
    assert browser_calls == ["http://127.0.0.1:5810/ws_rgb/viewer"]


def test_game_session_opens_arena_visual_workspace_after_persisting_visual_sidecar(
    monkeypatch,
    tmp_path,
) -> None:
    browser_calls: list[str] = []

    class _StubArenaVisualServer:
        def __init__(self) -> None:
            self.started = False

        def start(self) -> None:
            self.started = True

        def stop(self) -> None:
            self.started = False

        def build_viewer_url(self, session_id: str, *, run_id: str | None = None) -> str:
            suffix = "" if run_id is None else f"?run_id={run_id}"
            return f"http://127.0.0.1:5810/sessions/{session_id}{suffix}"

    class _StubRuntimeServiceHub:
        def __init__(self) -> None:
            self.visualizer = _StubArenaVisualServer()

        def ensure_visualizer(self, factory):
            _ = factory
            return self.visualizer

    class FakeEnvironment:
        def __init__(self) -> None:
            self._terminal = False

        def get_active_player(self) -> str:
            return "alpha"

        def observe(self, player):
            return {
                "board_text": f"board:{player}",
                "legal_moves": ["A1"],
                "active_player": "alpha",
            }

        def apply(self, action):
            del action
            self._terminal = True
            return None

        def is_terminal(self) -> bool:
            return self._terminal

        def build_result(self, *, result: str, reason: str | None):
            return GameResult(
                winner="alpha",
                result=result,
                reason=reason,
                move_count=1,
                illegal_move_count=0,
                final_board="board",
                move_log=[{"player": "alpha", "move": "A1"}],
            )

    runtime_hub = _StubRuntimeServiceHub()
    monkeypatch.setenv("GAGE_EVAL_SAVE_DIR", str(tmp_path / "runs-root"))
    monkeypatch.setattr(
        "gage_eval.role.arena.core.game_session.webbrowser.open",
        lambda url: browser_calls.append(url) or True,
    )
    monkeypatch.setattr(
        "gage_eval.role.arena.core.game_session._build_arena_visual_server",
        lambda config, *, service_hub: runtime_hub.visualizer,
    )

    resolved = SimpleNamespace(
        game_kit=SimpleNamespace(kit_id="gomoku", seat_spec={}, defaults={}),
        env_spec=SimpleNamespace(
            env_id="gomoku_standard",
            defaults={
                "env_factory": (
                    lambda *, sample, resolved, resources, player_specs: FakeEnvironment()
                )
            },
        ),
        scheduler=TurnScheduler(binding_id="turn/default"),
        resource_spec={},
        player_bindings=(),
        player_driver_registry=PlayerDriverRegistry(),
        observation_workflow=None,
        support_workflow=None,
        visualization_spec=SimpleNamespace(
            plugin_id="arena.visualization.gomoku.board_v1",
            game_id="gomoku",
            observer_schema={"supported_modes": ["global", "player"]},
            visual_kind="board",
        ),
    )

    session = GameSession.from_resolved(
        ArenaSample(game_kit="gomoku", env="gomoku_standard"),
        resolved,
        resources={},
        invocation_context=GameArenaInvocationContext(
            adapter_id="arena",
            sample_id="sample-live-2",
            trace=SimpleNamespace(run_id="run-live-2", sample_id="sample-live-2"),
            visualizer_config={
                "enabled": True,
                "launch_browser": True,
                "mode": "arena_visual",
                "port": 5810,
            },
            runtime_service_hub=runtime_hub,
        ),
    )

    session.observe()
    session.apply(ArenaAction(player="alpha", move="A1", raw="A1"))
    session.finalize()

    manifest_path = (
        tmp_path
        / "runs-root"
        / "run-live-2"
        / "replays"
        / "sample-live-2"
        / "arena_visual_session"
        / "v1"
        / "manifest.json"
    )

    assert runtime_hub.visualizer.started is True
    assert manifest_path.exists()
    assert browser_calls == ["http://127.0.0.1:5810/sessions/sample-live-2?run_id=run-live-2"]


def test_game_session_attaches_visualization_resource_to_runtime_resources(
    monkeypatch,
    tmp_path,
) -> None:
    class _StubArenaVisualServer:
        def __init__(self) -> None:
            self.started = False
            self.live_sources: list[object] = []
            self.unregistered: list[tuple[str, str | None]] = []

        def start(self) -> None:
            self.started = True

        def stop(self) -> None:
            self.started = False

        def build_viewer_url(self, session_id: str, *, run_id: str | None = None) -> str:
            suffix = "" if run_id is None else f"?run_id={run_id}"
            return f"http://127.0.0.1:5810/sessions/{session_id}{suffix}"

        def register_live_session(self, source: object) -> None:
            self.live_sources.append(source)

        def unregister_live_session(self, *, session_id: str, run_id: str | None = None) -> None:
            self.unregistered.append((session_id, run_id))

    class _StubRuntimeServiceHub:
        def __init__(self) -> None:
            self.visualizer = _StubArenaVisualServer()

        def ensure_visualizer(self, factory):
            _ = factory
            return self.visualizer

    class FakeEnvironment:
        def __init__(self) -> None:
            self._terminal = False

        def get_active_player(self) -> str:
            return "alpha"

        def observe(self, player):
            return {
                "board_text": f"board:{player}",
                "legal_moves": ["A1"],
                "active_player": "alpha",
            }

        def apply(self, action):
            del action
            self._terminal = True
            return None

        def is_terminal(self) -> bool:
            return self._terminal

        def build_result(self, *, result: str, reason: str | None):
            return GameResult(
                winner="alpha",
                result=result,
                reason=reason,
                move_count=1,
                illegal_move_count=0,
                final_board="board",
                move_log=[{"player": "alpha", "move": "A1"}],
            )

    runtime_hub = _StubRuntimeServiceHub()
    monkeypatch.setenv("GAGE_EVAL_SAVE_DIR", str(tmp_path / "runs-root"))
    monkeypatch.setattr(
        "gage_eval.role.arena.core.game_session.webbrowser.open",
        lambda url: True,
    )
    monkeypatch.setattr(
        "gage_eval.role.arena.core.game_session._build_arena_visual_server",
        lambda config, *, service_hub: runtime_hub.visualizer,
    )

    resources = ArenaResources(
        game_runtime=SimpleNamespace(
            close=lambda: None,
            terminate=lambda: None,
            reap=lambda: None,
        )
    )
    resolved = SimpleNamespace(
        game_kit=SimpleNamespace(kit_id="gomoku", seat_spec={}, defaults={}),
        env_spec=SimpleNamespace(
            env_id="gomoku_standard",
            defaults={
                "env_factory": (
                    lambda *, sample, resolved, resources, player_specs: FakeEnvironment()
                )
            },
        ),
        scheduler=TurnScheduler(binding_id="turn/default"),
        resource_spec={},
        player_bindings=(),
        player_driver_registry=PlayerDriverRegistry(),
        observation_workflow=None,
        support_workflow=None,
        visualization_spec=SimpleNamespace(
            plugin_id="arena.visualization.gomoku.board_v1",
            game_id="gomoku",
            observer_schema={"supported_modes": ["global", "player"]},
            visual_kind="board",
        ),
    )

    session = GameSession.from_resolved(
        ArenaSample(game_kit="gomoku", env="gomoku_standard"),
        resolved,
        resources=resources,
        invocation_context=GameArenaInvocationContext(
            adapter_id="arena",
            sample_id="sample-live-resource",
            trace=SimpleNamespace(
                run_id="run-live-resource",
                sample_id="sample-live-resource",
            ),
            visualizer_config={
                "enabled": True,
                "launch_browser": True,
                "mode": "arena_visual",
                "port": 5810,
            },
            runtime_service_hub=runtime_hub,
        ),
    )

    assert isinstance(resources.visualization, VisualizationSession)
    assert resources.visualization.phase is VisualizationPhase.LIVE
    assert "visualization_resource" in resources.resource_categories

    session.observe()
    session.apply(ArenaAction(player="alpha", move="A1", raw="A1"))
    session.finalize()

    assert resources.visualization.phase is VisualizationPhase.REPLAY_READY
    assert resources.visualization.artifacts["replay_ref"].endswith("replay.json")

    ArenaResourceControl().release(resources)

    assert resources.visualization.phase is VisualizationPhase.RELEASED
    assert resources.resource_artifacts["visualization_phase"] == "released"
    assert runtime_hub.visualizer.unregistered == [
        ("sample-live-resource", "run-live-resource")
    ]


def test_game_session_registers_live_arena_visual_source_before_finalize(
    monkeypatch,
) -> None:
    browser_calls: list[str] = []

    class _StubArenaVisualServer:
        def __init__(self) -> None:
            self.started = False
            self.live_sources: list[object] = []

        def start(self) -> None:
            self.started = True

        def stop(self) -> None:
            self.started = False

        def build_viewer_url(self, session_id: str, *, run_id: str | None = None) -> str:
            suffix = "" if run_id is None else f"?run_id={run_id}"
            return f"http://127.0.0.1:5810/sessions/{session_id}{suffix}"

        def register_live_session(self, source: object) -> None:
            self.live_sources.append(source)

    class _StubRuntimeServiceHub:
        def __init__(self) -> None:
            self.visualizer = _StubArenaVisualServer()

        def ensure_visualizer(self, factory):
            _ = factory
            return self.visualizer

    class FakeEnvironment:
        def __init__(self) -> None:
            self._terminal = False

        def get_active_player(self) -> str:
            return "alpha"

        def observe(self, player):
            return {
                "board_text": f"board:{player}",
                "legal_moves": ["A1"],
                "active_player": "alpha",
            }

        def apply(self, action):
            del action
            self._terminal = True
            return None

        def is_terminal(self) -> bool:
            return self._terminal

    runtime_hub = _StubRuntimeServiceHub()
    monkeypatch.setattr(
        "gage_eval.role.arena.core.game_session.webbrowser.open",
        lambda url: browser_calls.append(url) or True,
    )
    monkeypatch.setattr(
        "gage_eval.role.arena.core.game_session._build_arena_visual_server",
        lambda config, *, service_hub: runtime_hub.visualizer,
    )

    resolved = SimpleNamespace(
        game_kit=SimpleNamespace(kit_id="gomoku", seat_spec={}, defaults={}),
        env_spec=SimpleNamespace(
            env_id="gomoku_standard",
            defaults={
                "env_factory": (
                    lambda *, sample, resolved, resources, player_specs: FakeEnvironment()
                )
            },
        ),
        scheduler=TurnScheduler(binding_id="turn/default"),
        resource_spec={},
        player_bindings=(),
        player_driver_registry=PlayerDriverRegistry(),
        observation_workflow=None,
        support_workflow=None,
        visualization_spec=SimpleNamespace(
            plugin_id="arena.visualization.gomoku.board_v1",
            game_id="gomoku",
            observer_schema={"supported_modes": ["global", "player"]},
            visual_kind="board",
        ),
    )

    session = GameSession.from_resolved(
        ArenaSample(game_kit="gomoku", env="gomoku_standard"),
        resolved,
        resources={},
        invocation_context=GameArenaInvocationContext(
            adapter_id="arena",
            sample_id="sample-live-3",
            trace=SimpleNamespace(run_id="run-live-3", sample_id="sample-live-3"),
            visualizer_config={
                "enabled": True,
                "launch_browser": True,
                "mode": "arena_visual",
                "live_scene_scheme": "http_pull",
                "port": 5810,
            },
            runtime_service_hub=runtime_hub,
        ),
    )

    assert session is not None
    assert runtime_hub.visualizer.started is True
    assert len(runtime_hub.visualizer.live_sources) == 1
    live_source = runtime_hub.visualizer.live_sources[0]
    assert getattr(live_source, "session_id") == "sample-live-3"
    assert getattr(live_source, "run_id") == "run-live-3"
    assert getattr(live_source, "live_scene_scheme") == "http_pull"
    assert getattr(live_source, "visualization_spec") is resolved.visualization_spec
    assert browser_calls == ["http://127.0.0.1:5810/sessions/sample-live-3?run_id=run-live-3"]


def test_game_session_arena_visual_requires_explicit_finish_after_post_end_interaction(
    monkeypatch,
) -> None:
    class _StubArenaVisualServer:
        def __init__(self) -> None:
            self.started = False
            self.live_sources: list[object] = []

        def start(self) -> None:
            self.started = True

        def stop(self) -> None:
            self.started = False

        def build_viewer_url(self, session_id: str, *, run_id: str | None = None) -> str:
            suffix = "" if run_id is None else f"?run_id={run_id}"
            return f"http://127.0.0.1:5810/sessions/{session_id}{suffix}"

        def register_live_session(self, source: object) -> None:
            self.live_sources.append(source)

    class _StubRuntimeServiceHub:
        def __init__(self) -> None:
            self.visualizer = _StubArenaVisualServer()

        def ensure_visualizer(self, factory):
            _ = factory
            return self.visualizer

    class FakeEnvironment:
        def __init__(self) -> None:
            self._terminal = False

        def get_active_player(self) -> str:
            return "alpha"

        def observe(self, player):
            return {
                "board_text": f"board:{player}",
                "legal_moves": ["A1"],
                "active_player": "alpha",
            }

        def apply(self, action):
            del action
            self._terminal = True
            return None

        def is_terminal(self) -> bool:
            return self._terminal

        def build_result(self, *, result: str, reason: str | None):
            return GameResult(
                winner="alpha",
                result=result,
                reason=reason,
                move_count=1,
                illegal_move_count=0,
                final_board="board",
                move_log=[{"player": "alpha", "move": "A1"}],
            )

    runtime_hub = _StubRuntimeServiceHub()
    monkeypatch.setattr(
        "gage_eval.role.arena.core.game_session.webbrowser.open",
        lambda url: True,
    )
    monkeypatch.setattr(
        "gage_eval.role.arena.core.game_session._build_arena_visual_server",
        lambda config, *, service_hub: runtime_hub.visualizer,
    )

    resolved = SimpleNamespace(
        game_kit=SimpleNamespace(kit_id="gomoku", seat_spec={}, defaults={}),
        env_spec=SimpleNamespace(
            env_id="gomoku_standard",
            defaults={
                "env_factory": (
                    lambda *, sample, resolved, resources, player_specs: FakeEnvironment()
                )
            },
        ),
        scheduler=TurnScheduler(binding_id="turn/default"),
        resource_spec={},
        player_bindings=(),
        player_driver_registry=PlayerDriverRegistry(),
        observation_workflow=None,
        support_workflow=None,
        visualization_spec=SimpleNamespace(
            plugin_id="arena.visualization.gomoku.board_v1",
            game_id="gomoku",
            observer_schema={"supported_modes": ["global", "player"]},
            visual_kind="board",
        ),
    )

    session = GameSession.from_resolved(
        ArenaSample(game_kit="gomoku", env="gomoku_standard"),
        resolved,
        resources={},
        invocation_context=GameArenaInvocationContext(
            adapter_id="arena",
            sample_id="sample-live-finish",
            trace=SimpleNamespace(run_id="run-live-finish", sample_id="sample-live-finish"),
            visualizer_config={
                "enabled": True,
                "launch_browser": True,
                "mode": "arena_visual",
                "live_scene_scheme": "http_pull",
                "linger_after_finish_s": 0.05,
                "port": 5810,
            },
            runtime_service_hub=runtime_hub,
        ),
    )
    session.observe()
    session.apply(ArenaAction(player="alpha", move="A1", raw="A1"))

    finalize_thread = threading.Thread(target=session.finalize, daemon=True)
    started_at = time.monotonic()
    finalize_thread.start()

    assert len(runtime_hub.visualizer.live_sources) == 1
    live_source = runtime_hub.visualizer.live_sources[0]
    time.sleep(0.02)
    assert getattr(live_source, "load_session")().lifecycle == "live_ended"
    getattr(live_source, "apply_control_command")(ControlCommand(command_type="replay"))

    finalize_thread.join(timeout=0.15)
    assert finalize_thread.is_alive() is True

    getattr(live_source, "apply_control_command")(ControlCommand(command_type="finish"))
    finalize_thread.join(timeout=1.0)

    assert finalize_thread.is_alive() is False
    assert time.monotonic() - started_at >= 0.05


def test_game_session_arena_visual_finish_stops_a_live_run_before_finalize(
    monkeypatch,
) -> None:
    class _StubArenaVisualServer:
        def __init__(self) -> None:
            self.started = False
            self.live_sources: list[object] = []

        def start(self) -> None:
            self.started = True

        def stop(self) -> None:
            self.started = False

        def build_viewer_url(self, session_id: str, *, run_id: str | None = None) -> str:
            suffix = "" if run_id is None else f"?run_id={run_id}"
            return f"http://127.0.0.1:5811/sessions/{session_id}{suffix}"

        def register_live_session(self, source: object) -> None:
            self.live_sources.append(source)

    class _StubRuntimeServiceHub:
        def __init__(self) -> None:
            self.visualizer = _StubArenaVisualServer()

        def ensure_visualizer(self, factory):
            _ = factory
            return self.visualizer

    class FakeEnvironment:
        def __init__(self) -> None:
            self.closed = False

        def get_active_player(self) -> str:
            return "alpha"

        def observe(self, player):
            return {
                "board_text": f"board:{player}",
                "legal_moves": ["A1"],
                "active_player": "alpha",
            }

        def apply(self, action):
            del action
            return None

        def is_terminal(self) -> bool:
            return False

        def close(self) -> None:
            self.closed = True

        def build_result(self, *, result: str, reason: str | None):
            return GameResult(
                winner=None,
                result=result,
                reason=reason,
                move_count=0,
                illegal_move_count=0,
                final_board="board",
                move_log=[],
            )

    runtime_hub = _StubRuntimeServiceHub()
    monkeypatch.setattr(
        "gage_eval.role.arena.core.game_session.webbrowser.open",
        lambda url: True,
    )
    monkeypatch.setattr(
        "gage_eval.role.arena.core.game_session._build_arena_visual_server",
        lambda config, *, service_hub: runtime_hub.visualizer,
    )

    resolved = SimpleNamespace(
        game_kit=SimpleNamespace(kit_id="gomoku", seat_spec={}, defaults={}),
        env_spec=SimpleNamespace(
            env_id="gomoku_standard",
            defaults={
                "env_factory": (
                    lambda *, sample, resolved, resources, player_specs: FakeEnvironment()
                )
            },
        ),
        scheduler=RealTimeTickScheduler(binding_id="real_time_tick/default"),
        resource_spec={},
        player_bindings=(),
        player_driver_registry=PlayerDriverRegistry(),
        observation_workflow=None,
        support_workflow=None,
        visualization_spec=SimpleNamespace(
            plugin_id="arena.visualization.gomoku.board_v1",
            game_id="gomoku",
            observer_schema={"supported_modes": ["global", "player"]},
            visual_kind="board",
        ),
    )

    session = GameSession.from_resolved(
        ArenaSample(game_kit="gomoku", env="gomoku_standard"),
        resolved,
        resources={},
        invocation_context=GameArenaInvocationContext(
            adapter_id="arena",
            sample_id="sample-live-stop",
            trace=SimpleNamespace(run_id="run-live-stop", sample_id="sample-live-stop"),
            visualizer_config={
                "enabled": True,
                "launch_browser": True,
                "mode": "arena_visual",
                "live_scene_scheme": "low_latency_channel",
                "linger_after_finish_s": 5.0,
                "port": 5811,
            },
            runtime_service_hub=runtime_hub,
        ),
    )

    assert len(runtime_hub.visualizer.live_sources) == 1
    live_source = runtime_hub.visualizer.live_sources[0]

    started_at = time.monotonic()
    getattr(live_source, "apply_control_command")(ControlCommand(command_type="finish"))

    assert session.should_stop() is True
    assert isinstance(session.final_result, GameResult)
    assert session.final_result.result == "terminated"
    assert session.final_result.reason == "user_finish"
    assert getattr(session.environment, "closed", False) is True

    result = session.finalize()

    assert isinstance(result, GameResult)
    assert result.result == "terminated"
    assert result.reason == "user_finish"
    assert time.monotonic() - started_at < 0.5

def test_game_session_records_action_trace_fields_for_illegal_result() -> None:
    class FakeEnvironment:
        def __init__(self) -> None:
            self._terminal = False

        def get_active_player(self) -> str:
            return "alpha"

        def observe(self, player):
            return {
                "player": player,
                "legal_moves": ["A1", "B2"],
                "active_player": "alpha",
            }

        def apply(self, action):
            self._terminal = True
            return GameResult(
                winner="beta",
                result="loss",
                reason="occupied",
                move_count=0,
                illegal_move_count=1,
                final_board="board",
                move_log=[],
            )

        def is_terminal(self) -> bool:
            return self._terminal

    class FakePlayer:
        player_id = "alpha"

        def next_action(self, observation) -> ArenaAction:
            del observation
            return ArenaAction(
                player="alpha",
                move="B2",
                raw='{"move":"B2"}',
                metadata={
                    "driver_id": "player_driver/llm_backend",
                    "player_type": "llm",
                    "retry_count": "2",
                },
            )

    session = GameSession(
        sample=ArenaSample(game_kit="gomoku", env="gomoku_standard"),
        environment=FakeEnvironment(),
        player_specs=(FakePlayer(),),
    )

    observation = session.observe()
    action = session.decide_current_player(observation)
    session.apply(action)

    assert len(session.arena_trace) == 1
    entry = session.arena_trace[0]
    assert entry["player_id"] == "alpha"
    assert entry["action_raw"] == '{"move":"B2"}'
    assert entry["action_applied"] == "B2"
    assert entry["retry_count"] == 2
    assert entry["is_action_legal"] is False
    assert entry["illegal_reason"] == "occupied"
    assert entry["t_action_submitted_ms"] >= entry["t_obs_ready_ms"]

def test_game_session_advance_uses_environment_reported_progress() -> None:
    class FakeEnvironment:
        def __init__(self) -> None:
            self._deltas = [0, 1, 0]

        def consume_session_progress_delta(self) -> int:
            if self._deltas:
                return self._deltas.pop(0)
            return 1

        def is_terminal(self) -> bool:
            return False

    session = GameSession(
        sample=ArenaSample(game_kit="vizdoom", env="duel_map01"),
        environment=FakeEnvironment(),
    )

    session.advance()
    assert session.tick == 0
    assert session.step == 0

    session.advance()
    assert session.tick == 1
    assert session.step == 1

    session.advance()
    assert session.tick == 1
    assert session.step == 1


def build_scheduler_owned_openra_session_with_empty_queue() -> GameSession:
    class FakeEnvironment:
        def get_active_player(self) -> str:
            return "player_0"

        def observe(self, player_id: str) -> object:
            return SimpleNamespace(
                active_player=player_id,
                legal_actions_items=("noop",),
                view_text="openra frame",
                board_text="openra frame",
            )

        def consume_session_progress_delta(self) -> int:
            return 1

        def is_terminal(self) -> bool:
            return False

    return GameSession(
        sample=ArenaSample(game_kit="openra", env="ra_skirmish_1v1"),
        environment=FakeEnvironment(),
        player_specs=(SimpleNamespace(player_id="player_0", player_kind="human"),),
        runtime_profile=ResolvedRuntimeProfile(
            scheduler_binding="real_time_tick/default",
            scheduler_family="real_time_tick",
            tick_interval_ms=50,
            pure_human_realtime=True,
            scheduler_owns_realtime_clock=True,
            supports_low_latency_realtime_input=True,
            supports_realtime_input_websocket=True,
            human_realtime_inputs=(
                HumanRealtimeInputProfile(
                    player_id="player_0",
                    semantics="queued_command",
                    tick_interval_ms=50,
                ),
            ),
            realtime_human_control=RealtimeHumanControlProfile(
                mode="scheduler_owned_human_realtime",
                activation_scope="pure_human_only",
                input_model="queued_command",
                tick_interval_ms=50,
                input_transport="realtime_ws",
                frame_output_hz=20,
                artifact_sampling_mode="async_decimated_live",
                fallback_move="noop",
            ),
        ),
    )


def test_scheduler_owned_queued_command_idle_tick_does_not_increment_step_or_move_count() -> None:
    session = build_scheduler_owned_openra_session_with_empty_queue()

    session.observe()
    session.advance()

    assert session.tick == 1
    assert session.step == 0
    assert session.final_result is None


def test_game_session_advance_records_fresh_visual_snapshot_from_environment_frame() -> None:
    class FakeEnvironment:
        def consume_session_progress_delta(self) -> int:
            return 1

        def is_terminal(self) -> bool:
            return False

        def get_last_frame(self) -> dict[str, object]:
            return {
                "active_player_id": "pilot_1",
                "board_text": "fresh frame state",
                "legal_actions": {"items": ["RIGHT"]},
                "last_move": "RIGHT",
                "reward": 1.0,
                "move_count": 1,
                "metadata": {"reward": 1.0, "last_move": "RIGHT"},
            }

    recorder = ArenaVisualSessionRecorder(
        plugin_id="arena.visualization.pettingzoo.frame_v1",
        game_id="pettingzoo",
        scheduling_family="agent_cycle",
        session_id="sample-visual-snapshot",
        visual_kind="frame",
    )
    session = GameSession(
        sample=ArenaSample(game_kit="pettingzoo", env="space_invaders"),
        environment=FakeEnvironment(),
        visual_recorder=recorder,
    )
    session._last_observation = {  # noqa: SLF001
        "board_text": "stale observation state",
        "active_player": "pilot_0",
        "legal_actions": {"items": ["NOOP"]},
        "view": {"text": "stale observation state"},
        "context": {"step": 0},
    }

    session.advance()

    snapshot = recorder.export_live_state().snapshot_payloads[0]["snapshot"]
    observation = snapshot["observation"]

    assert observation["board_text"] == "fresh frame state"
    assert observation["active_player"] == "pilot_1"
    assert observation["legal_actions"] == {"items": ["RIGHT"]}


def test_game_session_realtime_live_snapshots_throttle_inline_frame_images(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeEnvironment:
        def consume_session_progress_delta(self) -> int:
            return 1

        def is_terminal(self) -> bool:
            return False

        def get_last_frame(self) -> dict[str, object]:
            return {
                "board_text": "live frame state",
                "stream_id": "main",
                "metadata": {"stream_id": "main"},
                "view": {
                    "text": "live frame state",
                    "image": {
                        "data_url": "data:image/png;base64,Zm9v",
                        "mimeType": "image/png",
                    },
                },
            }

    recorder = ArenaVisualSessionRecorder(
        plugin_id="arena.visualization.retro.frame_v1",
        game_id="retro_mario",
        scheduling_family="real_time_tick",
        session_id="sample-live-cadence",
        visual_kind="frame",
    )
    session = GameSession(
        sample=ArenaSample(game_kit="retro_platformer", env="retro_mario"),
        environment=FakeEnvironment(),
        visual_recorder=recorder,
        runtime_profile=ResolvedRuntimeProfile(
            scheduler_binding="real_time_tick/default",
            scheduler_family="real_time_tick",
            tick_interval_ms=16,
            pure_human_realtime=True,
            scheduler_owns_realtime_clock=True,
            supports_low_latency_realtime_input=True,
            supports_realtime_input_websocket=True,
            human_realtime_inputs=(
                HumanRealtimeInputProfile(
                    player_id="player_0",
                    semantics="continuous_state",
                    tick_interval_ms=16,
                ),
            ),
        ),
    )
    session._visualization_mode = "arena_visual"  # noqa: SLF001
    session._visualization_live_scene_scheme = "low_latency_channel"  # noqa: SLF001

    current_ts = {"value": 1000}
    monkeypatch.setattr(
        "gage_eval.role.arena.core.game_session.wall_clock_ms",
        lambda: current_ts["value"],
    )

    session.advance()
    session.capture_output_tick()
    current_ts["value"] = 1260
    session.advance()
    session.capture_output_tick()

    snapshots = recorder.export_live_state().snapshot_payloads
    first_observation = snapshots[0]["snapshot"]["observation"]
    second_observation = snapshots[1]["snapshot"]["observation"]

    assert first_observation["view"]["image"]["data_url"] == "data:image/png;base64,Zm9v"
    assert "image" not in second_observation["view"]


def test_game_session_realtime_snapshot_cadence_respects_frame_output_hz_policy(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeEnvironment:
        def consume_session_progress_delta(self) -> int:
            return 1

        def is_terminal(self) -> bool:
            return False

        def get_last_frame(self) -> dict[str, object]:
            return {
                "board_text": "live frame state",
                "stream_id": "main",
                "metadata": {"stream_id": "main"},
                "view": {"text": "live frame state"},
            }

    recorder = ArenaVisualSessionRecorder(
        plugin_id="arena.visualization.retro.frame_v1",
        game_id="retro_mario",
        scheduling_family="real_time_tick",
        session_id="sample-live-policy-cadence",
        visual_kind="frame",
    )
    session = GameSession(
        sample=ArenaSample(game_kit="retro_platformer", env="retro_mario"),
        environment=FakeEnvironment(),
        visual_recorder=recorder,
        runtime_profile=ResolvedRuntimeProfile(
            scheduler_binding="real_time_tick/default",
            scheduler_family="real_time_tick",
            tick_interval_ms=16,
            pure_human_realtime=True,
            scheduler_owns_realtime_clock=True,
            supports_low_latency_realtime_input=True,
            supports_realtime_input_websocket=True,
            human_realtime_inputs=(
                HumanRealtimeInputProfile(
                    player_id="player_0",
                    semantics="continuous_state",
                    tick_interval_ms=16,
                ),
            ),
            realtime_human_control=RealtimeHumanControlProfile(
                mode="scheduler_owned_human_realtime",
                activation_scope="pure_human_only",
                input_model="continuous_state",
                tick_interval_ms=16,
                input_transport="realtime_ws",
                frame_output_hz=10,
                artifact_sampling_mode="async_decimated_live",
                fallback_move="noop",
            ),
        ),
    )
    session._visualization_mode = "arena_visual"  # noqa: SLF001
    session._visualization_live_scene_scheme = "low_latency_channel"  # noqa: SLF001

    current_ts = {"value": 1000}
    monkeypatch.setattr(
        "gage_eval.role.arena.core.game_session.wall_clock_ms",
        lambda: current_ts["value"],
    )

    session.advance()
    session.capture_output_tick()
    current_ts["value"] = 1090
    session.advance()
    session.capture_output_tick()
    current_ts["value"] = 1105
    session.advance()
    session.capture_output_tick()

    recorder.export_live_header()
    snapshots = recorder.export_live_state().snapshot_payloads
    assert len(snapshots) == 2
    assert snapshots[0]["tsMs"] == 1000
    assert snapshots[1]["tsMs"] == 1105


def test_game_session_scheduler_owned_realtime_uses_lightweight_visual_events(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeEnvironment:
        def get_active_player(self) -> str:
            return "player_0"

        def observe(self, player: str) -> ArenaObservation:
            del player
            return SimpleNamespace(
                active_player="player_0",
                legal_actions_items=("noop", "right"),
                view_text="frame",
                board_text="frame",
            )

        def apply(self, action: ArenaAction) -> None:
            del action
            return None

        def consume_session_progress_delta(self) -> int:
            return 1

        def is_terminal(self) -> bool:
            return False

        def get_last_frame(self) -> dict[str, object]:
            return {"board_text": "frame", "stream_id": "main", "metadata": {"stream_id": "main"}}

    recorder = ArenaVisualSessionRecorder(
        plugin_id="arena.visualization.retro.frame_v1",
        game_id="retro_mario",
        scheduling_family="real_time_tick",
        session_id="sample-lightweight-realtime",
        visual_kind="frame",
    )
    session = GameSession(
        sample=ArenaSample(game_kit="retro_platformer", env="retro_mario"),
        environment=FakeEnvironment(),
        player_specs=(
            SimpleNamespace(player_id="player_0", player_kind="human"),
        ),
        visual_recorder=recorder,
        runtime_profile=ResolvedRuntimeProfile(
            scheduler_binding="real_time_tick/default",
            scheduler_family="real_time_tick",
            tick_interval_ms=16,
            pure_human_realtime=True,
            scheduler_owns_realtime_clock=True,
            supports_low_latency_realtime_input=True,
            supports_realtime_input_websocket=True,
            human_realtime_inputs=(
                HumanRealtimeInputProfile(
                    player_id="player_0",
                    semantics="continuous_state",
                    tick_interval_ms=16,
                ),
            ),
        ),
    )
    session._visualization_mode = "arena_visual"  # noqa: SLF001
    session._visualization_live_scene_scheme = "low_latency_channel"  # noqa: SLF001
    monkeypatch.setattr(
        "gage_eval.role.arena.core.game_session.wall_clock_ms",
        lambda: 1000,
    )

    observation = session.observe()
    action = ArenaAction(player="player_0", move="noop", raw="noop")
    session._last_observation = observation  # noqa: SLF001
    session.apply(action)
    session.advance()
    session.capture_output_tick()

    live_state = recorder.export_live_state()
    visual_session = recorder.build_visual_session()

    assert [event.type for event in live_state.timeline_events] == ["snapshot"]
    assert visual_session.scheduling.accepts_human_intent is True


def test_game_session_low_latency_live_snapshots_strip_inline_images_for_mixed_realtime_sessions() -> None:
    class FakeEnvironment:
        def __init__(self) -> None:
            self._tick = 0

        def reset(self) -> None:
            return None

        def get_active_player(self) -> str:
            return "player_0"

        def observe(self, player: str) -> ArenaObservation:
            del player
            return ArenaObservation(
                active_player="player_0",
                legal_actions_items=("noop", "right"),
                view_text="live frame state",
                metadata={"stream_id": "main"},
            )

        def apply_action(self, action: ArenaAction) -> None:
            del action
            return None

        def advance(self) -> None:
            self._tick += 1

        def is_terminal(self) -> bool:
            return False

        def get_last_frame(self) -> dict[str, object]:
            return {
                "board_text": "live frame state",
                "stream_id": "main",
                "metadata": {"stream_id": "main"},
                "view": {
                    "text": "live frame state",
                    "image": {
                        "data_url": "data:image/png;base64,Zm9v",
                        "mimeType": "image/png",
                    },
                },
            }

    recorder = ArenaVisualSessionRecorder(
        plugin_id="arena.visualization.retro.frame_v1",
        game_id="retro_mario",
        scheduling_family="real_time_tick",
        session_id="sample-live-mixed-cadence",
        visual_kind="frame",
    )
    session = GameSession(
        sample=ArenaSample(game_kit="retro_platformer", env="retro_mario"),
        environment=FakeEnvironment(),
        visual_recorder=recorder,
        runtime_profile=ResolvedRuntimeProfile(
            scheduler_binding="real_time_tick/default",
            scheduler_family="real_time_tick",
            tick_interval_ms=16,
            pure_human_realtime=False,
            scheduler_owns_realtime_clock=False,
            supports_low_latency_realtime_input=True,
            supports_realtime_input_websocket=True,
            human_realtime_inputs=(
                HumanRealtimeInputProfile(
                    player_id="player_0",
                    semantics="continuous_state",
                    tick_interval_ms=16,
                ),
            ),
        ),
    )
    session._visualization_mode = "arena_visual"  # noqa: SLF001
    session._visualization_live_scene_scheme = "low_latency_channel"  # noqa: SLF001

    session.advance()
    session.advance()

    snapshots = recorder.export_live_state().snapshot_payloads
    first_observation = snapshots[0]["snapshot"]["observation"]
    second_observation = snapshots[1]["snapshot"]["observation"]

    assert first_observation["view"]["image"]["data_url"] == "data:image/png;base64,Zm9v"
    assert "image" not in second_observation["view"]


def test_vizdoom_realtime_session_max_steps_counts_backend_flush_rounds() -> None:
    class ScriptedPlayer:
        def __init__(self, player_id: str, moves: list[str]) -> None:
            self.player_id = player_id
            self._moves = list(moves)
            self._index = 0

        def next_action(self, observation) -> ArenaAction:
            del observation
            move = self._moves[self._index]
            self._index += 1
            return ArenaAction(player=self.player_id, move=move, raw=move)

    environment = DuelMap01Environment(
        backend_mode="dummy",
        stub_max_rounds=10,
        player_ids=("doom_alpha", "doom_beta"),
        player_names={"doom_alpha": "doom_alpha", "doom_beta": "doom_beta"},
        replay_output_dir=None,
        show_pov=False,
        show_automap=False,
    )
    environment.reset()
    session = GameSession(
        sample=ArenaSample(game_kit="vizdoom", env="duel_map01"),
        environment=environment,
        player_specs=(
            ScriptedPlayer("doom_alpha", ["1", "2", "1"]),
            ScriptedPlayer("doom_beta", ["3", "2", "3"]),
        ),
        max_steps=3,
    )

    RealTimeTickScheduler(binding_id="real_time_tick/default").run(session)

    assert session.tick == 3
    assert session.step == 3
    assert environment._tick == 3  # noqa: SLF001
    assert session.final_result is not None
    assert session.final_result.result == "max_steps"
    assert session.final_result.move_count == 3


def test_realtime_tick_scheduler_uses_fixed_cadence_for_scheduler_owned_human_realtime(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class RealtimeHumanPlayer:
        player_id = "player_0"
        display_name = "player_0"
        seat = "player_0"
        player_kind = "human"
        metadata: dict[str, object] = {}

        def __init__(self) -> None:
            self.observations: list[object] = []

        def next_action(self, observation) -> ArenaAction:
            self.observations.append(observation)
            return ArenaAction(player="player_0", move="noop", raw="noop")

    class AsyncEnvironment:
        def __init__(self) -> None:
            self._terminal = False
            self.applied_moves: list[str] = []

        def get_active_player(self) -> str:
            return "player_0"

        def observe(self, player_id: str) -> object:
            return SimpleNamespace(
                active_player=player_id,
                legal_actions_items=("noop", "right"),
                view_text="retro frame",
                board_text="retro frame",
                last_action=None,
            )

        def apply(self, action: ArenaAction) -> GameResult:
            self.applied_moves.append(action.move)
            self._terminal = True
            return GameResult(
                winner=action.player,
                result="completed",
                reason="applied",
                move_count=1,
                illegal_move_count=0,
                final_board="board",
                move_log=[{"move": action.move}],
            )

        def is_terminal(self) -> bool:
            return self._terminal

    current_time = {"value": 100.0}
    slept: list[float] = []

    def fake_monotonic() -> float:
        return current_time["value"]

    def fake_sleep(delay: float) -> None:
        slept.append(delay)
        current_time["value"] += delay

    monkeypatch.setattr(
        real_time_tick_module,
        "time",
        SimpleNamespace(sleep=fake_sleep, monotonic=fake_monotonic),
        raising=False,
    )

    player = RealtimeHumanPlayer()
    environment = AsyncEnvironment()
    session = GameSession(
        sample=ArenaSample(game_kit="retro_platformer", env="retro_mario"),
        environment=environment,
        player_specs=(player,),
        max_steps=1,
        runtime_profile=ResolvedRuntimeProfile(
            scheduler_binding="real_time_tick/default",
            scheduler_family="real_time_tick",
            tick_interval_ms=16,
            pure_human_realtime=True,
            scheduler_owns_realtime_clock=True,
            supports_low_latency_realtime_input=True,
            supports_realtime_input_websocket=True,
            human_realtime_inputs=(
                HumanRealtimeInputProfile(
                    player_id="player_0",
                    semantics="continuous_state",
                    tick_interval_ms=16,
                ),
            ),
        ),
    )

    RealTimeTickScheduler(binding_id="real_time_tick/default").run(session)

    assert len(player.observations) == 1
    assert sum(slept) <= 0.016001
    assert environment.applied_moves == ["noop"]
    assert session.final_result is not None
    assert session.final_result.result == "completed"


def test_arena_output_writer_recursively_freezes_nested_trace_values() -> None:
    session = GameSession(sample=ArenaSample(game_kit="gomoku", env=None))
    session.arena_trace.append(
        {
            "event": "tick",
            "payload": {
                "scores": [{"name": "agent_a", "points": [1, 2]}],
                "round": 1,
            },
        }
    )

    output = ArenaOutputWriter().finalize(session)
    trace_entry = output.arena_trace[0]

    session.arena_trace[0]["payload"]["scores"][0]["points"].append(3)
    session.arena_trace[0]["payload"]["round"] = 2

    assert trace_entry["payload"]["scores"][0]["points"] == (1, 2)
    assert trace_entry["payload"]["round"] == 1

    with pytest.raises(TypeError):
        trace_entry["payload"]["round"] = 3


def test_arena_output_writer_freezes_sample_and_result_snapshots() -> None:
    session = GameSession(
        sample=ArenaSample(
            game_kit="gomoku",
            env="gomoku_standard",
            players=({"seat": "black"},),
            runtime_overrides={"board_size": 3},
        ),
        final_result=GameResult(
            winner="Black",
            result="win",
            reason="five_in_row",
            move_count=5,
            illegal_move_count=0,
            final_board="board",
            move_log=[{"index": 1, "player": "Black", "coord": "A1"}],
        ),
    )

    output = ArenaOutputWriter().finalize(session)

    session.sample.players[0]["seat"] = "mutated"
    session.sample.runtime_overrides["board_size"] = 9
    session.final_result.move_log[0]["coord"] = "Z9"

    assert output.sample.players[0]["seat"] == "black"
    assert output.sample.runtime_overrides["board_size"] == 3
    assert output.result.move_log[0]["coord"] == "A1"

    with pytest.raises(TypeError):
        output.sample.players[0]["seat"] = "changed"
    with pytest.raises(TypeError):
        output.result.move_log[0]["coord"] = "B2"
