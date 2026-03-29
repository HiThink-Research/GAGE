from __future__ import annotations

import inspect
import os
import shutil
import subprocess
import sys
import time
import webbrowser
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import TYPE_CHECKING
from typing import Any, Mapping

from loguru import logger

from gage_eval.role.arena.core.invocation import GameArenaInvocationContext
from gage_eval.role.arena.core.players import BoundPlayer
from gage_eval.role.arena.core.types import ArenaSample
from gage_eval.role.arena.human_input_protocol import SampleActionRouter
from gage_eval.role.arena.schedulers._scheduler_utils import (
    detect_illegal_reason,
    finalize_trace_entry,
    infer_legality,
    infer_retry_count,
    make_trace_entry,
    set_trace_action_fields,
    wall_clock_ms,
)
from gage_eval.role.arena.support.context import SupportContext
from gage_eval.role.arena.support.hooks import SupportHook
from gage_eval.role.arena.types import ArenaAction
from gage_eval.role.arena.visualization.recorder import ArenaVisualSessionRecorder
from gage_eval.role.arena.visualization.artifacts import to_visual_json_safe
from gage_eval.role.arena.replay_paths import resolve_replay_manifest_path
from gage_eval.role.arena.replay_schema_writer import update_replay_manifest_visual_session_ref

if TYPE_CHECKING:
    from gage_eval.tools.ws_rgb_server import DisplayRegistration, WsRgbHubServer


@dataclass
class GameSession:
    sample: ArenaSample
    environment: object | None = None
    player_specs: tuple[BoundPlayer, ...] = ()
    observation_workflow: object = None
    support_workflow: object = None
    visualization_spec: object | None = None
    max_steps: int = 256
    final_result: object | None = None
    tick: int = 0
    step: int = 0
    arena_trace: list[dict[str, object]] = field(default_factory=list)
    invocation_context: GameArenaInvocationContext | None = None
    visual_recorder: ArenaVisualSessionRecorder | None = None
    _current_trace_entry: dict[str, Any] | None = field(default=None, init=False, repr=False)
    _last_observation: Any | None = field(default=None, init=False, repr=False)
    _finalized: bool = field(default=False, init=False, repr=False)
    _sample_action_router: SampleActionRouter | None = field(default=None, init=False, repr=False)
    _sample_action_server: Any | None = field(default=None, init=False, repr=False)
    _visualization_display_id: str | None = field(default=None, init=False, repr=False)
    _visualization_mode: str | None = field(default=None, init=False, repr=False)
    _visualization_viewer_url: str | None = field(default=None, init=False, repr=False)
    _visualization_run_id: str | None = field(default=None, init=False, repr=False)
    _visualization_launch_browser: bool = field(default=False, init=False, repr=False)
    _visualization_browser_launched: bool = field(default=False, init=False, repr=False)
    _visualization_linger_s: float = field(default=0.0, init=False, repr=False)
    _visualization_linger_done: bool = field(default=False, init=False, repr=False)
    _visualization_finish_gate: object | None = field(default=None, init=False, repr=False)
    _visual_artifacts_error: str | None = field(default=None, init=False, repr=False)

    @classmethod
    def from_resolved(
        cls,
        sample,
        resolved,
        resources,
        *,
        invocation_context: GameArenaInvocationContext | None = None,
    ):
        (
            invocation_context,
            sample_action_router,
            sample_action_server,
        ) = _prepare_human_action_routing(
            sample=sample,
            resolved=resolved,
            invocation_context=invocation_context,
        )
        player_specs = _bind_players(resolved=resolved, invocation_context=invocation_context)
        environment = _build_environment(
            sample=sample,
            resolved=resolved,
            resources=resources,
            player_specs=player_specs,
            invocation_context=invocation_context,
        )
        session = cls(
            sample=sample,
            environment=environment,
            player_specs=player_specs,
            observation_workflow=resolved.observation_workflow,
            support_workflow=getattr(resolved, "support_workflow", None),
            visualization_spec=getattr(resolved, "visualization_spec", None),
            max_steps=_resolve_max_steps(sample=sample, resolved=resolved),
            invocation_context=invocation_context,
            visual_recorder=_build_visual_recorder(
                sample=sample,
                resolved=resolved,
                invocation_context=invocation_context,
            ),
        )
        session._sample_action_router = sample_action_router
        session._sample_action_server = sample_action_server
        session._initialize_visualization()
        return session

    def should_stop(self) -> bool:
        if self.final_result is not None:
            return True
        if self.environment is None:
            return True
        if self.step >= self.max_steps:
            self.final_result = self._build_result(result="max_steps", reason="max_steps")
            return True
        return bool(self.environment.is_terminal())

    def observe(self):
        if self.environment is None:
            raise RuntimeError("Game session environment is not initialized")
        player_id = str(self.environment.get_active_player())
        observation = self.environment.observe(player_id)
        workflow = self.observation_workflow
        if workflow is not None:
            observation = workflow.build(observation, self)
        support_context = self._run_support_hook(
            SupportHook.AFTER_OBSERVE,
            {"observation": observation, "player_id": player_id},
        )
        observation = support_context.payload.get("observation", observation)
        self._last_observation = observation
        self._current_trace_entry = make_trace_entry(
            step_index=self.step,
            player_id=player_id,
            timestamp_ms=wall_clock_ms(),
            t_obs_ready_ms=wall_clock_ms(),
        )
        self._record_visual_decision_window_open(player_id=player_id, observation=observation)
        return observation

    def decide_current_player(self, observation) -> ArenaAction:
        if self.environment is None:
            raise RuntimeError("Game session environment is not initialized")
        player_id = str(self.environment.get_active_player())
        before_context = self._run_support_hook(
            SupportHook.BEFORE_DECIDE,
            {"observation": observation, "player_id": player_id},
        )
        observation = before_context.payload.get("observation", observation)
        player = self._require_player(player_id)
        action = player.next_action(observation)
        after_context = self._run_support_hook(
            SupportHook.AFTER_DECIDE,
            {
                "observation": observation,
                "player_id": player_id,
                "action": action.move,
                "action_object": action,
            },
        )
        updated_action = after_context.payload.get("action", action.move)
        resolved_action = self._coerce_action(action, updated_action)
        trace_entry = self._ensure_trace_entry(player_id)
        set_trace_action_fields(trace_entry, resolved_action, action_format="flat")
        trace_entry["t_action_submitted_ms"] = wall_clock_ms()
        trace_entry["retry_count"] = infer_retry_count(resolved_action)
        trace_entry["is_action_legal"] = infer_legality(resolved_action)
        self._record_visual_action_intent(
            player_id=player_id,
            action=resolved_action,
            observation=observation,
        )
        return resolved_action

    def apply(self, action: ArenaAction) -> None:
        if self.environment is None:
            raise RuntimeError("Game session environment is not initialized")
        before_context = self._run_support_hook(
            SupportHook.BEFORE_APPLY,
            {
                "player_id": action.player,
                "action": action.move,
                "action_object": action,
            },
        )
        action = self._coerce_action(action, before_context.payload.get("action", action.move))
        trace_entry = self._ensure_trace_entry(action.player)
        trace_entry["player_id"] = action.player
        set_trace_action_fields(trace_entry, action, action_format="flat")
        trace_entry["t_action_submitted_ms"] = wall_clock_ms()
        trace_entry["retry_count"] = infer_retry_count(action)
        trace_entry["is_action_legal"] = infer_legality(action)
        result = self.environment.apply(action)
        if result is not None:
            self.final_result = result
        after_context = self._run_support_hook(
            SupportHook.AFTER_APPLY,
            {
                "player_id": action.player,
                "action": action.move,
                "action_object": action,
                "result": result,
            },
        )
        updated_result = after_context.payload.get("result", result)
        if updated_result is not None:
            self.final_result = updated_result
        illegal_reason = self._resolve_illegal_reason(updated_result)
        trace_entry["illegal_reason"] = illegal_reason
        if illegal_reason is not None:
            trace_entry["is_action_legal"] = False
        finalized_trace_entry = finalize_trace_entry(trace_entry)
        self._record_visual_action_committed(
            action=action,
            trace_entry=finalized_trace_entry,
            result=updated_result,
        )
        self._record_visual_decision_window_close(player_id=action.player)
        self.arena_trace.append(finalized_trace_entry)
        self._current_trace_entry = None

    def advance(self) -> None:
        delta = self._resolve_progress_delta()
        self.tick += delta
        self.step += delta
        self._record_visual_snapshot()
        if self.final_result is None and self.environment is not None and self.environment.is_terminal():
            self.final_result = self._build_result(result="completed", reason="completed")

    def capture_output_tick(self) -> None:
        """Hook for schedulers that capture per-tick output on the v2 path."""
        return None

    def ensure_result(self) -> object | None:
        if self.final_result is not None:
            return self.final_result
        if self.environment is None:
            return None
        if self.environment.is_terminal():
            self.final_result = self._build_result(result="completed", reason="completed")
        return self.final_result

    def finalize(self) -> object | None:
        if self._finalized:
            return self.final_result
        self.ensure_result()
        final_context = self._run_support_hook(
            SupportHook.ON_FINALIZE,
            {"result": self.final_result, "sample": self.sample},
        )
        self.final_result = final_context.payload.get("result", self.final_result)
        self._record_visual_result()
        self._persist_visual_recorder()
        self._maybe_launch_visualization_browser()
        self._clear_sample_action_routing()
        self._finalized = True
        self._linger_for_visualization()
        return self.final_result

    def _initialize_visualization(self) -> None:
        if self.environment is None:
            return
        invocation = self.invocation_context
        if invocation is None:
            return
        visualizer_config = dict(invocation.visualizer_config or {})
        if not _is_visualizer_enabled(visualizer_config):
            return
        service_hub = invocation.runtime_service_hub
        if service_hub is None:
            return
        self._visualization_mode = _resolve_visualizer_mode(visualizer_config)
        self._visualization_linger_s = _resolve_linger_seconds(visualizer_config)
        self._visualization_launch_browser = bool(visualizer_config.get("launch_browser", False))
        if self._visualization_mode == "arena_visual":
            from gage_eval.role.arena.visualization.live_session import ArenaVisualFinishGate

            self._visualization_finish_gate = ArenaVisualFinishGate(
                idle_timeout_s=self._visualization_linger_s
            )
            visual_server = service_hub.ensure_visualizer(
                lambda: _build_arena_visual_server(visualizer_config, service_hub=service_hub)
            )
            start = getattr(visual_server, "start", None)
            if callable(start):
                start()
            session_id = _resolve_visualization_session_id(self)
            run_id = _resolve_invocation_run_id(invocation)
            self._visualization_run_id = run_id
            build_viewer_url = getattr(visual_server, "build_viewer_url", None)
            if callable(build_viewer_url) and session_id is not None:
                self._visualization_viewer_url = build_viewer_url(session_id, run_id=run_id)
            register_live_session = getattr(visual_server, "register_live_session", None)
            if callable(register_live_session) and self.visual_recorder is not None and session_id is not None:
                from gage_eval.role.arena.visualization.live_session import RecorderLiveSessionSource

                register_live_session(
                    RecorderLiveSessionSource(
                        recorder=self.visual_recorder,
                        run_id=run_id,
                        visualization_spec=self.visualization_spec,
                        live_scene_scheme=_resolve_live_scene_scheme(visualizer_config),
                        finish_gate=self._visualization_finish_gate,
                    )
                )
            logger.info(
                "Arena visual workspace ready session_id={} viewer_url={}",
                session_id,
                self._visualization_viewer_url or "",
            )
            self._maybe_launch_visualization_browser()
            return
        frame_source = _resolve_frame_source(self.environment)
        if frame_source is None:
            return
        ws_hub = service_hub.ensure_ws_rgb_hub(
            lambda: _build_ws_rgb_hub(visualizer_config)
        )
        start = getattr(ws_hub, "start", None)
        if callable(start):
            start()
        display_id = _build_display_id(self.sample, invocation=invocation)
        from gage_eval.tools.ws_rgb_server import DisplayRegistration

        registration = DisplayRegistration(
            display_id=display_id,
            label=_build_display_label(self.sample, visualizer_config=visualizer_config),
            human_player_id=_resolve_human_player_id(
                self.player_specs,
                environment=self.environment,
            ),
            frame_source=frame_source,
        )
        service_hub.register_display(
            display_id=display_id,
            hub=ws_hub,
            registration=registration,
        )
        self._visualization_display_id = display_id
        logger.info(
            "Arena live viewer ready display_id={} viewer_url={}",
            display_id,
            getattr(ws_hub, "viewer_url", ""),
        )
        _maybe_open_browser(
            getattr(ws_hub, "viewer_url", ""),
            enabled=bool(visualizer_config.get("launch_browser", False)),
        )

    def _linger_for_visualization(self) -> None:
        if self._visualization_linger_done:
            return
        linger_s = max(0.0, float(self._visualization_linger_s))
        if linger_s <= 0.0:
            return
        self._visualization_linger_done = True
        if self._visualization_mode == "arena_visual" and self._visualization_finish_gate is not None:
            arm = getattr(self._visualization_finish_gate, "arm", None)
            wait = getattr(self._visualization_finish_gate, "wait", None)
            if callable(arm) and callable(wait):
                arm()
                wait()
                return
        time.sleep(linger_s)

    def _require_player(self, player_id: str) -> BoundPlayer:
        for player in self.player_specs:
            if player.player_id == player_id:
                return player
        available = ", ".join(player.player_id for player in self.player_specs) or "none"
        raise KeyError(f"Unknown active player '{player_id}'. Available players: {available}")

    def _build_result(self, *, result: str, reason: str | None) -> object | None:
        if self.environment is None:
            return None
        builder = getattr(self.environment, "build_result", None)
        if callable(builder):
            return builder(result=result, reason=reason)
        return {
            "winner": None,
            "result": result,
            "reason": reason,
            "move_count": self.step,
        }

    def _run_support_hook(self, hook: SupportHook, payload: dict[str, object]) -> SupportContext:
        context = SupportContext(
            payload=payload,
            state={
                "session": self,
                "hook": hook,
                "sample": self.sample,
                "environment": self.environment,
                "tick": self.tick,
                "step": self.step,
            },
        )
        workflow = self.support_workflow
        if workflow is None:
            return context
        runner = getattr(workflow, "run", None)
        if not callable(runner):
            return context
        result = runner(hook, context)
        if isinstance(result, SupportContext):
            return result
        return context

    @staticmethod
    def _coerce_action(action: ArenaAction, payload_action: object) -> ArenaAction:
        if payload_action is action.move:
            return action
        if isinstance(payload_action, ArenaAction):
            return payload_action
        if isinstance(payload_action, Mapping):
            player = str(payload_action.get("player", action.player))
            move = payload_action.get("move", action.move)
            raw = payload_action.get("raw", action.raw)
            metadata = payload_action.get("metadata", action.metadata)
            return ArenaAction(
                player=player,
                move=move,  # type: ignore[arg-type]
                raw=raw,  # type: ignore[arg-type]
                metadata=dict(metadata) if isinstance(metadata, Mapping) else action.metadata,
            )
        return ArenaAction(
            player=action.player,
            move=payload_action,  # type: ignore[arg-type]
            raw=action.raw,
            metadata=action.metadata,
        )

    def _resolve_progress_delta(self) -> int:
        if self.environment is None:
            return 1
        resolver = getattr(self.environment, "consume_session_progress_delta", None)
        if not callable(resolver):
            return 1
        try:
            delta = int(resolver())
        except Exception:
            return 1
        return max(0, delta)

    def _ensure_trace_entry(self, player_id: str) -> dict[str, Any]:
        if self._current_trace_entry is None:
            now_ms = wall_clock_ms()
            self._current_trace_entry = make_trace_entry(
                step_index=self.step,
                player_id=player_id,
                timestamp_ms=now_ms,
                t_obs_ready_ms=now_ms,
            )
        return self._current_trace_entry

    @staticmethod
    def _resolve_illegal_reason(result: object | None) -> str | None:
        if result is None:
            return None
        if isinstance(result, Mapping):
            reason = result.get("reason")
            if reason in {
                "invalid_format",
                "illegal_move",
                "unknown_player",
                "wrong_player",
                "out_of_bounds",
                "occupied",
                "illegal_action",
            }:
                return str(reason)
            return None
        try:
            return detect_illegal_reason(result)
        except AttributeError:
            return None

    def _record_visual_decision_window_open(self, *, player_id: str, observation: object) -> None:
        recorder = self.visual_recorder
        if recorder is None:
            return
        recorder.record_decision_window_open(
            ts_ms=wall_clock_ms(),
            step=self.step,
            tick=self.tick,
            player_id=player_id,
            observation=_visual_payload_snapshot(observation),
        )

    def _record_visual_action_committed(
        self,
        *,
        action: ArenaAction,
        trace_entry: dict[str, Any],
        result: object | None,
    ) -> None:
        recorder = self.visual_recorder
        if recorder is None:
            return
        recorder.record_action_committed(
            ts_ms=wall_clock_ms(),
            step=self.step,
            tick=self.tick,
            player_id=action.player,
            action=action,
            trace_entry=trace_entry,
            result=result,
        )

    def _record_visual_action_intent(
        self,
        *,
        player_id: str,
        action: ArenaAction,
        observation: object,
    ) -> None:
        recorder = self.visual_recorder
        if recorder is None:
            return
        recorder.record_action_intent(
            ts_ms=wall_clock_ms(),
            step=self.step,
            tick=self.tick,
            player_id=player_id,
            action=action,
            observation=_visual_payload_snapshot(observation),
        )

    def _record_visual_decision_window_close(self, *, player_id: str) -> None:
        recorder = self.visual_recorder
        if recorder is None:
            return
        recorder.record_decision_window_close(
            ts_ms=wall_clock_ms(),
            step=self.step,
            tick=self.tick,
            player_id=player_id,
        )

    def _record_visual_snapshot(self) -> None:
        recorder = self.visual_recorder
        if recorder is None:
            return
        recorder.record_snapshot(
            ts_ms=wall_clock_ms(),
            step=self.step,
            tick=self.tick,
            snapshot={
                "step": self.step,
                "tick": self.tick,
                "playerId": self._current_trace_entry.get("player_id") if self._current_trace_entry else None,
                "observation": _visual_payload_snapshot(self._last_observation),
                "arenaTrace": _visual_payload_snapshot(self.arena_trace[-1]) if self.arena_trace else None,
                "result": _visual_payload_snapshot(self.final_result),
            },
        )

    def _record_visual_result(self) -> None:
        recorder = self.visual_recorder
        if recorder is None or self.final_result is None:
            return
        recorder.record_result(
            ts_ms=wall_clock_ms(),
            step=self.step,
            tick=self.tick,
            result=self.final_result,
        )

    def _persist_visual_recorder(self) -> None:
        recorder = self.visual_recorder
        if recorder is None:
            return
        replay_path = _resolve_visual_replay_path(
            result=self.final_result,
            invocation_context=self.invocation_context,
        )
        if replay_path is None:
            return
        try:
            artifacts = recorder.persist(replay_path)
            replay_manifest_path = Path(replay_path)
            if replay_manifest_path.exists():
                updated = update_replay_manifest_visual_session_ref(
                    replay_path=replay_path,
                    visual_session_ref=artifacts.visual_session_ref,
                )
                if not updated:
                    self._visual_artifacts_error = "replay_manifest_update_failed"
                    logger.warning(
                        "Arena visual replay manifest update failed for replay_path={}",
                        replay_path,
                    )
            else:
                logger.debug(
                    "Arena visual replay manifest not yet present for replay_path={}, skipping visual_session_ref wiring",
                    replay_path,
                )
        except Exception as exc:  # pragma: no cover - defensive sidecar guard
            self._visual_artifacts_error = str(exc)
            logger.warning(
                "Arena visual sidecar persistence failed for replay_path={}: {}",
                replay_path,
                exc,
            )

    def _maybe_launch_visualization_browser(self) -> None:
        if self._visualization_mode != "arena_visual":
            return
        if self._visualization_browser_launched:
            return
        viewer_url = str(self._visualization_viewer_url or "").strip()
        if not viewer_url:
            return
        self._visualization_browser_launched = True
        _maybe_open_browser(viewer_url, enabled=self._visualization_launch_browser)

    def _clear_sample_action_routing(self) -> None:
        router = self._sample_action_router
        if router is None:
            return
        invocation = self.invocation_context
        service_hub = None if invocation is None else invocation.runtime_service_hub
        sample_id = _resolve_session_sample_id(invocation)
        if service_hub is not None and sample_id is not None:
            action_server = self._sample_action_server
            if action_server is None:
                action_server = getattr(service_hub, "peek_action_server", lambda: None)()
            clear_routes = getattr(service_hub, "clear_sample_routes", None)
            if callable(clear_routes):
                clear_routes(
                    sample_id=sample_id,
                    action_server=action_server,
                    visualizer=None,
                )
            elif action_server is not None:
                unregister = getattr(action_server, "unregister_action_queue", None)
                if callable(unregister):
                    unregister(sample_id)
        self._sample_action_router = None
        self._sample_action_server = None


def _resolve_max_steps(*, sample: ArenaSample, resolved) -> int:
    runtime_overrides = sample.runtime_overrides or {}
    for key in ("max_steps", "max_turns", "max_ticks"):
        value = runtime_overrides.get(key)
        if value is not None:
            return max(1, int(value))
    defaults = getattr(resolved.scheduler, "defaults", {}) or {}
    for key in ("max_steps", "max_turns", "max_ticks"):
        value = defaults.get(key)
        if value is not None:
            return max(1, int(value))
    return 256


def _build_visual_recorder(
    *,
    sample: ArenaSample,
    resolved,
    invocation_context: GameArenaInvocationContext | None,
) -> ArenaVisualSessionRecorder | None:
    visualization_spec = getattr(resolved, "visualization_spec", None)
    plugin_id = getattr(visualization_spec, "plugin_id", None)
    if not plugin_id:
        plugin_id = invocation_context.adapter_id if invocation_context is not None else None
    if not plugin_id:
        plugin_id = sample.game_kit
    game_id = sample.game_kit or getattr(visualization_spec, "game_id", None) or sample.env
    session_id = None
    if invocation_context is not None:
        session_id = invocation_context.sample_id
        if session_id is None and invocation_context.trace is not None:
            session_id = getattr(invocation_context.trace, "sample_id", None)
    if not session_id:
        session_id = f"{sample.game_kit}:{sample.env or 'default'}"
    return ArenaVisualSessionRecorder(
        plugin_id=str(plugin_id or "arena"),
        game_id=str(game_id or sample.game_kit or "arena"),
        scheduling_family=_resolve_scheduler_family(sample=sample, resolved=resolved),
        session_id=str(session_id),
        observer_modes=_resolve_visual_observer_modes(visualization_spec),
        visual_kind=_resolve_visual_kind(visualization_spec),
    )


def _resolve_visual_observer_modes(visualization_spec: Any) -> tuple[str, ...]:
    observer_schema = getattr(visualization_spec, "observer_schema", {}) or {}
    supported_modes = observer_schema.get("supported_modes")
    if not isinstance(supported_modes, (list, tuple)):
        return ()
    return tuple(str(mode) for mode in supported_modes if str(mode).strip())


def _resolve_visual_kind(visualization_spec: Any) -> str | None:
    visual_kind = getattr(visualization_spec, "visual_kind", None)
    if visual_kind is None:
        return None
    normalized = str(visual_kind).strip()
    return normalized or None


def _resolve_scheduler_family(*, sample: ArenaSample, resolved) -> str:
    scheduler = getattr(resolved, "scheduler", None)
    family = getattr(scheduler, "family", None)
    if family:
        return str(family)
    scheduler_binding = str(sample.scheduler or "").strip()
    if scheduler_binding:
        return scheduler_binding.split("/", 1)[0]
    return "turn"


def _build_environment(
    *,
    sample: ArenaSample,
    resolved,
    resources,
    player_specs,
    invocation_context: GameArenaInvocationContext | None,
) -> object:
    env_factory = resolved.env_spec.defaults.get("env_factory")
    if not callable(env_factory):
        raise KeyError(
            f"Env '{resolved.env_spec.env_id}' for game kit '{resolved.game_kit.kit_id}' "
            "does not define a callable 'env_factory'"
        )
    kwargs = {
        "sample": sample,
        "resolved": resolved,
        "resources": resources,
        "player_specs": player_specs,
    }
    if _accepts_keyword(env_factory, "invocation_context"):
        kwargs["invocation_context"] = invocation_context
    return env_factory(**kwargs)


def _bind_players(
    *,
    resolved,
    invocation_context: GameArenaInvocationContext | None,
) -> tuple[BoundPlayer, ...]:
    bindings = tuple(getattr(resolved, "player_bindings", ()) or ())
    if not bindings:
        return ()
    registry = getattr(resolved, "player_driver_registry", None)
    if registry is None:
        raise RuntimeError("resolved runtime binding is missing player_driver_registry")
    return tuple(registry.bind_all(bindings, invocation=invocation_context))


def _prepare_human_action_routing(
    *,
    sample: ArenaSample,
    resolved,
    invocation_context: GameArenaInvocationContext | None,
) -> tuple[GameArenaInvocationContext | None, SampleActionRouter | None, Any | None]:
    del sample
    if invocation_context is None:
        return None, None, None
    if not _is_human_input_routing_enabled(invocation_context):
        return invocation_context, None, None
    service_hub = invocation_context.runtime_service_hub
    if service_hub is None:
        return invocation_context, None, None

    sample_id = _resolve_session_sample_id(invocation_context)
    human_player_ids = _collect_human_player_ids(resolved)
    if sample_id is None or not human_player_ids:
        return invocation_context, None, None

    action_router = SampleActionRouter(sample_id=sample_id, player_ids=human_player_ids)
    action_server = _ensure_action_server(service_hub, invocation_context=invocation_context)
    if action_server is not None:
        bind_sample_routes = getattr(service_hub, "bind_sample_routes", None)
        if callable(bind_sample_routes):
            bind_sample_routes(
                sample_id=sample_id,
                action_server=action_server,
                action_router=action_router,
                visualizer=None,
            )
        elif hasattr(action_server, "register_action_queue"):
            action_server.register_action_queue(sample_id, action_router)

    merged_action_queues = dict(invocation_context.player_action_queues or {})
    merged_action_queues.update(action_router.player_queues)
    updated_invocation_context = replace(
        invocation_context,
        player_action_queues=merged_action_queues,
    )
    return updated_invocation_context, action_router, action_server


def _collect_human_player_ids(resolved) -> tuple[str, ...]:
    bindings = tuple(getattr(resolved, "player_bindings", ()) or ())
    human_player_ids = [
        str(binding.player_id)
        for binding in bindings
        if str(getattr(binding, "player_kind", "") or "").strip() == "human"
        and str(getattr(binding, "player_id", "") or "").strip()
    ]
    return tuple(human_player_ids)


def _ensure_action_server(
    service_hub: Any,
    *,
    invocation_context: GameArenaInvocationContext,
) -> Any | None:
    ensure_action_server = getattr(service_hub, "ensure_action_server", None)
    if not callable(ensure_action_server):
        return None
    return ensure_action_server(lambda: _build_action_server(invocation_context))


def _build_action_server(invocation_context: GameArenaInvocationContext) -> Any:
    from gage_eval.tools.action_server import ActionQueueServer

    config = dict(invocation_context.human_input_config or {})
    server = ActionQueueServer(
        host=str(config.get("host") or "127.0.0.1"),
        port=int(config.get("port") or 8001),
        allow_origin=str(config.get("allow_origin") or "*"),
    )
    start = getattr(server, "start", None)
    if callable(start):
        start()
    return server


def _is_human_input_routing_enabled(invocation_context: GameArenaInvocationContext) -> bool:
    config = dict(invocation_context.human_input_config or {})
    if not config:
        return False
    enabled = config.get("enabled")
    if enabled is None:
        return True
    return bool(enabled)


def _resolve_session_sample_id(
    invocation_context: GameArenaInvocationContext | None,
) -> str | None:
    if invocation_context is None:
        return None
    sample_id = invocation_context.sample_id
    if sample_id:
        return str(sample_id)
    trace = invocation_context.trace
    if trace is not None:
        trace_sample_id = getattr(trace, "sample_id", None)
        if trace_sample_id:
            return str(trace_sample_id)
    return None


def _resolve_invocation_run_id(
    invocation_context: GameArenaInvocationContext | None,
) -> str | None:
    if invocation_context is None:
        return None
    trace = invocation_context.trace
    if trace is None:
        return None
    run_id = getattr(trace, "run_id", None)
    if run_id is None:
        return None
    text = str(run_id).strip()
    return text or None


def _resolve_visualization_session_id(session: GameSession) -> str | None:
    recorder = session.visual_recorder
    if recorder is not None and str(recorder.session_id).strip():
        return str(recorder.session_id)
    return _resolve_session_sample_id(session.invocation_context)


def _accepts_keyword(callable_obj: object, keyword: str) -> bool:
    try:
        signature = inspect.signature(callable_obj)
    except (TypeError, ValueError):
        return True
    if keyword in signature.parameters:
        return True
    return any(
        parameter.kind is inspect.Parameter.VAR_KEYWORD
        for parameter in signature.parameters.values()
    )


def _is_visualizer_enabled(config: Mapping[str, Any]) -> bool:
    return bool(config.get("enabled", False))


def _resolve_visualizer_mode(config: Mapping[str, Any]) -> str:
    mode = str(config.get("mode") or "").strip().lower()
    if mode == "arena_visual":
        return "arena_visual"
    return "ws_rgb"


def _resolve_live_scene_scheme(config: Mapping[str, Any]) -> str:
    scheme = str(config.get("live_scene_scheme") or "").strip().lower()
    return scheme or "http_pull"


def _resolve_result_replay_path(result: object | None) -> str | None:
    if result is None:
        return None
    if isinstance(result, Mapping):
        replay_path = result.get("replay_path")
        return str(replay_path) if replay_path not in (None, "") else None
    replay_path = getattr(result, "replay_path", None)
    if replay_path in (None, ""):
        return None
    return str(replay_path)


def _resolve_visual_replay_path(
    *,
    result: object | None,
    invocation_context: GameArenaInvocationContext | None,
) -> str | None:
    replay_path = _resolve_result_replay_path(result)
    if replay_path is not None:
        return replay_path
    if invocation_context is None:
        return None
    run_id = _resolve_invocation_run_id(invocation_context)
    sample_id = _resolve_session_sample_id(invocation_context)
    if run_id is None or sample_id is None:
        return None
    resolved = resolve_replay_manifest_path(
        run_id=run_id,
        sample_id=sample_id,
        base_dir=os.environ.get("GAGE_EVAL_SAVE_DIR"),
    )
    if resolved is None:
        return None
    return str(resolved)


def _build_ws_rgb_hub(config: Mapping[str, Any]):
    from gage_eval.tools.ws_rgb_server import WsRgbHubServer

    hub = WsRgbHubServer(
        host=str(config.get("host") or "127.0.0.1"),
        port=int(config.get("port") or 5800),
        allow_origin=str(config.get("allow_origin") or "*"),
    )
    hub.start()
    return hub


def _build_arena_visual_server(
    config: Mapping[str, Any],
    *,
    service_hub: Any,
) -> Any:
    from gage_eval.role.arena.visualization.http_server import ArenaVisualHTTPServer

    return ArenaVisualHTTPServer(
        host=str(config.get("host") or "127.0.0.1"),
        port=int(config.get("port") or 5800),
        base_dir=str(config.get("base_dir") or os.environ.get("GAGE_EVAL_SAVE_DIR") or "./runs"),
        action_submitter=getattr(service_hub, "submit_action_intent", None),
        chat_submitter=getattr(service_hub, "submit_chat_message", None),
        control_submitter=getattr(service_hub, "submit_control_command", None),
        allow_origin=str(config.get("allow_origin") or "*"),
    )


def _resolve_frame_source(environment: object) -> object | None:
    getter = getattr(environment, "get_last_frame", None)
    if not callable(getter):
        return None

    def _frame_source() -> dict[str, Any]:
        return _normalize_frame_payload(getter())

    return _frame_source


def _normalize_frame_payload(payload: object) -> dict[str, Any]:
    if isinstance(payload, Mapping):
        return dict(payload)
    if payload is None:
        return {}
    return {"board_text": str(payload)}


def _visual_payload_snapshot(payload: object | None) -> object | None:
    return to_visual_json_safe(payload)


def _build_display_id(
    sample: ArenaSample,
    *,
    invocation: GameArenaInvocationContext,
) -> str:
    adapter_id = str(invocation.adapter_id or "arena")
    sample_id = str(invocation.sample_id or "sample")
    env_id = str(sample.env or sample.game_kit or "arena")
    return f"{adapter_id}:{sample_id}:{env_id}"


def _build_display_label(
    sample: ArenaSample,
    *,
    visualizer_config: Mapping[str, Any],
) -> str:
    title = str(visualizer_config.get("title") or "").strip()
    if title:
        return title
    return f"{sample.game_kit}:{sample.env or 'default'}"


def _resolve_human_player_id(
    player_specs: tuple[BoundPlayer, ...],
    *,
    environment: object,
) -> str:
    if player_specs:
        return str(player_specs[0].player_id)
    getter = getattr(environment, "get_active_player", None)
    if callable(getter):
        try:
            return str(getter())
        except Exception:
            return "player_0"
    return "player_0"


def _resolve_linger_seconds(config: Mapping[str, Any]) -> float:
    try:
        return max(0.0, float(config.get("linger_after_finish_s", 0.0) or 0.0))
    except (TypeError, ValueError):
        return 0.0


def _maybe_open_browser(viewer_url: str, *, enabled: bool) -> None:
    if not enabled or not viewer_url:
        return
    try:
        opened = bool(webbrowser.open(viewer_url))
        if not opened:
            opened = _open_browser_fallback(viewer_url)
        logger.info(
            "Arena live viewer browser_open viewer_url={} opened={}",
            viewer_url,
            opened,
        )
    except Exception as exc:
        logger.warning(
            "Arena live viewer browser_open_failed viewer_url={} error={}",
            viewer_url,
            exc,
        )
        return


def _open_browser_fallback(viewer_url: str) -> bool:
    commands: list[list[str]] = []
    if sys.platform == "darwin":
        commands.append(["open", viewer_url])
    elif sys.platform.startswith("linux"):
        commands.append(["xdg-open", viewer_url])
    elif sys.platform.startswith("win"):
        commands.append(["cmd", "/c", "start", "", viewer_url])
    for command in commands:
        if shutil.which(command[0]) is None:
            continue
        try:
            subprocess.Popen(command)  # noqa: S603
            return True
        except Exception:
            continue
    return False
