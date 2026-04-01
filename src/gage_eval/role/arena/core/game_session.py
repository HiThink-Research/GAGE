from __future__ import annotations

import base64
import inspect
import io
import os
import shutil
import subprocess
import sys
import time
import webbrowser
from dataclasses import dataclass, field, replace
from pathlib import Path
from threading import Lock
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
from gage_eval.role.arena.types import ArenaAction, GameResult
from gage_eval.role.arena.visualization.recorder import ArenaVisualSessionRecorder
from gage_eval.role.arena.visualization.artifacts import to_scene_json_safe
from gage_eval.role.arena.replay_paths import resolve_replay_manifest_path
from gage_eval.role.arena.replay_schema_writer import (
    ReplaySchemaWriter,
    update_replay_manifest_visual_session_ref,
)

try:
    from PIL import Image
except Exception:  # pragma: no cover - optional dependency
    Image = None

if TYPE_CHECKING:
    from gage_eval.tools.ws_rgb_server import DisplayRegistration, WsRgbHubServer

_LIVE_MEDIA_INLINE_SCHEMES = {"binary_stream", "low_latency_channel"}
_REALTIME_INLINE_SNAPSHOT_TARGET_MS = 250


@dataclass
class GameSession:
    sample: ArenaSample
    environment: object | None = None
    player_specs: tuple[BoundPlayer, ...] = ()
    runtime_profile: object | None = None
    observation_workflow: object = None
    support_workflow: object = None
    visualization_spec: object | None = None
    resources: object | None = None
    max_steps: int = 256
    final_result: object | None = None
    tick: int = 0
    step: int = 0
    arena_trace: list[dict[str, object]] = field(default_factory=list)
    support_errors: list[dict[str, object]] = field(default_factory=list)
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
    _visualization_live_scene_scheme: str = field(default="http_pull", init=False, repr=False)
    _visualization_linger_done: bool = field(default=False, init=False, repr=False)
    _visualization_finish_gate: object | None = field(default=None, init=False, repr=False)
    _visual_artifacts_error: str | None = field(default=None, init=False, repr=False)
    _stop_requested: bool = field(default=False, init=False, repr=False)
    _stop_reason: str | None = field(default=None, init=False, repr=False)
    _restart_requested: bool = field(default=False, init=False, repr=False)
    _latest_live_frame_payload: Mapping[str, Any] | None = field(default=None, init=False, repr=False)
    _latest_live_frame_ts_ms: int | None = field(default=None, init=False, repr=False)
    _latest_live_frame_version: int = field(default=0, init=False, repr=False)
    _last_realtime_snapshot_ts_ms: int = field(default=0, init=False, repr=False)
    _last_input_age_ms: float | None = field(default=None, init=False, repr=False)
    _stop_lock: Lock = field(default_factory=Lock, init=False, repr=False)

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
            runtime_profile=getattr(resolved, "runtime_profile", None),
            observation_workflow=resolved.observation_workflow,
            support_workflow=getattr(resolved, "support_workflow", None),
            visualization_spec=getattr(resolved, "visualization_spec", None),
            resources=resources,
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
        session._configure_visual_capabilities()
        return session

    def should_stop(self) -> bool:
        self._consume_requested_restart()
        if self._stop_requested:
            if self.final_result is None:
                self.final_result = self._build_result(
                    result="terminated",
                    reason=self._stop_reason or "user_finish",
                )
            return True
        if self.final_result is not None:
            return True
        if self.environment is None:
            return True
        if self.step >= self.max_steps:
            self.final_result = self._build_result(result="max_steps", reason="max_steps")
            return True
        return bool(self.environment.is_terminal())

    def request_stop(
        self,
        *,
        reason: str = "user_finish",
        result: str = "terminated",
    ) -> object | None:
        with self._stop_lock:
            self._stop_requested = True
            self._stop_reason = str(reason)
            if self.final_result is None:
                self.final_result = self._build_result(result=result, reason=reason)
            environment = self.environment
        _best_effort_stop_environment(environment)
        return self.final_result

    def request_restart(self) -> int:
        with self._stop_lock:
            self._restart_requested = True
        recorder = self.visual_recorder
        if recorder is None:
            return 0
        return recorder.build_visual_session().playback.cursor_event_seq

    def observe(self):
        self._consume_requested_restart()
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
        if self._uses_scheduler_owned_human_realtime():
            self._set_visual_scheduling_state(
                phase="waiting_for_intent",
                player_id=player_id,
                accepts_human_intent=self._is_human_player(player_id),
            )
        else:
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
        self._last_input_age_ms = _resolve_input_age_ms(resolved_action)
        if not self._uses_scheduler_owned_human_realtime():
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
        if self._uses_scheduler_owned_human_realtime():
            self._set_visual_scheduling_state(
                phase="advancing",
                player_id=action.player,
                accepts_human_intent=self._is_human_player(action.player),
            )
        else:
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
        if not self._uses_scheduler_owned_human_realtime():
            self._record_visual_snapshot()
        if self.final_result is None and self.environment is not None and self.environment.is_terminal():
            self.final_result = self._build_result(result="completed", reason="completed")

    def capture_output_tick(self) -> None:
        """Hook for schedulers that capture per-tick output on the v2 path."""
        if not self._uses_scheduler_owned_human_realtime():
            return None
        frame_payload = self._capture_realtime_live_frame()
        if self._should_record_realtime_snapshot():
            self._capture_realtime_snapshot(frame_payload=frame_payload)
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
        self._materialize_replay_artifact_if_missing()
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
        self._visualization_live_scene_scheme = _resolve_live_scene_scheme(visualizer_config)
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

                self._refresh_live_frame_buffer()
                register_live_session(
                    RecorderLiveSessionSource(
                        recorder=self.visual_recorder,
                        run_id=run_id,
                        visualization_spec=self.visualization_spec,
                        live_scene_scheme=self._visualization_live_scene_scheme,
                        finish_gate=self._visualization_finish_gate,
                        live_frame_supplier=self.get_live_frame_payload,
                        stop_callback=self.request_stop,
                        restart_callback=self.request_restart,
                    )
                )
            logger.info(
                "Arena visual workspace ready session_id={} viewer_url={}",
                session_id,
                self._visualization_viewer_url or "",
            )
            self._bind_visualization_resource(
                mode="arena_visual",
                viewer_url=self._visualization_viewer_url,
                session_id=session_id,
                run_id=run_id,
                replay_viewer_close=(
                    None
                    if session_id is None
                    else (
                        lambda: _unregister_arena_visual_live_session(
                            visual_server,
                            session_id=session_id,
                            run_id=run_id,
                        )
                    )
                ),
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
        self._bind_visualization_resource(
            mode="ws_rgb",
            viewer_url=getattr(ws_hub, "viewer_url", ""),
            display_id=display_id,
            replay_viewer_close=lambda: _unregister_ws_rgb_display(
                ws_hub,
                display_id=display_id,
            ),
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
            errors = result.state.get("support_errors")
            if isinstance(errors, list):
                for error in errors:
                    if isinstance(error, Mapping) and error not in self.support_errors:
                        self.support_errors.append(dict(error))
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

    def _is_human_player(self, player_id: str) -> bool:
        return any(
            player.player_id == player_id and player.player_kind == "human"
            for player in self.player_specs
        )

    def _uses_scheduler_owned_human_realtime(self) -> bool:
        profile = self.runtime_profile
        if profile is None:
            return False
        uses_scheduler_owned = getattr(profile, "uses_scheduler_owned_human_realtime", None)
        if callable(uses_scheduler_owned):
            return bool(uses_scheduler_owned())
        return bool(getattr(profile, "scheduler_owns_realtime_clock", False))

    def _set_visual_scheduling_state(
        self,
        *,
        phase: str,
        player_id: str | None,
        accepts_human_intent: bool,
    ) -> None:
        recorder = self.visual_recorder
        if recorder is None:
            return
        set_state = getattr(recorder, "set_scheduling_state", None)
        if callable(set_state):
            set_state(
                phase=phase,
                accepts_human_intent=accepts_human_intent,
                active_actor_id=player_id,
            )

    def get_live_frame_payload(self) -> Mapping[str, Any] | None:
        return self._latest_live_frame_payload

    def _refresh_live_frame_buffer(self) -> Mapping[str, Any] | None:
        frame_payload = _load_environment_frame_payload(self.environment)
        if frame_payload is not None:
            self._latest_live_frame_version += 1
            ts_ms = wall_clock_ms()
            normalized_frame_payload = dict(frame_payload)
            normalized_frame_payload["_live_frame_version"] = self._latest_live_frame_version
            normalized_frame_payload["_live_frame_ts_ms"] = ts_ms
            self._latest_live_frame_payload = normalized_frame_payload
            self._latest_live_frame_ts_ms = ts_ms
            return normalized_frame_payload
        self._latest_live_frame_payload = None
        self._latest_live_frame_ts_ms = None
        return frame_payload

    def _capture_realtime_live_frame(self) -> Mapping[str, Any] | None:
        if self._should_refresh_live_frame_buffer():
            return self._refresh_live_frame_buffer()
        return self._latest_live_frame_payload

    def _should_refresh_live_frame_buffer(self) -> bool:
        if self.final_result is not None:
            return True
        if self._latest_live_frame_payload is None or self._latest_live_frame_ts_ms is None:
            return True
        interval_ms = self._resolve_realtime_frame_interval_ms()
        if interval_ms is None:
            return True
        return max(0, wall_clock_ms() - self._latest_live_frame_ts_ms) >= interval_ms

    def _should_record_realtime_snapshot(self) -> bool:
        recorder = self.visual_recorder
        if recorder is None:
            return False
        if self.final_result is not None:
            return True
        if self._last_realtime_snapshot_ts_ms <= 0:
            return True
        now_ms = wall_clock_ms()
        return now_ms - self._last_realtime_snapshot_ts_ms >= self._resolve_realtime_snapshot_interval_ms()

    def _resolve_realtime_frame_interval_ms(self) -> int | None:
        control_profile = self._realtime_human_control_profile()
        if control_profile is None:
            return None
        value = getattr(control_profile, "frame_output_hz", None)
        try:
            frame_output_hz = int(value)
        except (TypeError, ValueError):
            return None
        if frame_output_hz <= 0:
            return None
        return max(1, int(round(1000.0 / float(frame_output_hz))))

    def _resolve_realtime_snapshot_interval_ms(self) -> int:
        interval_ms = self._resolve_realtime_frame_interval_ms()
        if interval_ms is not None:
            return interval_ms
        return _REALTIME_INLINE_SNAPSHOT_TARGET_MS

    def _realtime_human_control_profile(self) -> object | None:
        runtime_profile = self.runtime_profile
        if runtime_profile is None:
            return None
        return getattr(runtime_profile, "realtime_human_control", None)

    def _uses_async_decimated_live_artifacts(self) -> bool:
        control_profile = self._realtime_human_control_profile()
        if control_profile is None:
            return False
        return (
            str(getattr(control_profile, "artifact_sampling_mode", "") or "").strip()
            == "async_decimated_live"
        )

    def _flush_pending_visual_snapshots(self) -> int:
        recorder = self.visual_recorder
        if recorder is None:
            return 0
        flush_pending = getattr(recorder, "flush_pending_snapshots", None)
        if not callable(flush_pending):
            return 0
        return int(flush_pending())

    def record_realtime_tick_metrics(
        self,
        *,
        tick_interval_ms: int,
        tick_elapsed_ms: float,
    ) -> None:
        recorder = self.visual_recorder
        if recorder is None:
            return
        update_runtime_metrics = getattr(recorder, "update_runtime_metrics", None)
        if not callable(update_runtime_metrics):
            return
        live_frame_age_ms = None
        if self._latest_live_frame_ts_ms is not None:
            live_frame_age_ms = max(0, wall_clock_ms() - self._latest_live_frame_ts_ms)
        pending_snapshot_count = getattr(recorder, "pending_snapshot_count", None)
        artifact_queue_depth = 0
        if callable(pending_snapshot_count):
            artifact_queue_depth = int(pending_snapshot_count())
        update_runtime_metrics(
            tick_overshoot_ms=max(0.0, float(tick_elapsed_ms) - float(tick_interval_ms)),
            input_age_ms=self._last_input_age_ms,
            live_frame_age_ms=live_frame_age_ms,
            artifact_queue_depth=artifact_queue_depth,
        )

    def _record_visual_decision_window_open(self, *, player_id: str, observation: object) -> None:
        recorder = self.visual_recorder
        if recorder is None:
            return
        player = next(
            (candidate for candidate in self.player_specs if candidate.player_id == player_id),
            None,
        )
        recorder.record_decision_window_open(
            ts_ms=wall_clock_ms(),
            step=self.step,
            tick=self.tick,
            player_id=player_id,
            observation=_visual_payload_snapshot(
                observation,
                include_inline_media=not self._should_strip_inline_media_from_live_trace(),
            ),
            accepts_human_intent=bool(player and player.player_kind == "human"),
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
            observation=_visual_payload_snapshot(
                observation,
                include_inline_media=not self._should_strip_inline_media_from_live_trace(),
            ),
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

    def _record_visual_snapshot(self, *, frame_payload: Mapping[str, Any] | None = None) -> None:
        recorder = self.visual_recorder
        if recorder is None:
            return
        include_inline_frame_image = self._should_include_inline_snapshot_media()
        sanitized_snapshot = {
            "step": self.step,
            "tick": self.tick,
            "playerId": self._current_trace_entry.get("player_id") if self._current_trace_entry else None,
            "observation": _build_visual_snapshot_observation(
                environment=self.environment,
                fallback_observation=self._last_observation,
                tick=self.tick,
                step=self.step,
                include_inline_frame_image=include_inline_frame_image,
                frame_payload=frame_payload,
            ),
            "arenaTrace": _visual_payload_snapshot(self.arena_trace[-1]) if self.arena_trace else None,
            "result": _visual_payload_snapshot(self.final_result),
        }
        ts_ms = wall_clock_ms()
        recorder.record_snapshot(
            ts_ms=ts_ms,
            step=self.step,
            tick=self.tick,
            snapshot=sanitized_snapshot,
            snapshot_is_scene_safe=True,
        )
        self._last_realtime_snapshot_ts_ms = ts_ms

    def _capture_realtime_snapshot(self, *, frame_payload: Mapping[str, Any] | None = None) -> None:
        recorder = self.visual_recorder
        if recorder is None:
            return
        include_inline_frame_image = self._should_include_inline_snapshot_media()
        sanitized_snapshot = {
            "step": self.step,
            "tick": self.tick,
            "playerId": self._current_trace_entry.get("player_id") if self._current_trace_entry else None,
            "observation": _build_visual_snapshot_observation(
                environment=self.environment,
                fallback_observation=self._last_observation,
                tick=self.tick,
                step=self.step,
                include_inline_frame_image=include_inline_frame_image,
                frame_payload=frame_payload,
            ),
            "arenaTrace": _visual_payload_snapshot(self.arena_trace[-1]) if self.arena_trace else None,
            "result": _visual_payload_snapshot(self.final_result),
        }
        ts_ms = wall_clock_ms()
        if self._uses_async_decimated_live_artifacts():
            enqueue_snapshot = getattr(recorder, "enqueue_snapshot", None)
            if callable(enqueue_snapshot):
                enqueue_snapshot(
                    ts_ms=ts_ms,
                    step=self.step,
                    tick=self.tick,
                    snapshot=sanitized_snapshot,
                    snapshot_is_scene_safe=True,
                )
            else:
                recorder.record_snapshot(
                    ts_ms=ts_ms,
                    step=self.step,
                    tick=self.tick,
                    snapshot=sanitized_snapshot,
                    snapshot_is_scene_safe=True,
                )
        else:
            recorder.record_snapshot(
                ts_ms=ts_ms,
                step=self.step,
                tick=self.tick,
                snapshot=sanitized_snapshot,
                snapshot_is_scene_safe=True,
            )
        self._last_realtime_snapshot_ts_ms = ts_ms

    def _record_visual_result(self) -> None:
        recorder = self.visual_recorder
        if recorder is None or self.final_result is None:
            return
        self._flush_pending_visual_snapshots()
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
        stop_background_snapshot_drain = getattr(recorder, "stop_background_snapshot_drain", None)
        if callable(stop_background_snapshot_drain):
            stop_background_snapshot_drain()
        replay_path = _resolve_visual_replay_path(
            result=self.final_result,
            invocation_context=self.invocation_context,
        )
        if replay_path is None:
            return
        try:
            artifacts = recorder.persist(replay_path)
            self._switch_visualization_to_replay(str(replay_path))
            replay_manifest_path = Path(replay_path)
            if replay_manifest_path.exists():
                updated = update_replay_manifest_visual_session_ref(
                    replay_path=replay_path,
                    visual_session_ref=artifacts.visual_session_ref,
                )
                if not updated:
                    self._visual_artifacts_error = "replay_manifest_update_failed"
                    self._record_visualization_artifact_error(self._visual_artifacts_error)
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
            self._record_visualization_artifact_error(self._visual_artifacts_error)
            logger.warning(
                "Arena visual sidecar persistence failed for replay_path={}: {}",
                replay_path,
                exc,
            )

    def _materialize_replay_artifact_if_missing(self) -> None:
        result = self.final_result
        if result is None:
            return
        replay_path = _resolve_result_replay_path(result)
        if replay_path not in (None, ""):
            return
        replay_path = _resolve_visual_replay_path(
            result=result,
            invocation_context=self.invocation_context,
        )
        if replay_path is None:
            return
        replay_file = Path(replay_path).expanduser()
        replay_writer = ReplaySchemaWriter(
            run_dir=_resolve_replay_writer_run_dir(replay_file),
            sample_id=replay_file.parent.name,
            output_dir=str(replay_file.parent.parent),
        )
        materialized_result = _coerce_result_with_replay_path(result, replay_path=str(replay_file))
        replay_writer.write(
            scheduler_type=str(self.sample.scheduler or getattr(self.visual_recorder, "scheduling_family", "turn")),
            result=materialized_result,
            move_log=_resolve_result_move_log(materialized_result),
            arena_trace=self.arena_trace,
            extra_meta={
                "game_kit": self.sample.game_kit,
                "env": self.sample.env,
            },
        )
        self.final_result = materialized_result

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

    def _bind_visualization_resource(
        self,
        *,
        mode: str,
        viewer_url: str | None = None,
        display_id: str | None = None,
        session_id: str | None = None,
        run_id: str | None = None,
        replay_viewer_close=None,
    ) -> None:
        resources = self.resources
        if resources is None:
            return
        from gage_eval.role.arena.resources.visualization import (
            VisualizationPhase,
            VisualizationSession,
        )

        artifacts: dict[str, object] = {
            "visualization_mode": str(mode),
        }
        if viewer_url not in (None, ""):
            artifacts["viewer_url"] = str(viewer_url)
        if display_id not in (None, ""):
            artifacts["display_id"] = str(display_id)
        if session_id not in (None, ""):
            artifacts["session_id"] = str(session_id)
        if run_id not in (None, ""):
            artifacts["run_id"] = str(run_id)

        visualization = VisualizationSession(
            phase=VisualizationPhase.LIVE,
            display=_VisualizationDisplayAdapter(),
            replay_viewer=_VisualizationReplayViewerAdapter(close_callback=replay_viewer_close),
            artifacts=artifacts,
        )
        _set_resource_value(resources, "visualization", visualization)
        _ensure_resource_category(resources, "visualization_resource")
        _record_resource_lifecycle(
            resources,
            "allocated",
            resource_category="visualization_resource",
        )

    def _switch_visualization_to_replay(self, replay_path: str) -> None:
        visualization = _get_resource_value(self.resources, "visualization")
        if visualization is None:
            return
        switch_to_replay = getattr(visualization, "switch_to_replay", None)
        if not callable(switch_to_replay):
            return
        switch_to_replay(str(replay_path))

    def _record_visualization_artifact_error(self, message: str) -> None:
        if not message:
            return
        visualization = _get_resource_value(self.resources, "visualization")
        artifacts = getattr(visualization, "artifacts", None)
        if isinstance(artifacts, dict):
            artifacts["artifact_error"] = {
                "error_code": "visualization_failure",
                "message": str(message),
            }

    def _consume_requested_restart(self) -> None:
        with self._stop_lock:
            if not self._restart_requested:
                return
            self._restart_requested = False
            self._stop_requested = False
            self._stop_reason = None
        self._restart_live_round()

    def _restart_live_round(self) -> None:
        environment = self.environment
        reset = getattr(environment, "reset", None)
        if not callable(reset):
            return
        reset_observation = reset()
        self.final_result = None
        self.tick = 0
        self.step = 0
        self.arena_trace.clear()
        self.support_errors.clear()
        self._current_trace_entry = None
        self._last_observation = reset_observation
        self._latest_live_frame_payload = None
        self._latest_live_frame_ts_ms = None
        self._latest_live_frame_version = 0
        self._last_realtime_snapshot_ts_ms = 0
        self._last_input_age_ms = None
        self._finalized = False
        recorder = self.visual_recorder
        if recorder is not None:
            self._flush_pending_visual_snapshots()
            recorder.reopen_live_round()
        self._reset_bound_player_runtime()
        if self._uses_scheduler_owned_human_realtime():
            self.capture_output_tick()
        else:
            self._record_visual_snapshot()

    def _reset_bound_player_runtime(self) -> None:
        for player in self.player_specs:
            reset_runtime_state = getattr(player, "reset_runtime_state", None)
            if callable(reset_runtime_state):
                reset_runtime_state()

    def _configure_visual_capabilities(self) -> None:
        recorder = self.visual_recorder
        if recorder is None:
            return
        start_background_snapshot_drain = getattr(recorder, "start_background_snapshot_drain", None)
        if self._uses_async_decimated_live_artifacts() and callable(start_background_snapshot_drain):
            start_background_snapshot_drain()
        extra_capabilities = dict(getattr(recorder, "extra_capabilities", {}) or {})
        if self._supports_live_restart():
            extra_capabilities["supportsRestart"] = True
        else:
            extra_capabilities.pop("supportsRestart", None)
        if self._supports_low_latency_realtime_input():
            extra_capabilities["supportsLowLatencyRealtimeInput"] = True
        else:
            extra_capabilities.pop("supportsLowLatencyRealtimeInput", None)
        if self._supports_realtime_input_websocket():
            extra_capabilities["supportsRealtimeInputWebSocket"] = True
        else:
            extra_capabilities.pop("supportsRealtimeInputWebSocket", None)
        if self._supports_live_update_stream():
            extra_capabilities["supportsLiveUpdateStream"] = True
        else:
            extra_capabilities.pop("supportsLiveUpdateStream", None)
        recorder.extra_capabilities = extra_capabilities

    def _supports_live_restart(self) -> bool:
        if self.environment is None or not callable(getattr(self.environment, "reset", None)):
            return False
        if not bool(getattr(self.runtime_profile, "pure_human_realtime", False)):
            return False
        recorder = self.visual_recorder
        return recorder is not None and str(recorder.scheduling_family) == "real_time_tick"

    def _supports_low_latency_realtime_input(self) -> bool:
        return bool(getattr(self.runtime_profile, "supports_low_latency_realtime_input", False))

    def _supports_realtime_input_websocket(self) -> bool:
        return bool(getattr(self.runtime_profile, "supports_realtime_input_websocket", False))

    def _supports_live_update_stream(self) -> bool:
        if self.visual_recorder is None:
            return False
        if self._visualization_mode != "arena_visual":
            return False
        if self._visualization_live_scene_scheme != "low_latency_channel":
            return False
        return self._supports_low_latency_realtime_input()

    def _should_strip_inline_media_from_live_trace(self) -> bool:
        return (
            self._visualization_mode == "arena_visual"
            and self._visualization_live_scene_scheme in _LIVE_MEDIA_INLINE_SCHEMES
            and self._supports_low_latency_realtime_input()
        )

    def _should_include_inline_snapshot_media(self) -> bool:
        if not self._should_strip_inline_media_from_live_trace():
            return True
        if self.final_result is not None:
            return True
        if self.step <= 1:
            return True
        stride = self._resolve_inline_snapshot_stride()
        if stride <= 1:
            return True
        return self.step % stride == 0

    def _resolve_inline_snapshot_stride(self) -> int:
        if not self._should_strip_inline_media_from_live_trace():
            return 1
        tick_interval_ms = getattr(self.runtime_profile, "tick_interval_ms", None)
        try:
            normalized_tick_ms = int(tick_interval_ms)
        except (TypeError, ValueError):
            normalized_tick_ms = 0
        if normalized_tick_ms <= 0:
            return 1
        stride = round(self._resolve_realtime_snapshot_interval_ms() / float(normalized_tick_ms))
        return max(1, int(stride))


class _VisualizationDisplayAdapter:
    def close_inputs(self) -> None:
        return None


class _VisualizationReplayViewerAdapter:
    def __init__(self, *, close_callback=None) -> None:
        self._close_callback = close_callback

    def load(self, replay_uri: str) -> None:
        del replay_uri
        return None

    def close(self) -> None:
        if callable(self._close_callback):
            self._close_callback()


def _get_resource_value(resources: object | None, key: str, default: object | None = None):
    if resources is None:
        return default
    if isinstance(resources, dict):
        return resources.get(key, default)
    return getattr(resources, key, default)


def _set_resource_value(resources: object | None, key: str, value: object) -> None:
    if resources is None:
        return
    if isinstance(resources, dict):
        resources[key] = value
        return
    setattr(resources, key, value)


def _ensure_resource_category(resources: object | None, category: str) -> None:
    if resources is None:
        return
    existing = _get_resource_value(resources, "resource_categories", ())
    if isinstance(existing, (list, tuple, set, frozenset)):
        normalized = tuple(str(item) for item in existing if str(item).strip())
    else:
        normalized = ()
    if category in normalized:
        return
    _set_resource_value(resources, "resource_categories", normalized + (str(category),))


def _record_resource_lifecycle(
    resources: object | None,
    phase: str,
    *,
    resource_category: str | None = None,
    details: dict[str, object] | None = None,
) -> None:
    if resources is None:
        return
    recorder = getattr(resources, "record_lifecycle", None)
    if callable(recorder):
        recorder(
            phase,
            resource_category=resource_category,
            details=details,
        )
        return
    event: dict[str, object] = {"phase": str(phase)}
    if resource_category is not None:
        event["resource_category"] = str(resource_category)
    if details:
        event["details"] = dict(details)
    if isinstance(resources, dict):
        resources["lifecycle_phase"] = str(phase)
        events = resources.setdefault("lifecycle_events", [])
        if isinstance(events, list):
            events.append(event)
        return
    setattr(resources, "lifecycle_phase", str(phase))
    events = getattr(resources, "lifecycle_events", None)
    if isinstance(events, list):
        events.append(event)
        return
    setattr(resources, "lifecycle_events", [event])


def _unregister_arena_visual_live_session(
    visual_server: object,
    *,
    session_id: str,
    run_id: str | None,
) -> None:
    unregister = getattr(visual_server, "unregister_live_session", None)
    if callable(unregister):
        unregister(session_id=session_id, run_id=run_id)


def _unregister_ws_rgb_display(ws_hub: object, *, display_id: str) -> None:
    unregister = getattr(ws_hub, "unregister_display", None)
    if callable(unregister):
        unregister(display_id)


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
    observer_modes = _resolve_visual_observer_modes(visualization_spec)
    observer_id = ""
    observer_kind = "spectator"
    human_player_ids = _collect_human_player_ids(resolved)
    if len(human_player_ids) == 1 and "player" in observer_modes:
        observer_id = human_player_ids[0]
        observer_kind = "player"

    return ArenaVisualSessionRecorder(
        plugin_id=str(plugin_id or "arena"),
        game_id=str(game_id or sample.game_kit or "arena"),
        scheduling_family=_resolve_scheduler_family(sample=sample, resolved=resolved),
        session_id=str(session_id),
        observer_modes=observer_modes,
        visual_kind=_resolve_visual_kind(visualization_spec),
        observer_id=observer_id,
        observer_kind=observer_kind,
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
    bindings = _inject_realtime_driver_params(bindings, runtime_profile=getattr(resolved, "runtime_profile", None))
    registry = getattr(resolved, "player_driver_registry", None)
    if registry is None:
        raise RuntimeError("resolved runtime binding is missing player_driver_registry")
    return tuple(registry.bind_all(bindings, invocation=invocation_context))


def _inject_realtime_driver_params(bindings, *, runtime_profile) -> tuple[PlayerBindingSpec, ...]:
    if runtime_profile is None:
        return tuple(bindings)
    uses_scheduler_owned = getattr(runtime_profile, "uses_scheduler_owned_human_realtime", None)
    if callable(uses_scheduler_owned):
        scheduler_owned = bool(uses_scheduler_owned())
    else:
        scheduler_owned = bool(getattr(runtime_profile, "scheduler_owns_realtime_clock", False))
    if not scheduler_owned:
        return tuple(bindings)
    realtime_control = getattr(runtime_profile, "realtime_human_control", None)
    tick_interval_ms = getattr(runtime_profile, "tick_interval_ms", None)
    updated: list[PlayerBindingSpec] = []
    for binding in bindings:
        if str(getattr(binding, "player_kind", "") or "").strip() != "human":
            updated.append(binding)
            continue
        driver_params = dict(binding.driver_params or {})
        driver_params["scheduler_owned_realtime"] = True
        if tick_interval_ms is not None and "tick_interval_ms" not in driver_params:
            driver_params["tick_interval_ms"] = int(tick_interval_ms)
        if realtime_control is not None:
            fallback_move = getattr(realtime_control, "fallback_move", None)
            if fallback_move not in (None, "") and "timeout_fallback_move" not in driver_params:
                driver_params["timeout_fallback_move"] = str(fallback_move)
        updated.append(replace(binding, driver_params=driver_params))
    return tuple(updated)


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

    player_input_semantics = _collect_human_realtime_input_semantics(resolved)
    action_router = SampleActionRouter(
        sample_id=sample_id,
        player_ids=human_player_ids,
        realtime_input_semantics_by_player=player_input_semantics,
    )
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


def _collect_stateful_human_player_ids(resolved) -> tuple[str, ...]:
    bindings = tuple(getattr(resolved, "player_bindings", ()) or ())
    player_ids = [
        str(binding.player_id)
        for binding in bindings
        if str(getattr(binding, "player_kind", "") or "").strip() == "human"
        and str(getattr(binding, "player_id", "") or "").strip()
        and _player_binding_flag_enabled(binding, "stateful_actions")
    ]
    return tuple(player_ids)


def _collect_human_realtime_input_semantics(resolved) -> dict[str, str]:
    runtime_profile = getattr(resolved, "runtime_profile", None)
    human_profiles = tuple(getattr(runtime_profile, "human_realtime_inputs", ()) or ())
    semantics_by_player: dict[str, str] = {}
    for profile in human_profiles:
        player_id = str(getattr(profile, "player_id", "") or "").strip()
        semantics = str(getattr(profile, "semantics", "") or "").strip()
        if player_id and semantics:
            semantics_by_player[player_id] = semantics
    if semantics_by_player:
        return semantics_by_player
    return {
        player_id: "continuous_state"
        for player_id in _collect_stateful_human_player_ids(resolved)
    }


def _player_binding_flag_enabled(binding: object, key: str) -> bool:
    params = getattr(binding, "driver_params", None)
    if not isinstance(params, Mapping):
        return False
    value = params.get(str(key))
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "on"}:
        return True
    if text in {"0", "false", "no", "off"}:
        return False
    return False


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
    port_value = config.get("port")
    server = ActionQueueServer(
        host=str(config.get("host") or "127.0.0.1"),
        port=8001 if port_value is None else int(port_value),
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


def _resolve_result_move_log(result: object | None) -> list[dict[str, Any]]:
    if result is None:
        return []
    if isinstance(result, Mapping):
        raw_move_log = result.get("move_log")
    else:
        raw_move_log = getattr(result, "move_log", None)
    if not isinstance(raw_move_log, (list, tuple)):
        return []
    return [dict(entry) for entry in raw_move_log if isinstance(entry, Mapping)]


def _coerce_result_with_replay_path(result: object, *, replay_path: str) -> GameResult:
    if isinstance(result, GameResult):
        return replace(result, replay_path=replay_path)
    winner = result.get("winner") if isinstance(result, Mapping) else getattr(result, "winner", None)
    status = result.get("result") if isinstance(result, Mapping) else getattr(result, "result", None)
    reason = result.get("reason") if isinstance(result, Mapping) else getattr(result, "reason", None)
    move_count = result.get("move_count") if isinstance(result, Mapping) else getattr(result, "move_count", 0)
    illegal_move_count = (
        result.get("illegal_move_count")
        if isinstance(result, Mapping)
        else getattr(result, "illegal_move_count", 0)
    )
    final_board = result.get("final_board") if isinstance(result, Mapping) else getattr(result, "final_board", "")
    rule_profile = result.get("rule_profile") if isinstance(result, Mapping) else getattr(result, "rule_profile", None)
    win_direction = (
        result.get("win_direction") if isinstance(result, Mapping) else getattr(result, "win_direction", None)
    )
    line_length = result.get("line_length") if isinstance(result, Mapping) else getattr(result, "line_length", None)
    winning_line = (
        result.get("winning_line") if isinstance(result, Mapping) else getattr(result, "winning_line", None)
    )
    arena_trace = result.get("arena_trace") if isinstance(result, Mapping) else getattr(result, "arena_trace", ())
    return GameResult(
        winner=str(winner) if winner not in (None, "") else None,
        result=str(status or "completed"),
        reason=str(reason) if reason not in (None, "") else None,
        move_count=int(move_count or 0),
        illegal_move_count=int(illegal_move_count or 0),
        final_board=str(final_board or ""),
        move_log=_resolve_result_move_log(result),
        rule_profile=str(rule_profile) if rule_profile not in (None, "") else None,
        win_direction=str(win_direction) if win_direction not in (None, "") else None,
        line_length=int(line_length) if line_length not in (None, "") else None,
        winning_line=tuple(str(item) for item in winning_line) if isinstance(winning_line, (list, tuple)) else None,
        replay_path=replay_path,
        arena_trace=tuple(dict(entry) for entry in arena_trace) if isinstance(arena_trace, (list, tuple)) else (),
    )


def _resolve_replay_writer_run_dir(replay_file: Path) -> Path:
    replay_path = replay_file.expanduser()
    parents = replay_path.parents
    if len(parents) >= 3:
        return parents[2]
    return replay_path.parent


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


def _visual_payload_snapshot(
    payload: object | None,
    *,
    include_inline_media: bool = True,
) -> object | None:
    return to_scene_json_safe(payload, preserve_data_urls=include_inline_media)


def _build_visual_snapshot_observation(
    *,
    environment: object | None,
    fallback_observation: object | None,
    tick: int,
    step: int,
    include_inline_frame_image: bool,
    frame_payload: Mapping[str, Any] | None = None,
) -> object | None:
    fallback_snapshot = _visual_payload_snapshot(
        fallback_observation,
        include_inline_media=include_inline_frame_image,
    )
    if frame_payload is None:
        frame_payload = _load_environment_frame_payload(environment)
    if frame_payload is None:
        return fallback_snapshot

    observation: dict[str, Any] = {}
    if isinstance(fallback_snapshot, Mapping):
        observation.update(dict(fallback_snapshot))

    active_player = (
        _string_or_none(frame_payload.get("active_player_id"))
        or _string_or_none(frame_payload.get("actor"))
        or _string_or_none(frame_payload.get("active_player"))
        or _string_or_none(observation.get("active_player"))
    )
    if active_player is not None:
        observation["active_player"] = active_player

    board_text = _string_or_none(frame_payload.get("board_text")) or _string_or_none(
        observation.get("board_text")
    )
    if board_text is not None:
        observation["board_text"] = board_text

    last_move = _string_or_none(frame_payload.get("last_move")) or _string_or_none(
        observation.get("last_move")
    )
    if last_move is not None:
        observation["last_move"] = last_move

    reward = frame_payload.get("reward")
    if reward is not None:
        observation["reward"] = reward

    move_count = _coerce_int(frame_payload.get("move_count"))
    if move_count is not None:
        observation["move_count"] = move_count

    metadata = _merge_snapshot_mapping(observation.get("metadata"), frame_payload.get("metadata"))
    if reward is not None:
        metadata.setdefault("reward", reward)
    if last_move is not None:
        metadata.setdefault("last_move", last_move)
    if active_player is not None:
        metadata.setdefault("player_id", active_player)
    if metadata:
        observation["metadata"] = metadata

    legal_actions = frame_payload.get("legal_actions")
    if isinstance(legal_actions, Mapping):
        observation["legal_actions"] = dict(legal_actions)
    else:
        legal_moves = _coerce_string_sequence(frame_payload.get("legal_moves"))
        if legal_moves:
            observation["legal_moves"] = legal_moves
            observation["legal_actions"] = {"items": legal_moves}

    context = _merge_snapshot_mapping(observation.get("context"), None)
    frame_tick = _coerce_int(frame_payload.get("tick"))
    frame_step = _coerce_int(frame_payload.get("step"))
    context["tick"] = frame_tick if frame_tick is not None else int(tick)
    if move_count is not None:
        context["step"] = move_count
    elif frame_step is not None:
        context["step"] = frame_step
    else:
        context["step"] = int(step)
    if context:
        observation["context"] = context

    view = _merge_snapshot_mapping(observation.get("view"), frame_payload.get("view"))
    if board_text is not None:
        view.setdefault("text", board_text)
    if not include_inline_frame_image:
        view.pop("image", None)
    image_payload = _resolve_frame_image_payload(
        frame_payload,
        include_inline_frame_image=include_inline_frame_image,
    )
    if image_payload is not None:
        view["image"] = image_payload
    if view:
        observation["view"] = view

    return observation


def _load_environment_frame_payload(environment: object | None) -> Mapping[str, Any] | None:
    if environment is None:
        return None
    getter = getattr(environment, "get_last_frame", None)
    if not callable(getter):
        return None
    try:
        payload = getter()
    except Exception:
        return None
    if not isinstance(payload, Mapping):
        return None
    return payload


def _merge_snapshot_mapping(primary: object | None, secondary: object | None) -> dict[str, Any]:
    merged: dict[str, Any] = {}
    if isinstance(primary, Mapping):
        merged.update(dict(primary))
    if isinstance(secondary, Mapping):
        merged.update(dict(secondary))
    return merged


def _resolve_input_age_ms(action: ArenaAction) -> float | None:
    metadata = getattr(action, "metadata", None)
    if not isinstance(metadata, Mapping):
        return None
    client_ts_ms = metadata.get("input_client_ts_ms")
    try:
        normalized = float(client_ts_ms)
    except (TypeError, ValueError):
        return None
    if normalized <= 0:
        return None
    return max(0.0, float(wall_clock_ms()) - normalized)


def _coerce_string_sequence(value: object | None) -> list[str]:
    if not isinstance(value, (list, tuple)):
        return []
    return [text for item in value if (text := str(item).strip())]


def _coerce_int(value: object | None) -> int | None:
    try:
        return int(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def _string_or_none(value: object | None) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _resolve_frame_image_payload(
    frame_payload: Mapping[str, Any],
    *,
    include_inline_frame_image: bool,
) -> dict[str, Any] | None:
    view_payload = frame_payload.get("view")
    if isinstance(view_payload, Mapping):
        image_payload = view_payload.get("image")
        if isinstance(image_payload, Mapping):
            if include_inline_frame_image:
                data_url = _string_or_none(image_payload.get("data_url") or image_payload.get("dataUrl"))
                if data_url is not None:
                    return dict(image_payload)

    rgb_frame = frame_payload.get("_rgb")
    if rgb_frame is None:
        for key in ("rgb", "rgb_array", "frame_rgb"):
            if frame_payload.get(key) is not None:
                rgb_frame = frame_payload.get(key)
                break
    if rgb_frame is None:
        return None
    if not include_inline_frame_image:
        return None

    data_url = _encode_frame_data_url(rgb_frame)
    if data_url is None:
        return None

    image_payload: dict[str, Any] = {"data_url": data_url}
    shape = getattr(rgb_frame, "shape", None)
    if isinstance(shape, (list, tuple)):
        try:
            image_payload["shape"] = [int(dim) for dim in shape]
        except Exception:
            pass
    dtype = getattr(rgb_frame, "dtype", None)
    if dtype is not None:
        image_payload["dtype"] = str(dtype)
    return image_payload


def _encode_frame_data_url(frame: object) -> str | None:
    if Image is None:
        return None
    try:
        image = Image.fromarray(frame)
        if image.mode not in {"RGB", "RGBA", "L"}:
            image = image.convert("RGB")
        elif image.mode == "RGBA":
            image = image.convert("RGB")
        buffer = io.BytesIO()
        image.save(buffer, format="PNG", optimize=True)
    except Exception:
        return None
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


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


def _best_effort_stop_environment(environment: object | None) -> None:
    if environment is None:
        return
    terminator = getattr(environment, "terminate", None)
    if callable(terminator):
        try:
            terminator()
            return
        except Exception:
            logger.debug("Arena live stop failed to terminate environment cleanly", exc_info=True)
    closer = getattr(environment, "close", None)
    if callable(closer):
        try:
            closer()
        except Exception:
            logger.debug("Arena live stop failed to close environment cleanly", exc_info=True)


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
