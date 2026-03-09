"""Arena role adapter for game loops."""

from __future__ import annotations

import asyncio
import functools
import json
import os
import re
import time
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

from loguru import logger

from gage_eval.assets.prompts.renderers import PromptRenderer
from gage_eval.observability.trace import ObservabilityTrace
from gage_eval.registry import registry
from gage_eval.role.adapters.base import RoleAdapter, RoleAdapterState
from gage_eval.role.arena.interfaces import MoveParser
from gage_eval.role.arena.players.agent_player import AgentPlayer
from gage_eval.role.arena.players.human_player import HumanPlayer
from gage_eval.role.arena.players.llm_player import LLMPlayer
from gage_eval.role.arena.schedulers.multi_timeline_scheduler import MultiTimelineScheduler
from gage_eval.role.arena.schedulers.record_scheduler import RecordScheduler
from gage_eval.role.arena.schedulers.simultaneous_scheduler import SimultaneousScheduler
from gage_eval.role.arena.frame_capture import FrameCaptureRecorder
from gage_eval.role.arena.replay_schema_writer import ReplaySchemaWriter
from gage_eval.role.arena.schedulers.tick_scheduler import TickScheduler
from gage_eval.role.arena.schedulers.turn_scheduler import TurnScheduler
from gage_eval.role.arena.types import GameResult


@registry.asset(
    "roles",
    "arena",
    desc="Arena role adapter for interactive games",
    tags=("role", "arena"),
    role_type="arena",
)
class ArenaRoleAdapter(RoleAdapter):
    """Role adapter that runs an arena game loop."""

    _DEFAULT_GAME_LOG_INLINE_LIMIT = 1000
    _DEFAULT_GAME_LOG_INLINE_BYTES = 200_000
    _DEFAULT_GAME_LOG_PREVIEW_LIMIT = 50

    def __init__(
        self,
        adapter_id: str,
        *,
        environment: Optional[Dict[str, Any]] = None,
        rules: Optional[Dict[str, Any]] = None,
        scheduler: Optional[Dict[str, Any]] = None,
        parser: Optional[Dict[str, Any]] = None,
        visualizer: Optional[Dict[str, Any]] = None,
        human_input: Optional[Dict[str, Any]] = None,
        players: Optional[Sequence[Dict[str, Any]]] = None,
        prompt_renderer: Optional[PromptRenderer] = None,
        capabilities=(),
        role_type: str = "arena",
        **_,
    ) -> None:
        resolved_caps = tuple(capabilities) if capabilities else ("text",)
        super().__init__(adapter_id=adapter_id, role_type=role_type, capabilities=resolved_caps)
        self._environment_cfg = dict(environment or {})
        self._rules_cfg = dict(rules or {})
        self._scheduler_cfg = dict(scheduler or {})
        self._parser_cfg = dict(parser or {})
        self._visualizer_cfg = dict(visualizer or {})
        self._human_input_cfg = dict(human_input or {})
        self._player_specs = list(players or [])
        self._prompt_renderer = prompt_renderer
        self._shared_visualizer = None
        self._action_server = None
        self._ws_rgb_hub = None
        self._registered_displays: set[str] = set()

    def invoke(self, payload: Dict[str, Any], state: RoleAdapterState) -> Dict[str, Any]:
        """Run the arena loop in a synchronous context."""

        return self._invoke_sync(payload, state)

    async def ainvoke(self, payload: Dict[str, Any], state: RoleAdapterState) -> Dict[str, Any]:
        """Run the arena loop without nesting sync calls inside an active loop."""

        loop = asyncio.get_running_loop()
        func = functools.partial(self._invoke_sync, payload, state)
        return await loop.run_in_executor(None, func)

    def _invoke_sync(self, payload: Dict[str, Any], state: RoleAdapterState) -> Dict[str, Any]:
        sample = payload.get("sample") or {}
        trace = payload.get("trace") if isinstance(payload.get("trace"), ObservabilityTrace) else None
        role_manager = payload.get("role_manager")
        if role_manager is None:
            raise ValueError("ArenaRoleAdapter requires role_manager in payload")

        # STEP 1: Build core components for the game loop.
        player_specs, player_ids, player_names, start_player_id = self._normalize_player_specs(sample)
        env_impl = self._environment_cfg.get("impl", "gomoku_local_v1")
        env_impl_lower = str(env_impl).lower()
        model_labels: Dict[str, str] = {}
        if "doudizhu" in env_impl_lower:
            model_labels = self._resolve_player_labels(player_specs, role_manager)
            for player_id in player_ids:
                label = model_labels.get(player_id)
                if label and player_id not in player_names:
                    player_names[player_id] = label
        elif "mahjong" in env_impl_lower:
            model_labels = self._resolve_player_labels(player_specs, role_manager)
        parser = self._build_parser(sample)
        scheduler = self._build_scheduler(sample)
        visualizer, action_queue = self._ensure_visualizer(sample, player_specs)
        action_server, action_queue_server = self._ensure_action_server(player_specs)
        if action_queue is None:
            action_queue = action_queue_server
        environment = self._build_environment(
            sample,
            player_ids=player_ids,
            player_names=player_names,
            player_models=model_labels if "mahjong" in env_impl_lower else None,
            start_player_id=start_player_id,
            chat_queue=action_server.chat_queue if action_server is not None else None,
            trace=trace,
        )
        frame_recorder = self._build_frame_capture_recorder(sample=sample, trace=trace)
        if frame_recorder is not None:
            environment = _FrameCaptureEnvironment(environment, frame_recorder)
        self._maybe_register_ws_display(
            sample=sample,
            environment=environment,
            action_queue=action_queue,
            player_specs=player_specs,
            env_impl=env_impl_lower,
        )
        if visualizer is not None:
            visualizer.reset_state()
            visualizer.set_players(
                player_ids=player_ids,
                player_names=player_names,
                player_labels=self._resolve_player_labels(player_specs, role_manager),
                active_player=start_player_id,
            )
            environment = _VisualizedEnvironment(environment, visualizer)
        players = self._build_players(
            sample,
            role_manager,
            parser,
            trace=trace,
            action_queue=action_queue,
            player_specs=player_specs,
        )

        logger.info("ArenaRoleAdapter {} starting game", self.adapter_id)
        if trace:
            trace.emit("arena_start", {"adapter_id": self.adapter_id, "player_count": len(players)})

        # STEP 2: Run the scheduler loop and capture the final result.
        try:
            result = scheduler.run_loop(environment, players)
        finally:
            self._wait_for_pending_players(players)
            # NOTE: We do NOT stop the visualizer here because it is shared across samples.
            pass
        output = self._format_result(
            result,
            sample,
            trace,
            frame_events=frame_recorder.build_frame_events() if frame_recorder is not None else None,
        )

        if visualizer is not None and self._visualizer_cfg.get("wait_for_finish"):
            visualizer.wait_for_finish()

        if trace:
            trace.emit(
                "arena_end",
                {
                    "adapter_id": self.adapter_id,
                    "winner": result.winner,
                    "result": result.result,
                    "move_count": result.move_count,
                },
            )
        logger.info("ArenaRoleAdapter {} finished result={}", self.adapter_id, result.result)
        return output

    def _normalize_player_specs(
        self, sample: Dict[str, Any]
    ) -> Tuple[list[Dict[str, Any]], list[str], Dict[str, str], str]:
        metadata = sample.get("metadata") or {}
        player_ids = metadata.get("player_ids") or []
        if isinstance(player_ids, dict):
            player_ids = list(player_ids.values())
        player_ids = [str(player_id) for player_id in player_ids if player_id]

        player_names = metadata.get("player_names") or {}
        if isinstance(player_names, list):
            player_names = {
                player_ids[idx]: name
                for idx, name in enumerate(player_names)
                if idx < len(player_ids)
            }
        if not isinstance(player_names, dict):
            player_names = {}

        normalized_specs: list[Dict[str, Any]] = []
        for idx, spec in enumerate(self._player_specs):
            normalized = dict(spec)
            player_id = normalized.get("player_id") or normalized.get("id")
            if not player_id:
                name = normalized.get("name")
                if name:
                    if name in player_ids:
                        player_id = name
                    else:
                        for candidate_id, candidate_name in player_names.items():
                            if candidate_name == name:
                                player_id = candidate_id
                                break
                if not player_id and name:
                    player_id = name
                if not player_id and player_ids and idx < len(player_ids):
                    player_id = player_ids[idx]
            if not player_id:
                player_id = f"player_{idx}"
            normalized["player_id"] = str(player_id)
            normalized_specs.append(normalized)
            existing_name = player_names.get(normalized["player_id"])
            display_name = normalized.get("name") or existing_name or normalized["player_id"]
            is_generic_name = False
            if existing_name is not None:
                normalized_existing = str(existing_name).strip()
                is_generic_name = bool(re.match(r"^player\s*\d+$", normalized_existing, re.IGNORECASE))
            if existing_name is None or existing_name == normalized["player_id"] or is_generic_name:
                adapter_ref = normalized.get("ref")
                if adapter_ref and not normalized.get("name"):
                    display_name = str(adapter_ref)
                player_names[normalized["player_id"]] = display_name

        if player_ids:
            ordered_ids = list(player_ids)
            for spec in normalized_specs:
                if spec["player_id"] not in ordered_ids:
                    ordered_ids.append(spec["player_id"])
        else:
            ordered_ids = [spec["player_id"] for spec in normalized_specs]

        for player_id in ordered_ids:
            player_names.setdefault(player_id, player_id)

        start_player_id = (
            metadata.get("start_player_id")
            or self._environment_cfg.get("start_player_id")
            or self._scheduler_cfg.get("start_player_id")
        )
        if start_player_id:
            start_player_id = str(start_player_id)
            if start_player_id not in ordered_ids:
                for candidate_id, candidate_name in player_names.items():
                    if candidate_name == start_player_id:
                        start_player_id = candidate_id
                        break
        if not start_player_id or start_player_id not in ordered_ids:
            start_player_id = ordered_ids[0] if ordered_ids else "player_0"

        return normalized_specs, ordered_ids, player_names, start_player_id

    def _resolve_player_labels(self, player_specs: Sequence[Dict[str, Any]], role_manager) -> Dict[str, str]:
        labels: Dict[str, str] = {}
        for spec in player_specs:
            player_id = spec.get("player_id")
            player_type = spec.get("type")
            adapter_ref = spec.get("ref")
            if not player_id or not player_type:
                continue
            if player_type == "human":
                labels[player_id] = "Human"
                continue
            if player_type == "agent":
                labels[player_id] = adapter_ref or "Agent"
                continue
            if player_type == "backend":
                adapter = role_manager.get_adapter(adapter_ref) if adapter_ref else None
                model_name = None
                if adapter is not None:
                    backend = getattr(adapter, "backend", None)
                    config = getattr(backend, "config", {}) if backend is not None else {}
                    if isinstance(config, dict):
                        for key in ("model", "model_name", "model_path", "model_id", "model_repo"):
                            value = config.get(key)
                            if value:
                                model_name = str(value)
                                break
                labels[player_id] = model_name or adapter_ref or "Backend"
        return labels

    def _build_environment(
        self,
        sample: Dict[str, Any],
        *,
        player_ids: Optional[Sequence[str]] = None,
        player_names: Optional[Dict[str, str]] = None,
        player_models: Optional[Dict[str, str]] = None,
        start_player_id: Optional[str] = None,
        chat_queue=None,
        trace: Optional[ObservabilityTrace] = None,
    ):
        metadata = sample.get("metadata") or {}
        eval_config = sample.get("eval_config") or {}

        env_cfg = dict(self._environment_cfg)
        rules_cfg = dict(self._rules_cfg)

        board_size = int(metadata.get("board_size", env_cfg.get("board_size", 15)))
        win_len = int(metadata.get("win_len", rules_cfg.get("win_len", env_cfg.get("win_len", 5))))
        resolved_player_ids = list(
            player_ids
            or metadata.get("player_ids")
            or []
        )
        resolved_player_names = dict(player_names or metadata.get("player_names") or {})
        resolved_player_models = dict(player_models or metadata.get("player_models") or {})
        resolved_start_player = start_player_id or metadata.get("start_player_id")
        token_map = metadata.get("token_map") or env_cfg.get("token_map")
        coord_scheme = metadata.get("coord_scheme", env_cfg.get("coord_scheme", "A1"))

        illegal_policy = dict(rules_cfg.get("illegal_policy") or {})
        if "retry_illegal" in eval_config and "retry" not in illegal_policy:
            illegal_policy["retry"] = eval_config.get("retry_illegal")

        rule_profile = rules_cfg.get("rule_profile", metadata.get("rule_profile", "freestyle"))
        win_directions = rules_cfg.get("win_directions", metadata.get("win_directions"))
        chat_mode = env_cfg.get("chat_mode", metadata.get("chat_mode"))

        impl = env_cfg.get("impl", "gomoku_local_v1")
        if "mahjong" in str(impl).lower():
            try:
                import gage_eval.role.arena.games.mahjong.env
            except ImportError:
                pass
        env_cls = registry.get("arena_impls", impl)
        env_kwargs = {
            "board_size": board_size,
            "win_len": win_len,
            "player_ids": resolved_player_ids or None,
            "player_names": resolved_player_names or None,
            "token_map": token_map,
            "start_player_id": resolved_start_player,
            "coord_scheme": coord_scheme,
            "rule_profile": rule_profile,
            "win_directions": win_directions,
            "illegal_policy": illegal_policy,
        }
        if "pettingzoo" in str(impl).lower():
            if env_cfg.get("env_id") is not None:
                env_kwargs["env_id"] = env_cfg.get("env_id")
            if env_cfg.get("env_kwargs") is not None:
                env_kwargs["env_kwargs"] = env_cfg.get("env_kwargs")
            if env_cfg.get("seed") is not None:
                env_kwargs["seed"] = env_cfg.get("seed")
            if env_cfg.get("action_labels") is not None:
                env_kwargs["action_labels"] = env_cfg.get("action_labels")
            if env_cfg.get("use_action_meanings") is not None:
                env_kwargs["use_action_meanings"] = env_cfg.get("use_action_meanings")
            if env_cfg.get("include_raw_obs") is not None:
                env_kwargs["include_raw_obs"] = env_cfg.get("include_raw_obs")
            if env_cfg.get("agent_map") is not None:
                env_kwargs["agent_map"] = env_cfg.get("agent_map")
        if "vizdoom" in str(impl).lower():
            for key in (
                "use_single_process",
                "render_mode",
                "pov_view",
                "show_automap",
                "automap_scale",
                "automap_follow",
                "automap_stride",
                "show_pov",
                "capture_pov",
                "pov_stride",
                "allow_respawn",
                "respawn_grace_steps",
                "no_attack_seconds",
                "max_steps",
                "action_repeat",
                "sleep_s",
                "port",
                "config_path",
                "replay_output_dir",
                "game_id",
                "tick_rate_hz",
                "frame_stride",
                "time_source",
                "obs_image",
                "obs_image_history_len",
                "replay_in_env",
                "action_labels",
                "allow_partial_actions",
                "reset_retry_count",
                "death_check_warmup_steps",
            ):
                if env_cfg.get(key) is not None:
                    env_kwargs[key] = env_cfg.get(key)
            if self._replay_mode_includes_frame(self._resolve_replay_recording_mode()):
                env_kwargs.setdefault("capture_pov", True)
        if "retro" in str(impl).lower():
            for key in (
                "game",
                "state",
                "default_state",
                "rom_path",
                "runtime_policy",
                "display_mode",
                "record_bk2",
                "record_dir",
                "record_filename",
                "record_path",
                "action_mapping",
                "legal_moves",
                "info_feeder",
                "action_schema",
                "token_budget",
                "frame_stride",
                "snapshot_stride",
                "obs_image",
                "replay_output_dir",
                "replay_filename",
                "frame_output_dir",
                "seed",
            ):
                if env_cfg.get(key) is not None:
                    env_kwargs[key] = env_cfg.get(key)
            run_id = trace.run_id if trace is not None else os.environ.get("GAGE_EVAL_RUN_ID")
            sample_id = sample.get("id") or sample.get("sample_id") or os.environ.get("GAGE_EVAL_SAMPLE_ID")
            if run_id:
                env_kwargs["run_id"] = str(run_id)
            if sample_id:
                env_kwargs["sample_id"] = str(sample_id)
        if chat_mode is not None:
            env_kwargs["chat_mode"] = chat_mode
        if "mahjong" in str(impl).lower():
            run_id = trace.run_id if trace is not None else os.environ.get("GAGE_EVAL_RUN_ID")
            sample_id = sample.get("id") or sample.get("sample_id") or os.environ.get("GAGE_EVAL_SAMPLE_ID")
            if run_id:
                env_kwargs["run_id"] = str(run_id)
            if sample_id:
                env_kwargs["sample_id"] = str(sample_id)
            if env_cfg.get("chat_every_n") is not None:
                env_kwargs["chat_every_n"] = env_cfg.get("chat_every_n")
            if env_cfg.get("replay_live") is not None:
                env_kwargs["replay_live"] = env_cfg.get("replay_live")
            if env_cfg.get("replay_output_dir") is not None:
                env_kwargs["replay_output_dir"] = env_cfg.get("replay_output_dir")
            if env_cfg.get("replay_filename") is not None:
                env_kwargs["replay_filename"] = env_cfg.get("replay_filename")
            if chat_queue is not None:
                env_kwargs["chat_queue"] = chat_queue
            if resolved_player_models:
                env_kwargs["player_models"] = resolved_player_models
        if "doudizhu" in str(impl).lower():
            run_id = trace.run_id if trace is not None else os.environ.get("GAGE_EVAL_RUN_ID")
            sample_id = sample.get("id") or sample.get("sample_id") or os.environ.get("GAGE_EVAL_SAMPLE_ID")
            if run_id:
                env_kwargs["run_id"] = str(run_id)
            if sample_id:
                env_kwargs["sample_id"] = str(sample_id)
            if env_cfg.get("chat_every_n") is not None:
                env_kwargs["chat_every_n"] = env_cfg.get("chat_every_n")
            if env_cfg.get("replay_live") is not None:
                env_kwargs["replay_live"] = env_cfg.get("replay_live")
            if env_cfg.get("replay_output_dir") is not None:
                env_kwargs["replay_output_dir"] = env_cfg.get("replay_output_dir")
            if env_cfg.get("replay_filename") is not None:
                env_kwargs["replay_filename"] = env_cfg.get("replay_filename")
            if env_cfg.get("context_include_public") is not None:
                env_kwargs["context_include_public"] = env_cfg.get("context_include_public")
            if env_cfg.get("context_include_ui_state") is not None:
                env_kwargs["context_include_ui_state"] = env_cfg.get("context_include_ui_state")
            if env_cfg.get("fast_finish_action") is not None:
                env_kwargs["fast_finish_action"] = env_cfg.get("fast_finish_action")
            if env_cfg.get("fast_finish_human_only") is not None:
                env_kwargs["fast_finish_human_only"] = env_cfg.get("fast_finish_human_only")
            if chat_queue is not None:
                env_kwargs["chat_queue"] = chat_queue
        return env_cls(**env_kwargs)

    def _build_parser(self, sample: Dict[str, Any]) -> MoveParser:
        metadata = sample.get("metadata") or {}
        cfg = dict(self._parser_cfg)
        impl = cfg.get("impl") or cfg.get("implementation") or "grid_parser_v1"
        board_size = int(metadata.get("board_size", cfg.get("board_size", 15)))
        coord_scheme = cfg.get("coord_scheme", metadata.get("coord_scheme", "A1"))
        parser_cls = registry.get("parser_impls", impl)
        try:
            return parser_cls(board_size=board_size, coord_scheme=coord_scheme)
        except TypeError:
            return parser_cls(board_size=board_size)

    def _build_scheduler(self, sample: Dict[str, Any]):
        cfg = dict(self._scheduler_cfg)
        eval_cfg = sample.get("eval_config") or {}
        trace_options = self._resolve_trace_scheduler_options(cfg)
        scheduler_type = str(cfg.get("type", "turn")).strip().lower()
        max_turns = eval_cfg.get("max_turns", cfg.get("max_turns"))
        if scheduler_type == "tick":
            tick_ms = int(cfg.get("tick_ms", 100))
            max_ticks = cfg.get("max_ticks")
            return TickScheduler(
                tick_ms=tick_ms,
                max_ticks=max_ticks,
                trace_step_index_start=trace_options["step_index_start"],
                trace_timestamp_clock=trace_options["timestamp_clock"],
                trace_time_clock=trace_options["time_clock"],
                trace_finalize_timing=trace_options["finalize_timing"],
                trace_action_format=trace_options["action_format"],
            )
        if scheduler_type == "turn":
            return TurnScheduler(max_turns=max_turns)
        if scheduler_type == "record":
            max_ticks = eval_cfg.get(
                "max_turns",
                cfg.get("max_ticks", cfg.get("max_steps", cfg.get("max_turns"))),
            )
            tick_ms_value = cfg.get("tick_ms")
            if tick_ms_value is None and cfg.get("record_fps") is None:
                tick_ms_value = 33
            return RecordScheduler(
                record_fps=cfg.get("record_fps"),
                tick_ms=None if tick_ms_value is None else int(tick_ms_value),
                max_ticks=max_ticks,
                action_timeout_ms=cfg.get("action_timeout_ms"),
                timeout_fallback_move=str(cfg.get("timeout_fallback_move", "NOOP")),
                timeline_id=cfg.get("timeline_id"),
                trace_step_index_start=trace_options["step_index_start"],
                trace_timestamp_clock=trace_options["timestamp_clock"],
                trace_time_clock=trace_options["time_clock"],
                trace_finalize_timing=trace_options["finalize_timing"],
                trace_action_format=trace_options["action_format"],
            )
        if scheduler_type == "simultaneous":
            max_steps = eval_cfg.get("max_turns", cfg.get("max_steps", cfg.get("max_turns")))
            return SimultaneousScheduler(
                frames_per_action=int(cfg.get("frames_per_action", 1)),
                max_steps=max_steps,
                action_timeout_ms=cfg.get("action_timeout_ms"),
                timeout_fallback_move=str(cfg.get("timeout_fallback_move", "NOOP")),
                tick_ms=int(cfg.get("tick_ms", 0)),
                timeline_id=cfg.get("timeline_id"),
                trace_step_index_start=trace_options["step_index_start"],
                trace_timestamp_clock=trace_options["timestamp_clock"],
                trace_time_clock=trace_options["time_clock"],
                trace_finalize_timing=trace_options["finalize_timing"],
                trace_action_format=trace_options["action_format"],
            )
        if scheduler_type == "multi_timeline":
            return MultiTimelineScheduler(
                tick_ms=int(cfg.get("tick_ms", 33)),
                max_ticks=cfg.get("max_ticks"),
                default_fallback_move=str(cfg.get("default_fallback_move", "NOOP")),
                lane_registry=cfg.get("lane_registry"),
                timelines=cfg.get("timelines"),
                trace_step_index_start=trace_options["step_index_start"],
                trace_timestamp_clock=trace_options["timestamp_clock"],
                trace_time_clock=trace_options["time_clock"],
                trace_finalize_timing=trace_options["finalize_timing"],
                trace_action_format=trace_options["action_format"],
            )
        raise ValueError(f"Unsupported scheduler type: {scheduler_type}")

    @staticmethod
    def _resolve_trace_scheduler_options(cfg: Dict[str, Any]) -> Dict[str, Any]:
        """Resolve trace-related scheduler options from config."""

        trace_cfg = cfg.get("trace")
        if not isinstance(trace_cfg, dict):
            trace_cfg = {}

        return {
            "step_index_start": int(
                ArenaRoleAdapter._pick_trace_option(
                    cfg,
                    trace_cfg,
                    key="step_index_start",
                    default=0,
                )
            ),
            "timestamp_clock": str(
                ArenaRoleAdapter._pick_trace_option(
                    cfg,
                    trace_cfg,
                    key="timestamp_clock",
                    default="wall_clock",
                )
            ),
            "time_clock": str(
                ArenaRoleAdapter._pick_trace_option(
                    cfg,
                    trace_cfg,
                    key="time_clock",
                    default="monotonic",
                )
            ),
            "finalize_timing": str(
                ArenaRoleAdapter._pick_trace_option(
                    cfg,
                    trace_cfg,
                    key="finalize_timing",
                    default="after_env_apply",
                )
            ),
            "action_format": str(
                ArenaRoleAdapter._pick_trace_option(
                    cfg,
                    trace_cfg,
                    key="action_format",
                    default="flat",
                )
            ),
        }

    @staticmethod
    def _pick_trace_option(
        scheduler_cfg: Dict[str, Any],
        trace_cfg: Dict[str, Any],
        *,
        key: str,
        default: Any,
    ) -> Any:
        """Read trace option from nested or flat scheduler config."""

        if key in trace_cfg:
            return trace_cfg[key]
        prefixed_key = f"trace_{key}"
        if prefixed_key in scheduler_cfg:
            return scheduler_cfg[prefixed_key]
        return default

    def _build_players(
        self,
        sample: Dict[str, Any],
        role_manager,
        parser: MoveParser,
        *,
        trace: Optional[ObservabilityTrace],
        action_queue,
        player_specs: Sequence[Dict[str, Any]],
    ):
        if not player_specs:
            raise ValueError("ArenaRoleAdapter requires non-empty players configuration")

        players = []
        for spec in player_specs:
            player_type = spec.get("type")
            player_id = spec.get("player_id")
            adapter_ref = spec.get("ref")
            if not player_id or not player_type:
                raise ValueError("ArenaRoleAdapter player requires player_id and type")
            if player_type == "backend":
                if not adapter_ref:
                    raise ValueError("backend player requires ref backend_id")
                players.append(
                    LLMPlayer(
                        name=player_id,
                        adapter_id=adapter_ref,
                        role_manager=role_manager,
                        sample=sample,
                        parser=parser,
                        trace=trace,
                        max_retries=int(spec.get("max_retries", 0)),
                        legal_moves_limit=int(spec.get("legal_moves_limit", 40)),
                        sampling_params=spec.get("sampling_params"),
                        fallback_policy=spec.get("fallback_policy", "none"),
                        timeout_ms=spec.get("timeout_ms"),
                        timeout_fallback_move=spec.get("timeout_fallback_move"),
                        prompt_renderer=self._prompt_renderer,
                        scheduler_mode=self._scheduler_cfg.get("type"),
                        scheme_id=spec.get("scheme_id"),
                        scheme_params=spec.get("scheme_params"),
                    )
                )
            elif player_type == "agent":
                if not adapter_ref:
                    raise ValueError("agent player requires ref agent_id")
                players.append(
                    AgentPlayer(
                        name=player_id,
                        adapter_id=adapter_ref,
                        role_manager=role_manager,
                        sample=sample,
                        parser=parser,
                        trace=trace,
                        max_retries=int(spec.get("max_retries", 0)),
                        legal_moves_limit=int(spec.get("legal_moves_limit", 40)),
                        sampling_params=spec.get("sampling_params"),
                        prompt_renderer=self._prompt_renderer,
                    )
                )
            elif player_type == "human":
                if not adapter_ref:
                    raise ValueError("human player requires ref human_adapter")
                players.append(
                    HumanPlayer(
                        name=player_id,
                        adapter_id=adapter_ref,
                        role_manager=role_manager,
                        sample=sample,
                        parser=parser,
                        trace=trace,
                        action_queue=action_queue,
                        timeout_ms=spec.get("timeout_ms"),
                        timeout_fallback_move=spec.get("timeout_fallback_move"),
                    )
                )
            else:
                raise ValueError(f"Unsupported player type: {player_type}")
        return players

    def _wait_for_pending_players(self, players: Sequence[Any]) -> None:
        """Wait briefly for async player workers before adapter return.

        Args:
            players: All instantiated players in the arena run.
        """

        timeout_s = self._coerce_non_negative_float(
            self._scheduler_cfg.get("pending_wait_timeout_s"),
            default=5.0,
        )
        if timeout_s <= 0:
            return

        deadline = time.monotonic() + timeout_s
        for player in players:
            wait_for_pending = getattr(player, "wait_for_pending", None)
            if not callable(wait_for_pending):
                continue
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return
            try:
                wait_for_pending(timeout_s=remaining)
            except Exception as exc:
                logger.debug("Skip pending wait for player {} due to: {}", getattr(player, "name", "?"), exc)

    def _format_result(
        self,
        result: GameResult,
        sample: Dict[str, Any],
        trace: Optional[ObservabilityTrace],
        *,
        frame_events: Optional[Sequence[Mapping[str, Any]]] = None,
    ) -> Dict[str, Any]:
        resolved_status = str(getattr(result, "status", None) or result.result)
        output = {
            "winner": result.winner,
            "status": resolved_status,
            "result": resolved_status,
            "reason": result.reason,
            "move_count": result.move_count,
            "illegal_move_count": result.illegal_move_count,
            "final_board": result.final_board,
            "rule_profile": result.rule_profile,
            "win_direction": result.win_direction,
            "line_length": result.line_length,
        }
        scores = getattr(result, "scores", None)
        if isinstance(scores, dict):
            output["scores"] = dict(scores)
        metrics = getattr(result, "metrics", None)
        if isinstance(metrics, dict):
            output["metrics"] = dict(metrics)
        arena_trace = self._normalize_arena_trace_steps(getattr(result, "arena_trace", None))
        if arena_trace is not None:
            output["arena_trace"] = arena_trace
        output.update(self._format_game_log(result.move_log, sample, trace))
        if result.replay_path:
            output["replay_path"] = result.replay_path
        replay_v1_path = self._maybe_write_replay_v1(
            result=result,
            sample=sample,
            trace=trace,
            output=output,
            frame_events=frame_events,
        )
        if replay_v1_path:
            if self._resolve_replay_primary_mode():
                output["replay_path"] = replay_v1_path
            else:
                output["replay_v1_path"] = replay_v1_path
        return output

    @staticmethod
    def _normalize_arena_trace_steps(raw_trace: Any) -> Optional[list[dict[str, Any]]]:
        trace_source = raw_trace
        if isinstance(trace_source, Mapping):
            # NOTE: Keep compatibility with legacy {"schema": "...", "steps": [...]} payloads.
            legacy_steps = trace_source.get("steps")
            if isinstance(legacy_steps, Sequence) and not isinstance(legacy_steps, (str, bytes)):
                trace_source = legacy_steps
            else:
                return []
        if not isinstance(trace_source, Sequence) or isinstance(trace_source, (str, bytes)):
            return None
        return [dict(item) for item in trace_source if isinstance(item, Mapping)]

    def _format_game_log(
        self,
        move_log: Sequence[Dict[str, Any]],
        sample: Dict[str, Any],
        trace: Optional[ObservabilityTrace],
    ) -> Dict[str, Any]:
        move_log_entries = list(move_log)
        if not move_log_entries:
            return {"game_log": []}

        if not self._should_externalize_game_log(move_log_entries):
            return {"game_log": move_log_entries}

        run_dir = self._resolve_run_dir(trace)
        sample_id = self._resolve_sample_id(sample)
        payload = {
            "adapter_id": self.adapter_id,
            "sample_id": sample_id,
            "move_log": move_log_entries,
        }
        output_path = self._write_game_log(run_dir, sample_id, payload) if run_dir else None
        if output_path is None:
            logger.warning(
                "Large game_log detected but could not persist to run dir; returning preview only."
            )
            return self._preview_game_log(move_log_entries)

        preview = self._preview_game_log(move_log_entries)
        return {"game_log_path": str(output_path), **preview}

    def _should_externalize_game_log(self, move_log: Sequence[Dict[str, Any]]) -> bool:
        max_entries = self._read_int_env(
            "GAGE_EVAL_GAME_LOG_INLINE_LIMIT",
            self._DEFAULT_GAME_LOG_INLINE_LIMIT,
        )
        if max_entries >= 0 and len(move_log) > max_entries:
            return True

        max_bytes = self._read_int_env(
            "GAGE_EVAL_GAME_LOG_INLINE_BYTES",
            self._DEFAULT_GAME_LOG_INLINE_BYTES,
        )
        if max_bytes <= 0:
            return False
        payload_bytes = self._estimate_game_log_bytes(move_log)
        if payload_bytes is None:
            return True
        return payload_bytes > max_bytes

    def _preview_game_log(self, move_log: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
        preview_limit = self._read_int_env(
            "GAGE_EVAL_GAME_LOG_PREVIEW_LIMIT",
            self._DEFAULT_GAME_LOG_PREVIEW_LIMIT,
        )
        preview = list(move_log[: max(preview_limit, 0)])
        return {
            "game_log_preview": preview,
            "game_log_truncated": len(move_log) > len(preview),
            "game_log_total": len(move_log),
        }

    @staticmethod
    def _estimate_game_log_bytes(move_log: Sequence[Dict[str, Any]]) -> Optional[int]:
        try:
            payload = json.dumps(move_log, ensure_ascii=True, default=str)
        except (TypeError, ValueError):
            return None
        return len(payload.encode("utf-8"))

    def _resolve_run_dir(self, trace: Optional[ObservabilityTrace]) -> Optional[Path]:
        run_id = trace.run_id if trace is not None else os.environ.get("GAGE_EVAL_RUN_ID")
        if not run_id:
            return None
        base_dir = Path(os.environ.get("GAGE_EVAL_SAVE_DIR", "./runs")).expanduser().resolve()
        return base_dir / str(run_id)

    @staticmethod
    def _resolve_sample_id(sample: Dict[str, Any]) -> str:
        sample_id = sample.get("id") or sample.get("sample_id") or "sample"
        return str(sample_id)

    def _write_game_log(
        self,
        run_dir: Path,
        sample_id: str,
        payload: Dict[str, Any],
    ) -> Optional[Path]:
        safe_sample_id = self._sanitize_filename(sample_id)
        safe_adapter_id = self._sanitize_filename(self.adapter_id)
        target_dir = run_dir / "artifacts" / "arena"
        filename = f"{safe_sample_id}_{safe_adapter_id}_game_log.json"
        output_path = target_dir / filename
        try:
            target_dir.mkdir(parents=True, exist_ok=True)
            output_path.write_text(
                json.dumps(payload, ensure_ascii=True, indent=2, default=str),
                encoding="utf-8",
            )
        except Exception as exc:
            logger.warning("Failed to write game_log to {}: {}", output_path, exc)
            return None
        return output_path

    @staticmethod
    def _sanitize_filename(value: str) -> str:
        sanitized = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value)).strip("_")
        return sanitized or "unknown"

    @staticmethod
    def _read_int_env(name: str, default: int) -> int:
        raw = os.environ.get(name)
        if raw is None:
            return default
        try:
            return int(raw)
        except (TypeError, ValueError):
            return default

    @staticmethod
    def _coerce_non_negative_float(value: Any, *, default: float) -> float:
        if value is None:
            return default
        try:
            parsed = float(value)
        except (TypeError, ValueError):
            return default
        if parsed < 0:
            return 0.0
        return parsed

    @staticmethod
    def _coerce_bool(value: Any, *, default: bool) -> bool:
        if isinstance(value, bool):
            return value
        if value is None:
            return default
        if isinstance(value, (int, float)):
            return bool(value)
        normalized = str(value).strip().lower()
        if normalized in {"1", "true", "yes", "on"}:
            return True
        if normalized in {"0", "false", "no", "off"}:
            return False
        return default

    def _resolve_replay_recording_mode(self) -> str:
        """Resolve replay recording mode from environment.replay config."""

        replay_cfg = self._environment_cfg.get("replay")
        if not isinstance(replay_cfg, Mapping):
            return "action"
        explicit_mode = self._normalize_recording_mode(replay_cfg.get("mode"))
        if explicit_mode is not None:
            return explicit_mode
        action_enabled = self._resolve_replay_action_enabled(replay_cfg)
        frame_enabled = self._resolve_replay_frame_enabled(replay_cfg, default=False)
        if action_enabled and frame_enabled:
            return "both"
        if frame_enabled:
            return "frame"
        return "action"

    def _resolve_replay_action_enabled(self, replay_cfg: Mapping[str, Any]) -> bool:
        action_cfg = replay_cfg.get("action")
        if not isinstance(action_cfg, Mapping):
            return True
        raw_enabled = action_cfg.get("enabled")
        return self._coerce_bool_or_auto(raw_enabled, default=True)

    def _resolve_replay_frame_enabled(self, replay_cfg: Mapping[str, Any], *, default: bool) -> bool:
        frame_cfg = replay_cfg.get("frame")
        if isinstance(frame_cfg, Mapping):
            if "enabled" in frame_cfg:
                return self._coerce_bool_or_auto(frame_cfg.get("enabled"), default=True)
            return True
        legacy_frame_cfg = replay_cfg.get("frame_capture")
        if isinstance(legacy_frame_cfg, Mapping):
            if "enabled" in legacy_frame_cfg:
                return self._coerce_bool_or_auto(legacy_frame_cfg.get("enabled"), default=True)
            return True
        return default

    def _resolve_replay_frame_capture_cfg(self, replay_cfg: Mapping[str, Any]) -> dict[str, Any]:
        frame_cfg = replay_cfg.get("frame")
        frame_options = dict(frame_cfg) if isinstance(frame_cfg, Mapping) else {}
        legacy_frame_cfg = replay_cfg.get("frame_capture")
        legacy_options = dict(legacy_frame_cfg) if isinstance(legacy_frame_cfg, Mapping) else {}
        return {
            "frame_dir_name": self._first_non_none(
                frame_options.get("frame_dir_name"),
                frame_options.get("dir"),
                legacy_options.get("frame_dir_name"),
            )
            or "frames",
            "frame_stride": self._first_non_none(
                frame_options.get("stride"),
                frame_options.get("frame_stride"),
                legacy_options.get("stride"),
                legacy_options.get("frame_stride"),
            ),
            "max_frames": self._first_non_none(
                frame_options.get("max_frames"),
                legacy_options.get("max_frames"),
            ),
            "format": self._first_non_none(
                frame_options.get("encoding"),
                frame_options.get("format"),
                legacy_options.get("encoding"),
                legacy_options.get("format"),
            )
            or "jpeg",
            "quality": self._first_non_none(
                frame_options.get("quality"),
                legacy_options.get("quality"),
            ),
            "include_frame_snapshot": self._first_non_none(
                frame_options.get("include_frame_snapshot"),
                legacy_options.get("include_frame_snapshot"),
            ),
        }

    @staticmethod
    def _first_non_none(*values: Any) -> Any:
        for value in values:
            if value is not None:
                return value
        return None

    def _coerce_bool_or_auto(self, value: Any, *, default: bool) -> bool:
        if isinstance(value, str) and value.strip().lower() == "auto":
            return default
        return self._coerce_bool(value, default=default)

    @staticmethod
    def _normalize_recording_mode(value: Any) -> Optional[str]:
        if value is None:
            return None
        mode = str(value).strip().lower()
        if mode in {"action", "frame", "both"}:
            return mode
        return None

    @staticmethod
    def _replay_mode_includes_action(mode: str) -> bool:
        return mode in {"action", "both"}

    @staticmethod
    def _replay_mode_includes_frame(mode: str) -> bool:
        return mode in {"frame", "both"}

    def _build_frame_capture_recorder(
        self,
        *,
        sample: Dict[str, Any],
        trace: Optional[ObservabilityTrace],
    ) -> Optional[FrameCaptureRecorder]:
        """Build frame capture recorder when replay frame capture is enabled."""

        replay_cfg = self._environment_cfg.get("replay")
        if not isinstance(replay_cfg, Mapping):
            return None
        if not self._coerce_bool(replay_cfg.get("enabled"), default=False):
            return None
        recording_mode = self._resolve_replay_recording_mode()
        if not self._replay_mode_includes_frame(recording_mode):
            return None
        frame_cfg = self._resolve_replay_frame_capture_cfg(replay_cfg)

        run_dir = self._resolve_run_dir(trace)
        if run_dir is None:
            logger.warning("Frame capture enabled but run_id is missing; skip frame capture.")
            return None
        sample_id = self._resolve_sample_id(sample)
        replay_output_dir = replay_cfg.get("output_dir")
        replay_dir = ReplaySchemaWriter.resolve_replay_dir(
            run_dir=run_dir,
            sample_id=sample_id,
            output_dir=str(replay_output_dir) if replay_output_dir else None,
        )
        return FrameCaptureRecorder(
            replay_dir=replay_dir,
            frame_dir_name=str(frame_cfg.get("frame_dir_name") or "frames"),
            enabled=True,
            frame_stride=max(1, self._coerce_int(frame_cfg.get("frame_stride"), default=1)),
            max_frames=max(0, self._coerce_int(frame_cfg.get("max_frames"), default=0)),
            image_format=str(frame_cfg.get("format") or "jpeg"),
            jpeg_quality=max(1, min(95, self._coerce_int(frame_cfg.get("quality"), default=75))),
            include_frame_snapshot=self._coerce_bool(frame_cfg.get("include_frame_snapshot"), default=True),
        )

    @staticmethod
    def _coerce_int(value: Any, *, default: int) -> int:
        try:
            return int(value)
        except (TypeError, ValueError):
            return int(default)

    def _ensure_visualizer(self, sample: Dict[str, Any], player_specs: Sequence[Dict[str, Any]]):
        if self._shared_visualizer is not None:
            return self._shared_visualizer, self._shared_visualizer.action_queue

        if not self._visualizer_cfg:
            return None, None
        enabled = bool(self._visualizer_cfg.get("enabled", False))
        if not enabled:
            return None, None

        from gage_eval.role.arena.visualizers.gradio_visualizer import GradioVisualizer

        metadata = sample.get("metadata") or {}
        board_size = int(metadata.get("board_size", self._visualizer_cfg.get("board_size", 15)))
        port = int(self._visualizer_cfg.get("port", 7860))
        launch_browser = bool(self._visualizer_cfg.get("launch_browser", False))
        refresh_s = float(self._visualizer_cfg.get("refresh_s", 0.3))
        wait_for_finish = bool(self._visualizer_cfg.get("wait_for_finish", False))
        has_human = any(spec.get("type") == "human" for spec in player_specs)
        mode = "interactive" if has_human else "observer"
        coord_scheme = metadata.get("coord_scheme", self._visualizer_cfg.get("coord_scheme", "A1"))
        renderer_cfg = dict(self._visualizer_cfg.get("renderer") or {})
        renderer_impl = renderer_cfg.get(
            "impl",
            self._visualizer_cfg.get("renderer_impl", "gomoku_board_v1"),
        )
        renderer_params = dict(renderer_cfg.get("params") or {})
        sanitize_output = bool(self._visualizer_cfg.get("sanitize_output", True))
        max_output_chars = int(self._visualizer_cfg.get("max_output_chars", 2000))
        show_parsed_move = bool(self._visualizer_cfg.get("show_parsed_move", True))
        show_chat = bool(self._visualizer_cfg.get("show_chat", False))
        chat_max_entries = int(self._visualizer_cfg.get("chat_max_entries", 60))
        title = self._visualizer_cfg.get("title")

        visualizer = GradioVisualizer(
            board_size=board_size,
            port=port,
            launch_browser=launch_browser,
            mode=mode,
            refresh_s=refresh_s,
            auto_close=bool(self._visualizer_cfg.get("auto_close", False)),
            wait_for_finish=wait_for_finish,
            renderer_impl=renderer_impl,
            renderer_params=renderer_params,
            coord_scheme=coord_scheme,
            sanitize_output=sanitize_output,
            max_output_chars=max_output_chars,
            show_parsed_move=show_parsed_move,
            show_chat=show_chat,
            chat_max_entries=chat_max_entries,
            title=title,
        )
        visualizer.start()
        self._shared_visualizer = visualizer
        return visualizer, visualizer.action_queue

    def _ensure_action_server(self, player_specs: Sequence[Dict[str, Any]]):
        if self._action_server is not None:
            return self._action_server, self._action_server.action_queue

        if not self._human_input_cfg:
            return None, None
        enabled = bool(self._human_input_cfg.get("enabled", False))
        if not enabled:
            return None, None
        has_human = any(spec.get("type") == "human" for spec in player_specs)
        if not has_human:
            return None, None

        from gage_eval.tools.action_server import ActionQueueServer

        host = self._human_input_cfg.get("host", "127.0.0.1")
        port = int(self._human_input_cfg.get("port", 8001))
        allow_origin = self._human_input_cfg.get("allow_origin", "*")
        server = ActionQueueServer(host=str(host), port=port, allow_origin=str(allow_origin))
        server.start()
        self._action_server = server
        return server, server.action_queue

    def _maybe_register_ws_display(
        self,
        *,
        sample: Dict[str, Any],
        environment: Any,
        action_queue: Any,
        player_specs: Sequence[Dict[str, Any]],
        env_impl: str,
    ) -> None:
        """Register display metadata when websocket display mode is enabled."""

        display_mode = str(self._environment_cfg.get("display_mode") or "").lower()
        if display_mode not in {"websocket", "ws"}:
            return
        frame_source = getattr(environment, "get_last_frame", None)
        if not callable(frame_source):
            return
        input_mapper = self._bind_input_mapper(env_impl=env_impl)
        if input_mapper is None:
            return
        hub = self._ensure_ws_rgb_hub()
        if hub is None:
            return

        from gage_eval.tools.ws_rgb_server import DisplayRegistration

        display_id = self._build_display_id(sample=sample, env_impl=env_impl)
        legal_moves = self._environment_cfg.get("legal_moves")
        normalized_legal_moves = list(legal_moves) if isinstance(legal_moves, (list, tuple)) else None
        registration = DisplayRegistration(
            display_id=display_id,
            label=str(env_impl or "arena_display"),
            human_player_id=self._resolve_human_player_id(player_specs),
            frame_source=frame_source,
            input_mapper=input_mapper,
            legal_moves=normalized_legal_moves,
            action_queue=action_queue,
            default_context={
                "display_id": display_id,
                "sample_id": self._resolve_sample_id(sample),
                "adapter_id": self.adapter_id,
                "game_id": env_impl,
                "human_player_id": self._resolve_human_player_id(player_specs),
            },
        )
        hub.register_display(registration)
        self._registered_displays.add(display_id)

    def _ensure_ws_rgb_hub(self):
        """Start and cache a local WsRgbHubServer instance."""

        if self._ws_rgb_hub is not None:
            return self._ws_rgb_hub
        host = str(self._human_input_cfg.get("ws_host", self._human_input_cfg.get("host", "127.0.0.1")))
        port = int(self._human_input_cfg.get("ws_port", 5800))
        allow_origin = str(self._human_input_cfg.get("ws_allow_origin", self._human_input_cfg.get("allow_origin", "*")))
        from gage_eval.tools.ws_rgb_server import WsRgbHubServer

        hub = WsRgbHubServer(host=host, port=port, allow_origin=allow_origin)
        hub.start()
        logger.info("ArenaRoleAdapter {} ws_rgb hub ready at {}", self.adapter_id, hub.base_url)
        self._ws_rgb_hub = hub
        return self._ws_rgb_hub

    def _bind_input_mapper(self, *, env_impl: str):
        """Bind a game-specific mapper for websocket input routing."""

        normalized_env_impl = str(env_impl).lower()
        action_schema = self._environment_cfg.get("action_schema")

        if "retro" in normalized_env_impl:
            from gage_eval.role.arena.games.retro.retro_input_mapper import RetroInputMapper

            hold_ticks_default = None
            if isinstance(action_schema, dict):
                hold_ticks_default = action_schema.get("hold_ticks_default")
            if hold_ticks_default is None:
                hold_ticks_default = self._human_input_cfg.get("hold_ticks_default")
            if hold_ticks_default is None:
                return RetroInputMapper()
            try:
                return RetroInputMapper(default_hold_ticks=int(hold_ticks_default))
            except (TypeError, ValueError):
                return RetroInputMapper()

        if "mahjong" in normalized_env_impl:
            from gage_eval.role.arena.games.mahjong.mahjong_input_mapper import MahjongInputMapper

            key_map = None
            enforce_legal_moves = True
            if isinstance(action_schema, dict):
                key_map = action_schema.get("key_map")
                enforce_legal_moves = self._coerce_bool(
                    action_schema.get("enforce_legal_moves"),
                    default=True,
                )
            return MahjongInputMapper(
                key_map=key_map if isinstance(key_map, Mapping) else None,
                enforce_legal_moves=enforce_legal_moves,
            )

        if "doudizhu" in normalized_env_impl:
            from gage_eval.role.arena.games.doudizhu.doudizhu_input_mapper import DoudizhuInputMapper

            key_map = None
            enforce_legal_moves = True
            if isinstance(action_schema, dict):
                key_map = action_schema.get("key_map")
                enforce_legal_moves = self._coerce_bool(
                    action_schema.get("enforce_legal_moves"),
                    default=True,
                )
            return DoudizhuInputMapper(
                key_map=key_map if isinstance(key_map, Mapping) else None,
                enforce_legal_moves=enforce_legal_moves,
            )

        if "pettingzoo" in normalized_env_impl:
            from gage_eval.role.arena.games.pettingzoo.pettingzoo_input_mapper import (
                PettingZooDiscreteInputMapper,
            )

            key_map = None
            enforce_legal_moves = True
            if isinstance(action_schema, dict):
                key_map = action_schema.get("key_map")
                enforce_legal_moves = self._coerce_bool(
                    action_schema.get("enforce_legal_moves"),
                    default=True,
                )
            return PettingZooDiscreteInputMapper(
                key_map=key_map if isinstance(key_map, Mapping) else None,
                enforce_legal_moves=enforce_legal_moves,
            )

        if "vizdoom" in normalized_env_impl:
            from gage_eval.role.arena.games.vizdoom.vizdoom_input_mapper import ViZDoomInputMapper

            key_map = None
            enforce_legal_moves = True
            if isinstance(action_schema, dict):
                key_map = action_schema.get("key_map")
                enforce_legal_moves = self._coerce_bool(
                    action_schema.get("enforce_legal_moves"),
                    default=True,
                )
            return ViZDoomInputMapper(
                key_map=key_map if isinstance(key_map, Mapping) else None,
                enforce_legal_moves=enforce_legal_moves,
            )

        if "gomoku" in normalized_env_impl or "tictactoe" in normalized_env_impl:
            from gage_eval.role.arena.games.common.grid_coord_input_mapper import GridCoordInputMapper

            key_map = None
            enforce_legal_moves = True
            coord_scheme = self._environment_cfg.get("coord_scheme")
            if isinstance(action_schema, dict):
                key_map = action_schema.get("key_map")
                if action_schema.get("coord_scheme") is not None:
                    coord_scheme = action_schema.get("coord_scheme")
                enforce_legal_moves = self._coerce_bool(
                    action_schema.get("enforce_legal_moves"),
                    default=True,
                )
            return GridCoordInputMapper(
                key_map=key_map if isinstance(key_map, Mapping) else None,
                coord_scheme=str(coord_scheme) if coord_scheme else None,
                enforce_legal_moves=enforce_legal_moves,
            )

        return None

    def _build_display_id(self, *, sample: Dict[str, Any], env_impl: str) -> str:
        metadata = sample.get("metadata")
        if not isinstance(metadata, dict):
            metadata = {}
        task_id = str(sample.get("task_id") or metadata.get("task_id") or "task")
        sample_id = self._resolve_sample_id(sample)
        return f"{task_id}:{sample_id}:{self.adapter_id}:{env_impl}"

    @staticmethod
    def _resolve_human_player_id(player_specs: Sequence[Dict[str, Any]]) -> str:
        for spec in player_specs:
            if spec.get("type") != "human":
                continue
            player_id = spec.get("player_id") or spec.get("name")
            if player_id:
                return str(player_id)
        return "player_0"

    def _maybe_write_replay_v1(
        self,
        *,
        result: GameResult,
        sample: Dict[str, Any],
        trace: Optional[ObservabilityTrace],
        output: Dict[str, Any],
        frame_events: Optional[Sequence[Mapping[str, Any]]] = None,
    ) -> Optional[str]:
        """Write replay v1 artifacts when enabled by environment.replay."""

        if not self._resolve_replay_v1_enabled():
            return None

        # STEP 1: Resolve writer paths and replay options.
        run_dir = self._resolve_run_dir(trace)
        if run_dir is None:
            logger.debug("Replay v1 enabled but run_id is missing; skip replay v1 write.")
            return None
        sample_id = self._resolve_sample_id(sample)
        replay_cfg = self._environment_cfg.get("replay")
        output_dir = replay_cfg.get("output_dir") if isinstance(replay_cfg, dict) else None
        scheduler_type = str(self._scheduler_cfg.get("type", "turn")).strip().lower() or "turn"
        recording_mode = self._resolve_replay_recording_mode()
        explicit_mode = (
            self._normalize_recording_mode(replay_cfg.get("mode"))
            if isinstance(replay_cfg, Mapping)
            else None
        )
        if frame_events and not self._replay_mode_includes_frame(recording_mode) and explicit_mode is None:
            recording_mode = "both" if self._replay_mode_includes_action(recording_mode) else "frame"
        move_log = list(result.move_log) if self._replay_mode_includes_action(recording_mode) else []

        # STEP 2: Assemble metadata payload for replay manifest.
        extra_meta: Dict[str, Any] = {
            "adapter_id": self.adapter_id,
            "env_impl": self._environment_cfg.get("impl"),
            "env_id": self._environment_cfg.get("env_id"),
        }
        if frame_events:
            extra_meta["frame_events"] = [dict(item) for item in frame_events if isinstance(item, Mapping)]
        if output.get("game_log_path"):
            extra_meta["game_log_path"] = output.get("game_log_path")
        if result.replay_path:
            extra_meta["legacy_replay_path"] = result.replay_path
        arena_trace = self._normalize_arena_trace_steps(output.get("arena_trace"))

        # STEP 3: Persist replay.json + events.jsonl.
        writer = ReplaySchemaWriter(
            run_dir=run_dir,
            sample_id=sample_id,
            output_dir=str(output_dir) if output_dir else None,
        )
        try:
            replay_v1_path = writer.write(
                scheduler_type=scheduler_type,
                result=result,
                move_log=move_log,
                arena_trace=arena_trace,
                extra_meta=extra_meta,
                recording_mode=recording_mode,
            )
        except Exception as exc:  # pragma: no cover - defensive filesystem guard
            logger.warning("Failed to write replay v1 for sample {}: {}", sample_id, exc)
            return None

        if trace is not None:
            trace.emit(
                "arena_replay_v1_written",
                {
                    "adapter_id": self.adapter_id,
                    "sample_id": sample_id,
                    "path": replay_v1_path,
                    "primary_mode": self._resolve_replay_primary_mode(),
                },
            )
        return replay_v1_path

    def _resolve_replay_v1_enabled(self) -> bool:
        """Return whether replay v1 write is enabled."""

        replay_cfg = self._environment_cfg.get("replay")
        if not isinstance(replay_cfg, dict):
            return False
        return self._coerce_bool(replay_cfg.get("enabled"), default=False)

    def _resolve_replay_primary_mode(self) -> bool:
        """Return whether replay_path should point to replay v1 output."""

        replay_cfg = self._environment_cfg.get("replay")
        if not isinstance(replay_cfg, dict):
            return False
        return self._coerce_bool(replay_cfg.get("primary_mode"), default=False)

    def shutdown(self) -> None:
        """Shutdown the shared visualizer when the runtime ends."""

        if self._shared_visualizer is not None:
            try:
                self._shared_visualizer.stop()
            except Exception as exc:  # pragma: no cover - defensive
                logger.warning("ArenaRoleAdapter {} visualizer stop failed: {}", self.adapter_id, exc)
            self._shared_visualizer = None

        if self._action_server is not None:
            try:
                self._action_server.stop()
            except Exception as exc:  # pragma: no cover - defensive
                logger.warning("ArenaRoleAdapter {} action server stop failed: {}", self.adapter_id, exc)
            self._action_server = None

        if self._ws_rgb_hub is not None:
            try:
                for display_id in list(self._registered_displays):
                    self._ws_rgb_hub.unregister_display(display_id)
                self._ws_rgb_hub.stop()
            except Exception as exc:  # pragma: no cover - defensive
                logger.warning("ArenaRoleAdapter {} ws hub stop failed: {}", self.adapter_id, exc)
            self._ws_rgb_hub = None
            self._registered_displays = set()


class _VisualizedEnvironment:
    """Environment wrapper that pushes state snapshots into a visualizer."""

    def __init__(self, base_env, visualizer) -> None:
        self._base = base_env
        self._visualizer = visualizer
        self._last_action = None

    def reset(self) -> None:
        self._base.reset()
        self._update()

    def get_active_player(self) -> str:
        return self._base.get_active_player()

    def get_last_frame(self):
        frame_getter = getattr(self._base, "get_last_frame", None)
        if callable(frame_getter):
            return frame_getter()
        return {}

    def observe(self, player: str):
        return self._base.observe(player)

    def apply(self, action) -> Optional[GameResult]:
        self._last_action = action
        outcome = self._base.apply(action)
        self._update(outcome, action)
        return outcome

    def is_terminal(self) -> bool:
        return self._base.is_terminal()

    def build_result(self, *, result: str, reason: Optional[str]) -> GameResult:
        outcome = self._base.build_result(result=result, reason=reason)
        self._update(outcome, self._last_action)
        return outcome

    def __getattr__(self, item: str):
        return getattr(self._base, item)

    def _update(self, outcome: Optional[GameResult] = None, action=None) -> None:
        try:
            obs = self._base.observe(self._base.get_active_player())
            player_names = obs.metadata.get("player_names")
            if not isinstance(player_names, dict):
                player_names = {}
            active_label = player_names.get(obs.active_player, obs.active_player)
            status = ""
            if outcome is not None:
                winner_id = outcome.winner or "draw"
                winner = player_names.get(winner_id, winner_id)
                reason = outcome.reason or "unknown"
                status = f"Result: {outcome.result} Winner: {winner} Reason: {reason}"
            else:
                status = f"Turn: {active_label} Last: {obs.last_action or 'none'}"
            last_action_player = None
            last_action_raw = None
            last_action_move = None
            if action is not None:
                last_action_player = getattr(action, "player", None)
                last_action_raw = getattr(action, "raw", None)
                last_action_move = getattr(action, "move", None)
            self._visualizer.update(
                board_text=obs.view_text,
                status_text=status,
                last_move=obs.last_action,
                winning_line=obs.metadata.get("winning_line"),
                board_size=obs.metadata.get("board_size"),
                coord_scheme=obs.metadata.get("coord_scheme"),
                last_action_player=last_action_player,
                last_action_raw=last_action_raw,
                last_action_move=last_action_move,
                active_player=obs.active_player,
                chat_log=obs.metadata.get("chat_log"),
                final_state=outcome is not None,
            )
        except Exception:
            pass


class _FrameCaptureEnvironment:
    """Environment wrapper that captures replay frame events during execution."""

    def __init__(self, base_env, recorder: FrameCaptureRecorder) -> None:
        self._base = base_env
        self._recorder = recorder
        self._step_index = 0

    def reset(self) -> None:
        self._base.reset()
        self._step_index = 0
        self._capture_current(step=0, actor=None, force=True)

    def get_active_player(self) -> str:
        return self._base.get_active_player()

    def get_last_frame(self):
        frame_getter = getattr(self._base, "get_last_frame", None)
        if callable(frame_getter):
            return frame_getter()
        return {}

    def observe(self, player: str):
        return self._base.observe(player)

    def apply(self, action) -> Optional[GameResult]:
        outcome = self._base.apply(action)
        self._step_index += 1
        actor = getattr(action, "player", None)
        self._capture_current(
            step=self._step_index,
            actor=str(actor) if actor is not None else None,
            force=False,
        )
        return outcome

    def is_terminal(self) -> bool:
        return self._base.is_terminal()

    def build_result(self, *, result: str, reason: Optional[str]) -> GameResult:
        return self._base.build_result(result=result, reason=reason)

    def __getattr__(self, item: str):
        return getattr(self._base, item)

    def _capture_current(
        self,
        *,
        step: int,
        actor: Optional[str],
        force: bool,
    ) -> None:
        frame_getter = getattr(self._base, "get_last_frame", None)
        if not callable(frame_getter):
            return
        try:
            frame_payload = frame_getter()
        except Exception:
            return
        if not self._has_visible_frame_content(frame_payload):
            return
        self._recorder.capture(
            frame_payload,
            step=int(step),
            actor=actor,
            force=bool(force),
        )

    @staticmethod
    def _has_visible_frame_content(frame_payload: Any) -> bool:
        """Return whether frame payload carries visible replay content."""

        if frame_payload is None:
            return False
        if not isinstance(frame_payload, Mapping):
            return True

        for key in ("_rgb", "rgb", "rgb_array", "frame_rgb", "image_path", "frame_image_path", "_image_path_abs"):
            if frame_payload.get(key) is not None:
                return True
        image = frame_payload.get("image")
        if isinstance(image, Mapping) and image.get("path"):
            return True

        board_text = frame_payload.get("board_text")
        if isinstance(board_text, str) and board_text.strip():
            return True

        for key in (
            "public_state",
            "private_state",
            "ui_state",
            "state",
            "observation",
            "legal_moves",
            "legal_actions",
        ):
            if _FrameCaptureEnvironment._is_non_empty_value(frame_payload.get(key)):
                return True

        ignored_keys = {
            "metadata",
            "active_player_id",
            "observer_player_id",
            "player_id",
            "player_ids",
            "player_names",
            "move_count",
            "last_move",
            "reward",
            "termination",
            "truncation",
            "env_id",
            "agent_id",
            "timestamp_ms",
        }
        for key, value in frame_payload.items():
            if str(key) in ignored_keys:
                continue
            if _FrameCaptureEnvironment._is_non_empty_value(value):
                return True
        return False

    @staticmethod
    def _is_non_empty_value(value: Any) -> bool:
        if value is None:
            return False
        if isinstance(value, str):
            return bool(value.strip())
        if isinstance(value, Mapping):
            return bool(value)
        if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
            return len(value) > 0
        return bool(value)
