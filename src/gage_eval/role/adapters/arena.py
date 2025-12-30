"""Arena role adapter for game loops."""

from __future__ import annotations

import asyncio
import functools
from typing import Any, Dict, Optional, Sequence, Tuple

from loguru import logger

from gage_eval.observability.trace import ObservabilityTrace
from gage_eval.registry import registry
from gage_eval.role.adapters.base import RoleAdapter, RoleAdapterState
from gage_eval.role.arena.interfaces import MoveParser
from gage_eval.role.arena.players.agent_player import AgentPlayer
from gage_eval.role.arena.players.human_player import HumanPlayer
from gage_eval.role.arena.players.llm_player import LLMPlayer
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

    def __init__(
        self,
        adapter_id: str,
        *,
        environment: Optional[Dict[str, Any]] = None,
        rules: Optional[Dict[str, Any]] = None,
        scheduler: Optional[Dict[str, Any]] = None,
        parser: Optional[Dict[str, Any]] = None,
        visualizer: Optional[Dict[str, Any]] = None,
        players: Optional[Sequence[Dict[str, Any]]] = None,
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
        self._player_specs = list(players or [])
        self._shared_visualizer = None

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
        environment = self._build_environment(
            sample,
            player_ids=player_ids,
            player_names=player_names,
            start_player_id=start_player_id,
        )
        parser = self._build_parser(sample)
        scheduler = self._build_scheduler(sample)
        visualizer, action_queue = self._ensure_visualizer(sample, player_specs)
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
            # NOTE: We do NOT stop the visualizer here because it is shared across samples.
            pass
        output = self._format_result(result)

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
            display_name = normalized.get("name") or normalized["player_id"]
            player_names.setdefault(normalized["player_id"], display_name)

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
        start_player_id: Optional[str] = None,
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
        resolved_start_player = start_player_id or metadata.get("start_player_id")
        token_map = metadata.get("token_map") or env_cfg.get("token_map")
        coord_scheme = metadata.get("coord_scheme", env_cfg.get("coord_scheme", "A1"))

        illegal_policy = dict(rules_cfg.get("illegal_policy") or {})
        if "retry_illegal" in eval_config and "retry" not in illegal_policy:
            illegal_policy["retry"] = eval_config.get("retry_illegal")

        rule_profile = rules_cfg.get("rule_profile", metadata.get("rule_profile", "freestyle"))
        win_directions = rules_cfg.get("win_directions", metadata.get("win_directions"))

        impl = env_cfg.get("impl", "gomoku_local_v1")
        env_cls = registry.get("arena_impls", impl)
        return env_cls(
            board_size=board_size,
            win_len=win_len,
            player_ids=resolved_player_ids or None,
            player_names=resolved_player_names or None,
            token_map=token_map,
            start_player_id=resolved_start_player,
            coord_scheme=coord_scheme,
            rule_profile=rule_profile,
            win_directions=win_directions,
            illegal_policy=illegal_policy,
        )

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
        scheduler_type = str(cfg.get("type", "turn"))
        max_turns = eval_cfg.get("max_turns", cfg.get("max_turns"))
        if scheduler_type == "tick":
            tick_ms = int(cfg.get("tick_ms", 100))
            max_ticks = cfg.get("max_ticks")
            return TickScheduler(tick_ms=tick_ms, max_ticks=max_ticks)
        return TurnScheduler(max_turns=max_turns)

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
                    )
                )
            else:
                raise ValueError(f"Unsupported player type: {player_type}")
        return players

    @staticmethod
    def _format_result(result: GameResult) -> Dict[str, Any]:
        return {
            "winner": result.winner,
            "result": result.result,
            "reason": result.reason,
            "move_count": result.move_count,
            "illegal_move_count": result.illegal_move_count,
            "final_board": result.final_board,
            "game_log": list(result.move_log),
            "rule_profile": result.rule_profile,
            "win_direction": result.win_direction,
            "line_length": result.line_length,
        }

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
            title=title,
        )
        visualizer.start()
        self._shared_visualizer = visualizer
        return visualizer, visualizer.action_queue

    def shutdown(self) -> None:
        """Shutdown the shared visualizer when the runtime ends."""

        if self._shared_visualizer is None:
            return
        try:
            self._shared_visualizer.stop()
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning("ArenaRoleAdapter {} visualizer stop failed: {}", self.adapter_id, exc)
        self._shared_visualizer = None


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
        return self._base.build_result(result=result, reason=reason)

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
                status = f"Turn: {active_label} Last: {obs.last_move or 'none'}"
            last_action_player = None
            last_action_raw = None
            last_action_move = None
            if action is not None:
                last_action_player = getattr(action, "player", None)
                last_action_raw = getattr(action, "raw", None)
                last_action_move = getattr(action, "move", None)
            self._visualizer.update(
                board_text=obs.board_text,
                status_text=status,
                last_move=obs.last_move,
                winning_line=obs.metadata.get("winning_line"),
                board_size=obs.metadata.get("board_size"),
                coord_scheme=obs.metadata.get("coord_scheme"),
                last_action_player=last_action_player,
                last_action_raw=last_action_raw,
                last_action_move=last_action_move,
                active_player=obs.active_player,
                final_state=outcome is not None,
            )
        except Exception:
            pass
