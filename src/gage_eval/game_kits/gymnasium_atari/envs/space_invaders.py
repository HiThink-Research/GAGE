from __future__ import annotations

import importlib
import json
from dataclasses import dataclass, replace
from typing import Any, Optional, Sequence

from loguru import logger

from gage_eval.game_kits.aec_env_game.pettingzoo.action_codec import DiscreteActionCodec
from gage_eval.game_kits.aec_env_game.pettingzoo.observation import PettingZooPromptBuilder
from gage_eval.game_kits.real_time_game.backend_mode import normalize_backend_mode
from gage_eval.game_kits.replay_support import materialize_gamekit_replay
from gage_eval.registry import registry
from gage_eval.role.arena.replay_paths import resolve_invocation_run_sample_ids
from gage_eval.role.arena.resources.runtime_bridge import attach_runtime_resources
from gage_eval.role.arena.types import ArenaAction, ArenaObservation, GameResult

@dataclass(frozen=True)
class ActionSchema:
    """Defines macro-action hold tick constraints for Gymnasium Atari controls."""

    hold_ticks_min: int = 1
    hold_ticks_max: int = 1
    default_hold_ticks: int = 1


@registry.asset(
    "arena_impls",
    "gymnasium_atari_space_invaders_v1",
    desc="Gymnasium Atari Space Invaders arena adapter",
    tags=("gymnasium", "atari", "space_invaders", "arena"),
)
class SpaceInvadersGymEnvironment:
    """Single-player Space Invaders adapter backed by Gymnasium ALE."""

    def __init__(
        self,
        *,
        env_id: str = "ALE/SpaceInvaders-v5",
        env_kwargs: Optional[dict[str, Any]] = None,
        env: Optional[Any] = None,
        player_specs: Optional[Sequence[object]] = None,
        player_ids: Optional[Sequence[str]] = None,
        player_names: Optional[dict[str, str]] = None,
        seed: Optional[int] = None,
        action_labels: Optional[Sequence[str]] = None,
        use_action_meanings: bool = True,
        include_raw_obs: bool = False,
        illegal_policy: Optional[dict[str, str | int]] = None,
        action_schema: Optional[dict[str, Any]] = None,
        max_cycles: int = 1000,
        backend_mode: str = "real",
        replay_game_kit: Optional[str] = None,
        replay_env: Optional[str] = None,
        replay_output_dir: Optional[str] = None,
        run_id: Optional[str] = None,
        sample_id: Optional[str] = None,
        **_: object,
    ) -> None:
        self._env_id = str(env_id or "ALE/SpaceInvaders-v5")
        self._env_kwargs = dict(env_kwargs or {})
        self._seed = seed
        self._include_raw_obs = bool(include_raw_obs)
        self._use_action_meanings = bool(use_action_meanings)
        self._explicit_action_labels = [str(label) for label in action_labels or []]
        self._action_schema = self._build_action_schema(action_schema)
        self._max_cycles = max(1, int(max_cycles))
        self._backend_mode = normalize_backend_mode(backend_mode)
        self._illegal_policy = dict(illegal_policy or {})
        self._max_illegal = int(self._illegal_policy.get("retry", 0))
        self._illegal_on_fail = str(self._illegal_policy.get("on_fail", "loss"))
        self._replay_game_kit = str(replay_game_kit) if replay_game_kit else None
        self._replay_env = str(replay_env) if replay_env else None
        self._replay_output_dir = str(replay_output_dir) if replay_output_dir else None
        self._run_id = str(run_id) if run_id else None
        self._sample_id = str(sample_id) if sample_id else None

        self._player_ids, resolved_names = self._resolve_players(
            player_specs=player_specs,
            player_ids=player_ids,
        )
        self._player_names = {**resolved_names, **dict(player_names or {})}
        self._active_player_id = self._player_ids[0]
        self._codec = DiscreteActionCodec(self._explicit_action_labels or None)
        self._prompt_builder = PettingZooPromptBuilder()

        self._env = env if env is not None else self._build_env()
        self._render_logged = False
        self._illegal_counts: dict[str, int] = {}
        self._scores: dict[str, float] = {}
        self._move_log: list[dict[str, Any]] = []
        self._move_count = 0
        self._tick = 0
        self._last_move: Optional[str] = None
        self._last_obs: Any = None
        self._last_info: dict[str, Any] = {}
        self._last_reward = 0.0
        self._terminated = False
        self._truncated = False
        self._final_result: Optional[GameResult] = None
        self._last_frame: Optional[dict[str, Any]] = None
        self.reset()

    def reset(self) -> None:
        """Reset the Gymnasium environment and clear episode state."""

        obs, info = self._reset_env()
        self._last_obs = obs
        self._last_info = info
        self._last_reward = 0.0
        self._terminated = False
        self._truncated = False
        self._illegal_counts = {player_id: 0 for player_id in self._player_ids}
        self._scores = {player_id: 0.0 for player_id in self._player_ids}
        self._move_log = []
        self._move_count = 0
        self._tick = 0
        self._last_move = None
        self._final_result = None
        self._refresh_action_labels()
        self._last_frame = None
        self._refresh_last_frame()

    def get_active_player(self) -> str:
        return self._active_player_id

    def get_last_frame(self) -> dict[str, Any]:
        if self._last_frame is None:
            self._refresh_last_frame()
        return dict(self._last_frame or {})

    def observe(self, player: str) -> ArenaObservation:
        player_id = str(player or self._active_player_id)
        reward = self._last_reward
        termination = self._terminated
        truncation = self._truncated
        legal_moves = self._build_legal_moves(termination=termination, truncation=truncation)
        board_text = self._format_board_text(player_id, reward, termination, truncation)
        rgb_frame = self._resolve_frame_rgb(self._last_obs)
        metadata = self._build_metadata(
            player_id=player_id,
            reward=reward,
            termination=termination,
            truncation=truncation,
        )
        self._update_last_frame(
            observer_player_id=player_id,
            board_text=board_text,
            legal_moves=legal_moves,
            reward=reward,
            termination=termination,
            truncation=truncation,
            metadata=metadata,
            rgb_frame=rgb_frame,
        )
        view: dict[str, Any] = {"text": board_text}
        context = {
            "mode": "turn",
            "step": self._move_count,
            "tick": self._tick,
            "action_schema_config": self._action_schema_config(),
        }
        legal_actions = {"items": list(legal_moves)}
        prompt = self._prompt_builder.build(
            env_id=self._env_id,
            active_player=player_id,
            last_action=self._last_move,
            legal_moves=legal_moves,
            mode=str(context["mode"]),
            step=int(context["step"]),
            view_text=board_text,
            metadata=metadata,
        )
        return ArenaObservation(
            board_text=board_text,
            legal_moves=legal_moves,
            active_player=player_id,
            last_move=self._last_move,
            metadata=metadata,
            view=view,
            legal_actions=legal_actions,
            context=context,
            prompt=prompt,
        )

    def apply(self, action: ArenaAction) -> Optional[GameResult]:
        if self._final_result is not None:
            return self._final_result

        player_id = self._active_player_id
        if self._terminated or self._truncated:
            resolved_action = None
        else:
            try:
                resolved_action = self._codec.encode(action.move, action_space=self._env.action_space)
            except ValueError as exc:
                return self._handle_illegal(action, reason=str(exc) or "illegal_action")

        hold_ticks = self._resolve_action_hold_ticks(action)
        executed_ticks = 0
        total_reward = 0.0
        if resolved_action is None:
            executed_ticks = 1
        else:
            while executed_ticks < hold_ticks:
                reward_value = self._advance_one_tick(
                    action_id=resolved_action,
                    player_id=player_id,
                )
                total_reward += reward_value
                executed_ticks += 1
                if self._is_terminal_state():
                    break

        self._move_count += 1
        self._last_move = None if resolved_action is None else self._codec.decode(resolved_action)
        self._move_log.append(
            {
                "index": self._move_count,
                "player": player_id,
                "move": action.move,
                "action_id": resolved_action,
                "reward": total_reward,
                "termination": self._terminated,
                "truncation": self._truncated,
                "hold_ticks": max(1, executed_ticks),
            }
        )
        self._refresh_last_frame()

        if self._is_terminal_state():
            self._final_result = self.build_result(result="terminated", reason=self._resolve_reason())
            return self._final_result
        return None

    def tick_idle(self, *, frames: int = 1, move: str = "NOOP") -> Optional[GameResult]:
        """Advance the Gymnasium Atari backend without recording a player decision."""

        if self._final_result is not None:
            return self._final_result
        if self._is_terminal_state():
            self._final_result = self.build_result(result="terminated", reason=self._resolve_reason())
            return self._final_result

        player_id = self._active_player_id
        action_id = self._resolve_idle_action(move)
        target_ticks = max(1, int(frames))
        executed_ticks = 0
        while executed_ticks < target_ticks:
            self._advance_one_tick(action_id=action_id, player_id=player_id)
            executed_ticks += 1
            if self._is_terminal_state():
                break
        self._refresh_last_frame()

        if self._is_terminal_state():
            self._final_result = self.build_result(result="terminated", reason=self._resolve_reason())
            return self._final_result
        return None

    def is_terminal(self) -> bool:
        return self._final_result is not None or self._is_terminal_state()

    def build_result(self, *, result: str, reason: Optional[str]) -> GameResult:
        if self._final_result is not None:
            return self._final_result
        winner = self._resolve_winner()
        resolved_result = result
        if winner is not None and result not in {"loss", "draw"}:
            resolved_result = "win"
        self._final_result = self._materialize_replay(
            GameResult(
                winner=winner,
                result=resolved_result,
                reason=reason,
                move_count=self._move_count,
                illegal_move_count=sum(self._illegal_counts.values()),
                final_board=self._format_final_board(reason),
                move_log=list(self._move_log),
            )
        )
        return self._final_result

    def close(self) -> None:
        closer = getattr(self._env, "close", None)
        if callable(closer):
            try:
                closer()
            except Exception:
                pass

    def _build_env(self) -> Any:
        if self._backend_mode == "dummy":
            raise ValueError("gymnasium_atari does not provide a dummy backend")
        try:
            gym = importlib.import_module("gymnasium")
            try:
                ale_py = importlib.import_module("ale_py")
                register_envs = getattr(gym, "register_envs", None)
                if callable(register_envs):
                    register_envs(ale_py)
            except Exception:
                logger.debug("ale_py registration skipped", exc_info=True)
        except ModuleNotFoundError as exc:
            raise ValueError(
                "Gymnasium Atari backend requires gymnasium and ale_py to be installed"
            ) from exc
        env_kwargs = dict(self._env_kwargs)
        env_kwargs.setdefault("render_mode", "rgb_array")
        try:
            return gym.make(self._env_id, **env_kwargs)
        except Exception as exc:
            raise ValueError(
                f"Gymnasium Atari env '{self._env_id}' failed to initialize"
            ) from exc

    def _reset_env(self) -> tuple[Any, dict[str, Any]]:
        if self._seed is None:
            result = self._env.reset()
        else:
            try:
                result = self._env.reset(seed=self._seed)
            except TypeError:
                result = self._env.reset()
        if isinstance(result, tuple) and len(result) == 2:
            obs, info = result
        else:
            obs, info = result, {}
        return obs, info if isinstance(info, dict) else {}

    def _step_env(self, action_id: int) -> tuple[Any, float, bool, bool, dict[str, Any]]:
        result = self._env.step(action_id)
        if isinstance(result, tuple) and len(result) == 5:
            obs, reward, terminated, truncated, info = result
            return obs, reward, bool(terminated), bool(truncated), info
        if isinstance(result, tuple) and len(result) == 4:
            obs, reward, done, info = result
            return obs, reward, bool(done), False, info
        raise ValueError("Gymnasium env.step returned an unsupported transition shape")

    def _advance_one_tick(self, *, action_id: int, player_id: str) -> float:
        obs, reward, terminated, truncated, info = self._step_env(action_id)
        reward_value = _coerce_reward(reward)
        self._last_obs = obs
        self._last_reward = reward_value
        self._last_info = info if isinstance(info, dict) else {}
        self._terminated = bool(terminated)
        self._truncated = bool(truncated)
        self._tick += 1
        self._update_score(player_id, reward_value)
        self._maybe_render()
        if self._tick >= self._max_cycles:
            self._truncated = True
            self._last_info = {**self._last_info, "max_cycles": self._max_cycles}
        return reward_value

    def _resolve_idle_action(self, move: str) -> int:
        for candidate in (move, "NOOP", "noop", 0):
            try:
                return self._codec.encode(candidate, action_space=self._env.action_space)
            except ValueError:
                continue
        return 0

    def _maybe_render(self) -> None:
        render_mode = self._env_kwargs.get("render_mode")
        if render_mode is None:
            render_mode = getattr(self._env, "render_mode", None)
        if render_mode is None:
            render_mode = getattr(getattr(self._env, "unwrapped", None), "render_mode", None)
        if str(render_mode or "").lower() != "human":
            return
        render = getattr(self._env, "render", None)
        if callable(render):
            render()
        try:
            import pygame  # type: ignore

            if pygame.display.get_init():
                pygame.event.pump()
                if not self._render_logged:
                    self._render_logged = True
                    logger.info("Gymnasium Atari render initialized")
        except Exception:
            pass

    @staticmethod
    def _resolve_players(
        *,
        player_specs: Optional[Sequence[object]],
        player_ids: Optional[Sequence[str]],
    ) -> tuple[list[str], dict[str, str]]:
        if player_specs:
            ids = [str(getattr(player, "player_id")) for player in player_specs]
            names = {
                str(getattr(player, "player_id")): str(getattr(player, "display_name"))
                for player in player_specs
            }
            return ids or ["pilot_0"], names
        ids = [str(player_id) for player_id in player_ids or ("pilot_0",)]
        return ids or ["pilot_0"], {player_id: player_id for player_id in ids}

    def _refresh_action_labels(self) -> None:
        if self._explicit_action_labels or not self._use_action_meanings:
            return
        count = self._codec._resolve_action_count(self._env.action_space)
        candidate = None
        if hasattr(self._env, "get_action_meanings"):
            try:
                candidate = self._env.get_action_meanings()
            except Exception:
                candidate = None
        if candidate is None:
            unwrapped = getattr(self._env, "unwrapped", None)
            if unwrapped is not None and hasattr(unwrapped, "get_action_meanings"):
                try:
                    candidate = unwrapped.get_action_meanings()
                except Exception:
                    candidate = None
        if candidate is None:
            return
        labels = [str(label) for label in candidate]
        if len(labels) == count:
            self._codec = DiscreteActionCodec(labels)

    def _build_legal_moves(self, *, termination: bool, truncation: bool) -> list[str]:
        if termination or truncation:
            return []
        return self._codec.legal_moves(self._env.action_space)

    def _build_metadata(
        self,
        *,
        player_id: str,
        reward: float,
        termination: bool,
        truncation: bool,
    ) -> dict[str, Any]:
        metadata: dict[str, Any] = {
            "env_id": self._env_id,
            "player_id": player_id,
            "player_ids": list(self._player_ids),
            "player_names": dict(self._player_names),
            "reward": reward,
            "termination": termination,
            "truncation": truncation,
            "tick": self._tick,
            "last_move": self._last_move,
            "info": dict(self._last_info),
            "action_schema_config": self._action_schema_config(),
        }
        if self._include_raw_obs:
            metadata["raw_obs"] = self._last_obs
        else:
            metadata["obs_shape"] = getattr(self._last_obs, "shape", None)
        metadata["action_schema"] = self._format_action_schema(
            self._build_legal_moves(termination=termination, truncation=truncation)
        )
        return metadata

    def _refresh_last_frame(self) -> None:
        try:
            self._last_frame = self._build_frame_payload(observer_player_id=self._active_player_id)
        except Exception:
            self._last_frame = {
                "active_player_id": self._active_player_id,
                "observer_player_id": self._active_player_id,
                "env_id": self._env_id,
                "board_text": "",
                "legal_moves": [],
                "move_count": self._move_count,
                "last_move": self._last_move,
                "metadata": {},
            }

    def _update_last_frame(
        self,
        *,
        observer_player_id: str,
        board_text: str,
        legal_moves: Sequence[str],
        reward: float,
        termination: bool,
        truncation: bool,
        metadata: dict[str, Any],
        rgb_frame: Optional[Any],
    ) -> None:
        self._last_frame = {
            "active_player_id": self._active_player_id,
            "observer_player_id": str(observer_player_id),
            "env_id": self._env_id,
            "board_text": str(board_text),
            "legal_moves": list(legal_moves),
            "legal_actions": {"items": list(legal_moves)},
            "reward": reward,
            "termination": termination,
            "truncation": truncation,
            "move_count": self._move_count,
            "last_move": self._last_move,
            "metadata": dict(metadata),
            "scores": dict(self._scores),
        }
        if rgb_frame is not None:
            self._last_frame["_rgb"] = rgb_frame

    def _build_frame_payload(self, *, observer_player_id: str) -> dict[str, Any]:
        legal_moves = self._build_legal_moves(
            termination=self._terminated,
            truncation=self._truncated,
        )
        board_text = self._format_board_text(
            self._active_player_id,
            self._last_reward,
            self._terminated,
            self._truncated,
        )
        metadata = self._build_metadata(
            player_id=self._active_player_id,
            reward=self._last_reward,
            termination=self._terminated,
            truncation=self._truncated,
        )
        rgb_frame = self._resolve_frame_rgb(self._last_obs)
        frame_payload = {
            "active_player_id": self._active_player_id,
            "observer_player_id": str(observer_player_id),
            "env_id": self._env_id,
            "board_text": board_text,
            "legal_moves": list(legal_moves),
            "legal_actions": {"items": list(legal_moves)},
            "reward": self._last_reward,
            "termination": self._terminated,
            "truncation": self._truncated,
            "move_count": self._move_count,
            "last_move": self._last_move,
            "metadata": metadata,
            "scores": dict(self._scores),
        }
        if rgb_frame is not None:
            frame_payload["_rgb"] = rgb_frame
        return frame_payload

    def _resolve_frame_rgb(self, obs: Any) -> Optional[Any]:
        candidate = obs
        if isinstance(candidate, dict):
            for key in ("observation", "obs", "rgb", "rgb_array", "frame"):
                value = candidate.get(key)
                if value is not None:
                    candidate = value
                    break
        if self._is_rgb_like(candidate):
            return candidate
        render = getattr(self._env, "render", None)
        if callable(render):
            try:
                rendered = render()
            except Exception:
                rendered = None
            if self._is_rgb_like(rendered):
                return rendered
        return None

    @staticmethod
    def _is_rgb_like(value: Any) -> bool:
        shape = getattr(value, "shape", None)
        if shape is None:
            return False
        try:
            dims = tuple(int(dim) for dim in shape)
        except Exception:
            return False
        if len(dims) == 2:
            return True
        return len(dims) == 3 and dims[2] in {1, 3, 4}

    def _update_score(self, player_id: str, reward: float) -> None:
        self._scores[player_id] = self._scores.get(player_id, 0.0) + float(reward)

    def _is_terminal_state(self) -> bool:
        return bool(self._terminated or self._truncated)

    def _resolve_reason(self) -> str:
        if self._truncated:
            if self._tick >= self._max_cycles:
                return "max_cycles"
            return "truncated"
        if self._terminated:
            return "terminated"
        return "completed"

    def _resolve_winner(self) -> Optional[str]:
        score = self._scores.get(self._active_player_id, 0.0)
        if score > 0:
            return self._active_player_id
        return None

    def _format_board_text(
        self,
        player_id: str,
        reward: float,
        termination: bool,
        truncation: bool,
    ) -> str:
        return "\n".join(
            [
                f"Gymnasium Atari env: {self._env_id}",
                f"Active player: {player_id}",
                f"Step: {self._move_count}",
                f"Tick: {self._tick}",
                f"Reward: {reward}",
                f"Termination: {termination}",
                f"Truncation: {truncation}",
            ]
        )

    def _format_final_board(self, reason: Optional[str]) -> str:
        summary = f"Gymnasium Atari env finished after {self._move_count} moves and {self._tick} ticks."
        if reason:
            summary += f" Reason: {reason}."
        return summary

    def _handle_illegal(self, action: ArenaAction, *, reason: str) -> Optional[GameResult]:
        player = action.player
        if player in self._illegal_counts:
            self._illegal_counts[player] += 1
        if self._max_illegal < 0:
            return None
        if self._illegal_counts.get(player, 0) <= self._max_illegal:
            return None
        result = "draw" if self._illegal_on_fail == "draw" else "loss"
        self._final_result = self._materialize_replay(
            GameResult(
                winner=None,
                result=result,
                reason=reason,
                move_count=self._move_count,
                illegal_move_count=sum(self._illegal_counts.values()),
                final_board=self._format_final_board(reason),
                move_log=list(self._move_log),
            )
        )
        return self._final_result

    def _materialize_replay(self, result: GameResult) -> GameResult:
        if self._replay_game_kit is None or self._replay_env is None:
            return result
        materialized = materialize_gamekit_replay(
            result=result,
            game_kit=self._replay_game_kit,
            env=self._replay_env,
            run_id=self._run_id,
            sample_id=self._sample_id,
            replay_output_dir=self._replay_output_dir,
        )
        if materialized.arena_trace == result.arena_trace:
            return materialized
        return replace(materialized, arena_trace=result.arena_trace)

    def _resolve_action_hold_ticks(self, action: ArenaAction) -> int:
        metadata = action.metadata if isinstance(action.metadata, dict) else {}
        candidate = metadata.get("hold_ticks")
        if candidate is None:
            candidate = self._extract_hold_ticks_from_raw(action.raw)
        if candidate is None:
            hold_ticks = self._action_schema.default_hold_ticks
        else:
            try:
                hold_ticks = int(candidate)
            except (TypeError, ValueError):
                hold_ticks = self._action_schema.default_hold_ticks
        return max(
            self._action_schema.hold_ticks_min,
            min(self._action_schema.hold_ticks_max, hold_ticks),
        )

    @staticmethod
    def _extract_hold_ticks_from_raw(raw: Any) -> Optional[int]:
        if not isinstance(raw, str):
            return None
        stripped = raw.strip()
        if not stripped or not stripped.startswith("{"):
            return None
        try:
            payload = json.loads(stripped)
        except json.JSONDecodeError:
            return None
        if not isinstance(payload, dict):
            return None
        value = payload.get("hold_ticks")
        if value is None:
            value = payload.get("holdTicks")
        if value is None:
            return None
        try:
            return int(value)
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _build_action_schema(action_schema: Optional[dict[str, Any]]) -> ActionSchema:
        if not action_schema:
            return ActionSchema()
        hold_ticks_min = int(action_schema.get("hold_ticks_min", 1) or 1)
        default_hold_ticks = int(action_schema.get("hold_ticks_default", 1) or 1)
        hold_ticks_max = int(
            action_schema.get("hold_ticks_max", default_hold_ticks) or default_hold_ticks
        )
        hold_ticks_min = max(1, hold_ticks_min)
        hold_ticks_max = max(hold_ticks_min, hold_ticks_max)
        default_hold_ticks = max(hold_ticks_min, min(hold_ticks_max, default_hold_ticks))
        return ActionSchema(
            hold_ticks_min=hold_ticks_min,
            hold_ticks_max=hold_ticks_max,
            default_hold_ticks=default_hold_ticks,
        )

    def _action_schema_config(self) -> dict[str, int]:
        return {
            "hold_ticks_min": int(self._action_schema.hold_ticks_min),
            "hold_ticks_max": int(self._action_schema.hold_ticks_max),
            "hold_ticks_default": int(self._action_schema.default_hold_ticks),
        }

    def _format_action_schema(self, legal_moves: Sequence[str]) -> str:
        legal_hint = ", ".join(str(move) for move in legal_moves) or "none"
        return (
            '{ "move": "<legal_move>", "hold_ticks": <int> }\n'
            f"legal_move must be one of: {legal_hint}\n"
            f"hold_ticks range: {self._action_schema.hold_ticks_min}-"
            f"{self._action_schema.hold_ticks_max} "
            f"(default {self._action_schema.default_hold_ticks})."
        )


def _coerce_reward(value: Any) -> float:
    if value is None:
        return 0.0
    try:
        return float(value)
    except (TypeError, ValueError):
        item = getattr(value, "item", None)
        if callable(item):
            try:
                return float(item())
            except (TypeError, ValueError):
                pass
    return 0.0


def build_space_invaders_environment(
    *,
    sample,
    resolved,
    resources,
    player_specs,
    invocation_context=None,
) -> Any:
    defaults = {
        **dict(resolved.game_kit.defaults),
        **dict(resolved.env_spec.defaults),
        **dict(sample.runtime_overrides or {}),
    }
    env_kwargs = dict(defaults.get("env_kwargs") or {})
    env_id = str(env_kwargs.pop("env_id", defaults.get("env_id", "ALE/SpaceInvaders-v5")))
    run_id, sample_id = resolve_invocation_run_sample_ids(
        invocation_context=invocation_context,
        run_id=defaults.get("run_id"),
        sample_id=defaults.get("sample_id"),
    )
    raw_action_labels = defaults.get("action_labels")
    action_labels = None
    if raw_action_labels is not None:
        action_labels = tuple(str(label) for label in raw_action_labels)
    environment = SpaceInvadersGymEnvironment(
        env_id=env_id,
        env_kwargs=env_kwargs,
        player_specs=player_specs,
        seed=int(defaults["seed"]) if defaults.get("seed") is not None else None,
        action_labels=action_labels,
        use_action_meanings=bool(defaults.get("use_action_meanings", True)),
        include_raw_obs=bool(defaults.get("include_raw_obs", False)),
        illegal_policy=defaults.get("illegal_policy"),
        action_schema=(
            defaults.get("action_schema")
            if isinstance(defaults.get("action_schema"), dict)
            else None
        ),
        max_cycles=int(defaults.get("max_cycles", 1000)),
        backend_mode=str(defaults.get("backend_mode", "real")),
        replay_game_kit="gymnasium_atari",
        replay_env="space_invaders",
        replay_output_dir=defaults.get("replay_output_dir"),
        run_id=run_id,
        sample_id=sample_id,
    )
    return attach_runtime_resources(environment, resources)


__all__ = ["SpaceInvadersGymEnvironment", "build_space_invaders_environment"]
