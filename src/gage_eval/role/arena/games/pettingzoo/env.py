"""PettingZoo AEC environment adapter for arena games."""

from __future__ import annotations

import importlib
from typing import Any, Dict, Optional, Sequence

from loguru import logger

from gage_eval.registry import registry
from gage_eval.role.arena.parsers.pettingzoo_actions import DiscreteActionCodec
from gage_eval.role.arena.types import ArenaAction, ArenaObservation, GameResult


from .constants import ACTION_MEANINGS

@registry.asset(
    "arena_impls",
    "pettingzoo_aec_v1",
    desc="PettingZoo AEC environment adapter",
    tags=("pettingzoo", "arena"),
)
class PettingZooAecArenaEnvironment:
    """Arena adapter that wraps a PettingZoo AEC environment."""

    def __init__(
        self,
        *,
        env_id: Optional[str] = None,
        env_kwargs: Optional[Dict[str, Any]] = None,
        env: Optional[Any] = None,
        player_ids: Optional[Sequence[str]] = None,
        player_names: Optional[Dict[str, str]] = None,
        agent_map: Optional[Dict[str, str]] = None,
        seed: Optional[int] = None,
        action_labels: Optional[Sequence[str]] = None,
        use_action_meanings: bool = True,
        include_raw_obs: bool = False,
        illegal_policy: Optional[Dict[str, str | int]] = None,
        board_size: Optional[int] = None,
        win_len: Optional[int] = None,
        token_map: Optional[Dict[str, str]] = None,
        start_player_id: Optional[str] = None,
        coord_scheme: str = "A1",
        rule_profile: Optional[str] = None,
        win_directions: Optional[Sequence[str]] = None,
    ) -> None:
        """Initialize the PettingZoo adapter.

        Args:
            env_id: Import path for the PettingZoo environment module.
            env_kwargs: Keyword arguments passed to the PettingZoo env factory.
            env: Optional pre-built PettingZoo environment (used in tests).
            player_ids: Optional player identifiers to map onto PettingZoo agents.
            player_names: Optional mapping from player_id to display names.
            agent_map: Optional mapping from PettingZoo agent id to player_id.
            seed: Optional seed passed to env.reset.
            action_labels: Optional list of action labels (index aligned with action id).
            use_action_meanings: Whether to derive labels from env when available.
            include_raw_obs: Whether to store raw observations in metadata.
            illegal_policy: Policy for handling illegal actions.
            board_size: Unused (kept for ArenaRoleAdapter compatibility).
            win_len: Unused (kept for ArenaRoleAdapter compatibility).
            token_map: Unused (kept for ArenaRoleAdapter compatibility).
            start_player_id: Unused (kept for ArenaRoleAdapter compatibility).
            coord_scheme: Unused (kept for ArenaRoleAdapter compatibility).
            rule_profile: Unused (kept for ArenaRoleAdapter compatibility).
            win_directions: Unused (kept for ArenaRoleAdapter compatibility).
        """

        _ = (
            board_size,
            win_len,
            token_map,
            start_player_id,
            coord_scheme,
            rule_profile,
            win_directions,
        )
        self._env_id = env_id
        self._env_kwargs = dict(env_kwargs or {})
        self._seed = seed
        self._include_raw_obs = bool(include_raw_obs)
        self._use_action_meanings = bool(use_action_meanings)
        
        # Resolve action labels: Explicit > Constant > Env-derived
        self._explicit_action_labels = [str(label) for label in action_labels or []]
        if not self._explicit_action_labels and self._env_id and self._use_action_meanings:
            # Try to match game name from env_id (e.g. pettingzoo.atari.boxing_v2 -> boxing)
            # Strategy: check if any constant key is present in the env_id string
            for game_key, meanings in ACTION_MEANINGS.items():
                if f".{game_key}_" in self._env_id or f".{game_key}" in self._env_id:
                    self._explicit_action_labels = list(meanings)
                    break

        self._codec = DiscreteActionCodec(self._explicit_action_labels or None)
        self._agent_map_override = dict(agent_map or {})
        self._provided_player_ids = [str(pid) for pid in player_ids or []]
        self._player_names = dict(player_names or {})
        self._illegal_policy = dict(illegal_policy or {})
        self._max_illegal = int(self._illegal_policy.get("retry", 0))
        self._illegal_on_fail = str(self._illegal_policy.get("on_fail", "loss"))

        if env is not None:
            self._env = env
        else:
            self._env = self._build_env(self._env_id, self._env_kwargs)

        self._agent_to_player: Dict[str, str] = {}
        self._player_to_agent: Dict[str, str] = {}
        self._player_ids: list[str] = []
        self._illegal_counts: Dict[str, int] = {}
        self._scores: Dict[str, float] = {}
        self._move_log: list[Dict[str, Any]] = []
        self._move_count = 0
        self._last_move: Optional[str] = None
        self._last_transition: Optional[Dict[str, Any]] = None
        self._final_result: Optional[GameResult] = None
        self._render_logged = False

        self.reset()

    def reset(self) -> None:
        """Reset the underlying PettingZoo environment.

        Returns:
            None.
        """

        # STEP 1: Reset the environment and render once if requested.
        self._reset_env()
        self._maybe_render()
        self._sync_agents()

        # STEP 2: Clear episode-level counters and caches.
        self._illegal_counts = {player_id: 0 for player_id in self._player_ids}
        self._scores = {player_id: 0.0 for player_id in self._player_ids}
        self._move_log = []
        self._move_count = 0
        self._last_move = None
        self._last_transition = None
        self._final_result = None

    def get_active_player(self) -> str:
        """Return the player_id for the current PettingZoo agent.

        Returns:
            Player identifier string.
        """
        agent = getattr(self._env, "agent_selection", None)
        if agent is None:
            return self._player_ids[0] if self._player_ids else "player_0"
        return self._agent_to_player.get(str(agent), str(agent))

    def observe(self, player: str) -> ArenaObservation:
        """Return the active observation for the specified player.

        Args:
            player: Player identifier (unused when PettingZoo controls the active agent).

        Returns:
            Unified arena observation.
        """

        # STEP 0: Render and pump pygame events to keep window responsive and updated.
        self._maybe_render()

        agent = getattr(self._env, "agent_selection", None)
        agent_id = str(agent) if agent is not None else ""
        player_id = self._agent_to_player.get(agent_id, agent_id or player)
        obs, reward, termination, truncation, info = self._capture_last()
        self._update_score(player_id, reward)
        legal_moves = self._build_legal_moves(agent_id, info, termination, truncation)
        board_text = self._format_board_text(player_id, reward, termination, truncation)
        action_mask = self._extract_action_mask(info)
        metadata: Dict[str, Any] = {
            "env_id": self._env_id,
            "agent_id": agent_id,
            "player_id": player_id,
            "player_ids": list(self._player_ids),
            "player_names": dict(self._player_names),
            "reward": reward,
            "termination": termination,
            "truncation": truncation,
            "action_mask": action_mask,
            "last_move": self._last_move,
        }
        if self._include_raw_obs:
            metadata["raw_obs"] = obs
        else:
            metadata["obs_shape"] = getattr(obs, "shape", None)
        if isinstance(info, dict):
            metadata["info"] = info
        view = {"text": board_text}
        legal_actions: Dict[str, Any] = {"items": list(legal_moves)}
        if action_mask is not None:
            legal_actions["mask"] = action_mask
        context = {"mode": "turn", "step": self._move_count}
        return ArenaObservation(
            board_text=board_text,
            legal_moves=legal_moves,
            active_player=player_id,
            last_move=self._last_move,
            metadata=metadata,
            view=view,
            legal_actions=legal_actions,
            context=context,
        )

    def apply(self, action: ArenaAction) -> Optional[GameResult]:
        """Apply an arena action and advance the PettingZoo environment.

        Args:
            action: ArenaAction produced by a player.

        Returns:
            GameResult if the environment terminates, otherwise None.
        """

        if self._final_result is not None:
            return self._final_result

        agent = getattr(self._env, "agent_selection", None)
        agent_id = str(agent) if agent is not None else ""
        player_id = self._agent_to_player.get(agent_id, agent_id or action.player)
        if self._last_transition is None:
            self._capture_last()
        transition = self._last_transition or {}
        termination = bool(transition.get("termination"))
        truncation = bool(transition.get("truncation"))
        info = transition.get("info")
        action_space = self._action_space(agent_id)
        action_mask = self._extract_action_mask(info)

        # STEP 1: Resolve the action id to send to PettingZoo.
        if termination or truncation:
            resolved_action = None
        else:
            try:
                resolved_action = self._codec.encode(
                    action.move,
                    action_space=action_space,
                    action_mask=action_mask,
                )
            except ValueError as exc:
                return self._handle_illegal(action, reason=str(exc) or "illegal_action")

        # STEP 2: Advance the environment and record the step.
        self._env.step(resolved_action)
        self._maybe_render()
        self._move_count += 1
        self._last_move = None if resolved_action is None else self._codec.decode(resolved_action)
        self._move_log.append(
            {
                "index": self._move_count,
                "player": player_id,
                "agent": agent_id,
                "move": action.move,
                "action_id": resolved_action,
                "reward": transition.get("reward"),
                "termination": termination,
                "truncation": truncation,
            }
        )

        # STEP 3: Emit final result when all agents are done.
        if self._is_terminal_state():
            self._final_result = self.build_result(result="terminated", reason=self._resolve_reason())
            return self._final_result

        return None

    def is_terminal(self) -> bool:
        """Return True if the PettingZoo environment is terminated.

        Returns:
            True if the environment is finished.
        """

        return self._final_result is not None or self._is_terminal_state()

    def build_result(self, *, result: str, reason: Optional[str]) -> GameResult:
        """Build a GameResult snapshot for the current episode.

        Args:
            result: Result label (e.g., win/loss/draw/terminated).
            reason: Optional reason for termination.

        Returns:
            GameResult summary.
        """

        if self._final_result is not None:
            return self._final_result
        winner = self._resolve_winner()
        resolved_result = result
        if winner is not None and result not in {"loss", "draw"}:
            resolved_result = "win"
        final_board = self._format_final_board(reason)
        return GameResult(
            winner=winner,
            result=resolved_result,
            reason=reason,
            move_count=self._move_count,
            illegal_move_count=sum(self._illegal_counts.values()),
            final_board=final_board,
            move_log=list(self._move_log),
        )

    def close(self) -> None:
        """Clean up resources when the environment is no longer needed."""
        if hasattr(self._env, "close"):
            try:
                self._env.close()
            except Exception:
                pass

    def _build_env(self, env_id: Optional[str], env_kwargs: Dict[str, Any]) -> Any:
        if not env_id:
            raise ValueError("PettingZoo adapter requires env_id or env instance")
        try:
            factory = self._resolve_env_callable(env_id)
        except ModuleNotFoundError as exc:
            raise ValueError(
                "PettingZoo environment import failed. Install pettingzoo[atari] and required ROMs."
            ) from exc
        try:
            return factory(**env_kwargs)
        except TypeError:
            return factory()

    def _resolve_env_callable(self, env_id: str):
        if ":" in env_id:
            module_name, attr = env_id.split(":", 1)
            module = importlib.import_module(module_name)
            return getattr(module, attr)
        try:
            module = importlib.import_module(env_id)
            if hasattr(module, "env"):
                return getattr(module, "env")
            if hasattr(module, "parallel_env"):
                return getattr(module, "parallel_env")
        except ModuleNotFoundError:
            if "." not in env_id:
                raise
            module_name, attr = env_id.rsplit(".", 1)
            module = importlib.import_module(module_name)
            return getattr(module, attr)
        raise ValueError(f"PettingZoo env factory not found for '{env_id}'")

    def _reset_env(self) -> None:
        if self._seed is None:
            try:
                self._env.reset()
            except TypeError:
                self._env.reset()
            return
        try:
            self._env.reset(seed=self._seed)
        except TypeError:
            self._env.reset()

    def _maybe_render(self) -> None:
        render_mode = self._env_kwargs.get("render_mode")
        if render_mode is None:
            render_mode = getattr(self._env, "render_mode", None)
        if render_mode is None:
            render_mode = getattr(getattr(self._env, "unwrapped", None), "render_mode", None)
        if render_mode is None or str(render_mode).lower() != "human":
            return
        if not hasattr(self._env, "render"):
            return
        # NOTE: Some PettingZoo envs do not auto-render on step; render once to ensure a window appears.
        self._env.render()
        try:
            import pygame  # type: ignore

            if pygame.display.get_init():
                pygame.event.pump()
                if not self._render_logged:
                    self._render_logged = True
                    logger.info(
                        "PettingZoo render initialized (render_mode={}, driver={})",
                        render_mode,
                        pygame.display.get_driver(),
                    )
        except Exception as exc:
            if not self._render_logged:
                self._render_logged = True
                logger.warning("PettingZoo render init check failed: {}", exc)

    def _pump_display_events(self) -> None:
        """Pump pygame events to prevent window freezing during slow LLM calls.

        This is a workaround for synchronous TurnScheduler blocking the main thread
        while waiting for LLM responses. In the future, consider using TickScheduler
        with async player interface for proper event handling.
        """
        try:
            import pygame  # type: ignore

            if pygame.display.get_init():
                pygame.event.pump()
        except Exception:
            # NOTE: Silently ignore if pygame is not available or not initialized.
            pass

    def _sync_agents(self) -> None:
        agents = list(getattr(self._env, "possible_agents", None) or getattr(self._env, "agents", []) or [])
        if not agents:
            raise ValueError("PettingZoo env returned empty agent list")

        if self._agent_map_override:
            self._agent_to_player = {str(agent): str(self._agent_map_override[agent]) for agent in agents}
        elif self._provided_player_ids:
            if len(self._provided_player_ids) != len(agents):
                raise ValueError("player_ids length must match PettingZoo agents")
            self._agent_to_player = {
                str(agent): str(self._provided_player_ids[idx]) for idx, agent in enumerate(agents)
            }
        else:
            self._agent_to_player = {str(agent): str(agent) for agent in agents}

        self._player_to_agent = {player: agent for agent, player in self._agent_to_player.items()}
        self._player_ids = [self._agent_to_player[str(agent)] for agent in agents]
        for player_id in self._player_ids:
            self._player_names.setdefault(player_id, player_id)

        if not self._explicit_action_labels and self._use_action_meanings:
            self._refresh_action_labels(agents)

    def _refresh_action_labels(self, agents: Sequence[str]) -> None:
        action_space = self._action_space(str(agents[0]))
        count = self._codec._resolve_action_count(action_space)
        labels = self._extract_action_meanings(count)
        if labels:
            self._codec = DiscreteActionCodec(labels)

    def _extract_action_meanings(self, count: int) -> Optional[list[str]]:
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
            return None
        labels = [str(label) for label in candidate]
        if len(labels) != count:
            return None
        return labels

    def _capture_last(self) -> tuple[Any, float, bool, bool, Dict[str, Any]]:
        obs, reward, termination, truncation, info = self._env.last()
        transition = {
            "observation": obs,
            "reward": reward,
            "termination": termination,
            "truncation": truncation,
            "info": info,
        }
        self._last_transition = transition
        return obs, reward, bool(termination), bool(truncation), info

    def _action_space(self, agent: str) -> Any:
        if hasattr(self._env, "action_space"):
            space = self._env.action_space
            if callable(space):
                return space(agent)
            if isinstance(space, dict):
                return space[agent]
            return space
        raise ValueError("PettingZoo env does not expose action_space")

    def _build_legal_moves(
        self,
        agent: str,
        info: Optional[Dict[str, Any]],
        termination: bool,
        truncation: bool,
    ) -> list[str]:
        if termination or truncation:
            return []
        action_space = self._action_space(agent)
        action_mask = self._extract_action_mask(info)
        return self._codec.legal_moves(action_space, action_mask=action_mask)

    def _extract_action_mask(self, info: Optional[Dict[str, Any]]) -> Optional[Sequence[int]]:
        if not isinstance(info, dict):
            return None
        mask = info.get("action_mask")
        if mask is None:
            return None
        try:
            return list(mask)
        except TypeError:
            return None

    def _update_score(self, player_id: str, reward: float) -> None:
        if player_id not in self._scores:
            self._scores[player_id] = 0.0
        try:
            self._scores[player_id] += float(reward)
        except (TypeError, ValueError):
            return

    def _is_terminal_state(self) -> bool:
        terminations = getattr(self._env, "terminations", None)
        truncations = getattr(self._env, "truncations", None)
        if isinstance(terminations, dict) and terminations:
            if all(bool(value) for value in terminations.values()):
                return True
        if isinstance(truncations, dict) and truncations:
            if all(bool(value) for value in truncations.values()):
                return True
        return False

    def _resolve_reason(self) -> str:
        truncations = getattr(self._env, "truncations", None)
        terminations = getattr(self._env, "terminations", None)
        if isinstance(truncations, dict) and any(truncations.values()):
            return "truncated"
        if isinstance(terminations, dict) and any(terminations.values()):
            return "terminated"
        return "completed"

    def _resolve_winner(self) -> Optional[str]:
        if not self._scores:
            return None
        best_score = None
        winner = None
        for player_id, score in self._scores.items():
            if best_score is None or score > best_score:
                best_score = score
                winner = player_id
            elif best_score is not None and score == best_score:
                winner = None
        return winner

    def _format_board_text(
        self,
        player_id: str,
        reward: float,
        termination: bool,
        truncation: bool,
    ) -> str:
        lines = [
            f"PettingZoo env: {self._env_id}",
            f"Active player: {player_id}",
            f"Step: {self._move_count}",
            f"Reward: {reward}",
            f"Termination: {termination}",
            f"Truncation: {truncation}",
        ]
        return "\n".join(lines)

    def _format_final_board(self, reason: Optional[str]) -> str:
        summary = f"PettingZoo env finished after {self._move_count} steps."
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

        if self._illegal_on_fail == "draw":
            winner = None
            result = "draw"
        else:
            winner = self._other_player(player)
            result = "loss"

        self._final_result = GameResult(
            winner=winner,
            result=result,
            reason=reason,
            move_count=self._move_count,
            illegal_move_count=sum(self._illegal_counts.values()),
            final_board=self._format_final_board(reason),
            move_log=list(self._move_log),
        )
        return self._final_result

    def _other_player(self, player: str) -> Optional[str]:
        for candidate in self._player_ids:
            if candidate != player:
                return candidate
        return None
