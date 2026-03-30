from __future__ import annotations

from typing import Any, Sequence

from loguru import logger

from gage_eval.game_kits.aec_env_game.pettingzoo.environment import (
    PettingZooAecArenaEnvironment,
)
from gage_eval.game_kits.real_time_game.backend_mode import normalize_backend_mode
from gage_eval.role.arena.replay_paths import resolve_invocation_run_sample_ids
from gage_eval.role.arena.resources.runtime_bridge import attach_runtime_resources

try:
    import numpy as np
except Exception:  # pragma: no cover - optional dependency
    np = None


class _DiscreteActionSpace:
    def __init__(self, n: int) -> None:
        self.n = n

    def contains(self, item: int) -> bool:
        return isinstance(item, int) and 0 <= item < self.n


class _StubSpaceInvadersAecEnv:
    possible_agents = ("player_0", "player_1")

    def __init__(self, *, max_cycles: int = 4) -> None:
        self._max_cycles = max(1, int(max_cycles))
        self._action_spaces = {
            agent: _DiscreteActionSpace(6) for agent in self.possible_agents
        }
        self.render_mode = None
        self.reset()

    def reset(self, seed: Any = None, options: Any = None) -> None:
        del seed, options
        self.agents = list(self.possible_agents)
        self.agent_selection = self.agents[0]
        self._step_count = 0
        self.rewards = {agent: 0.0 for agent in self.agents}
        self.terminations = {agent: False for agent in self.agents}
        self.truncations = {agent: False for agent in self.agents}
        self.infos = {
            agent: {"action_mask": [1, 1, 1, 1, 1, 1]} for agent in self.agents
        }

    def action_space(self, agent: str) -> _DiscreteActionSpace:
        return self._action_spaces[agent]

    def observe(self, agent: str) -> dict[str, Any]:
        if np is None:
            frame = [[self._step_count, len(agent)]]
        else:
            frame = np.zeros((84, 84, 3), dtype=np.uint8)
            frame[:, :, 0] = (self._step_count * 32) % 256
            frame[:, :, 1] = 96 if agent == "player_0" else 180
            frame[20:64, 16:68, 2] = 210
            frame[40:44, :, :] = 255
        return {
            "agent": agent,
            "step": self._step_count,
            "frame": frame,
        }

    def last(self):
        agent = self.agent_selection
        return (
            self.observe(agent),
            self.rewards[agent],
            self.terminations[agent],
            self.truncations[agent],
            self.infos[agent],
        )

    def step(self, action: Any) -> None:
        agent = self.agent_selection
        if action is not None and not self.action_space(agent).contains(action):
            raise ValueError(f"Illegal action {action!r} for {agent}")
        if self.terminations[agent] or self.truncations[agent]:
            return

        self._step_count += 1
        self.rewards = {player: 0.0 for player in self.agents}
        if self._step_count == 2:
            self.rewards["player_0"] = 1.0
        for player in self.agents:
            self.infos[player] = {
                "action_mask": [1, 1, 1, 1, 1, 1],
                "step": self._step_count,
            }
        if self._step_count >= self._max_cycles:
            for player in self.agents:
                self.terminations[player] = True

        current_index = self.agents.index(agent)
        self.agent_selection = self.agents[(current_index + 1) % len(self.agents)]

    def close(self) -> None:
        return None


class SpaceInvadersEnvironment:
    def __init__(
        self,
        *,
        env_id: str,
        max_cycles: int,
        seed: int | None,
        use_action_meanings: bool,
        include_raw_obs: bool,
        illegal_policy: dict[str, str | int] | None,
        action_labels: Sequence[str] | None,
        player_specs: Sequence[object],
        backend_mode: str = "auto",
        env_kwargs: dict[str, Any] | None = None,
        replay_output_dir: str | None = None,
        run_id: str | None = None,
        sample_id: str | None = None,
    ) -> None:
        player_ids = [str(getattr(player, "player_id")) for player in player_specs]
        player_names = {
            str(getattr(player, "player_id")): str(getattr(player, "display_name"))
            for player in player_specs
        }
        resolved_backend_mode = normalize_backend_mode(backend_mode)
        adapter_kwargs = {
            "player_ids": player_ids,
            "player_names": player_names,
            "seed": seed,
            "action_labels": action_labels,
            "use_action_meanings": use_action_meanings,
            "include_raw_obs": include_raw_obs,
            "illegal_policy": illegal_policy,
            "replay_game_kit": "pettingzoo",
            "replay_env": "space_invaders",
            "replay_output_dir": replay_output_dir,
            "run_id": run_id,
            "sample_id": sample_id,
        }
        if resolved_backend_mode == "dummy":
            self._adapter = PettingZooAecArenaEnvironment(
                env=_StubSpaceInvadersAecEnv(max_cycles=max_cycles),
                env_id=env_id,
                **adapter_kwargs,
            )
            return

        resolved_env_kwargs = dict(env_kwargs or {})
        resolved_env_kwargs.setdefault("render_mode", "rgb_array")
        resolved_env_kwargs.setdefault("max_cycles", max_cycles)
        try:
            self._adapter = PettingZooAecArenaEnvironment(
                env_id=env_id,
                env_kwargs=resolved_env_kwargs,
                **adapter_kwargs,
            )
        except Exception:
            if resolved_backend_mode == "real":
                raise
            logger.warning(
                "SpaceInvadersEnvironment falling back to stub backend for env_id={}",
                env_id,
            )
            self._adapter = PettingZooAecArenaEnvironment(
                env=_StubSpaceInvadersAecEnv(max_cycles=max_cycles),
                env_id=env_id,
                **adapter_kwargs,
            )

    @classmethod
    def from_runtime(cls, *, sample, resolved, resources, player_specs, invocation_context=None):
        defaults = {
            **dict(resolved.game_kit.defaults),
            **dict(resolved.env_spec.defaults),
            **dict(sample.runtime_overrides or {}),
        }
        run_id, sample_id = resolve_invocation_run_sample_ids(
            invocation_context=invocation_context,
            run_id=defaults.get("run_id"),
            sample_id=defaults.get("sample_id"),
        )
        raw_action_labels = defaults.get("action_labels")
        action_labels = None
        if raw_action_labels is not None:
            action_labels = tuple(str(label) for label in raw_action_labels)
        environment = cls(
            env_id=str(defaults.get("env_id", "pettingzoo.atari.space_invaders_v2")),
            max_cycles=int(defaults.get("max_cycles", 4)),
            seed=int(defaults["seed"]) if defaults.get("seed") is not None else None,
            use_action_meanings=bool(defaults.get("use_action_meanings", True)),
            include_raw_obs=bool(defaults.get("include_raw_obs", False)),
            illegal_policy=defaults.get("illegal_policy"),
            action_labels=action_labels,
            player_specs=player_specs,
            backend_mode=str(defaults.get("backend_mode", "auto")),
            env_kwargs=defaults.get("env_kwargs"),
            replay_output_dir=defaults.get("replay_output_dir"),
            run_id=run_id,
            sample_id=sample_id,
        )
        return attach_runtime_resources(environment, resources)

    def get_active_player(self) -> str:
        return self._adapter.get_active_player()

    def observe(self, player: str):
        return self._adapter.observe(player)

    def apply(self, action):
        return self._adapter.apply(action)

    def get_last_frame(self):
        return self._adapter.get_last_frame()

    def is_terminal(self) -> bool:
        return self._adapter.is_terminal()

    def build_result(self, *, result: str, reason: str | None):
        return self._adapter.build_result(result=result, reason=reason)

    def close(self) -> None:
        closer = getattr(self._adapter, "close", None)
        if callable(closer):
            closer()


def build_space_invaders_environment(
    *,
    sample,
    resolved,
    resources,
    player_specs,
    invocation_context=None,
) -> Any:
    return SpaceInvadersEnvironment.from_runtime(
        sample=sample,
        resolved=resolved,
        resources=resources,
        player_specs=player_specs,
        invocation_context=invocation_context,
    )
