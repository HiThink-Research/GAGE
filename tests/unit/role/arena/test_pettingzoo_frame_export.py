from __future__ import annotations

from typing import Any, Dict

from gage_eval.game_kits.aec_env_game.pettingzoo.environment import (
    PettingZooAecArenaEnvironment,
)
from gage_eval.role.arena.types import ArenaAction


class _DummyActionSpace:
    def __init__(self, n: int) -> None:
        self.n = n

    def contains(self, item: int) -> bool:
        return isinstance(item, int) and 0 <= item < self.n


class _FakeAecEnv:
    def __init__(self, *, max_steps: int = 4, action_n: int = 3) -> None:
        self.possible_agents = ["agent_0", "agent_1"]
        self.agents = list(self.possible_agents)
        self._action_spaces = {agent: _DummyActionSpace(action_n) for agent in self.agents}
        self._max_steps = max_steps
        self._step_count = 0
        self.reset()

    def reset(self, seed: Any = None, options: Any = None) -> None:
        _ = (seed, options)
        self._step_count = 0
        self.agent_selection = self.agents[0]
        self.terminations = {agent: False for agent in self.agents}
        self.truncations = {agent: False for agent in self.agents}
        self.rewards = {agent: 0.0 for agent in self.agents}
        self.infos = {agent: {} for agent in self.agents}

    def action_space(self, agent: str) -> _DummyActionSpace:
        return self._action_spaces[agent]

    def observe(self, agent: str) -> Dict[str, Any]:
        return {"agent": agent, "step": self._step_count}

    def last(self):
        agent = self.agent_selection
        obs = self.observe(agent)
        reward = self.rewards[agent]
        termination = self.terminations[agent]
        truncation = self.truncations[agent]
        info = self.infos[agent]
        return obs, reward, termination, truncation, info

    def step(self, action: Any) -> None:
        _ = action
        agent = self.agent_selection
        if not self.terminations[agent] and not self.truncations[agent]:
            self.rewards[agent] = 1.0
        self._step_count += 1
        if self._step_count >= self._max_steps:
            for agent_id in self.agents:
                self.terminations[agent_id] = True
        idx = self.agents.index(agent)
        self.agent_selection = self.agents[(idx + 1) % len(self.agents)]


def test_pettingzoo_arena_exposes_get_last_frame() -> None:
    env = _FakeAecEnv(max_steps=4, action_n=3)
    adapter = PettingZooAecArenaEnvironment(
        env=env,
        env_id="pettingzoo.mock.env_v1",
        player_ids=["player_0", "player_1"],
        illegal_policy={"retry": 0, "on_fail": "loss"},
        use_action_meanings=False,
    )

    initial_frame = adapter.get_last_frame()
    assert isinstance(initial_frame, dict)
    assert initial_frame["active_player_id"] == "player_0"
    assert initial_frame["move_count"] == 0
    assert initial_frame["legal_moves"] == ["0", "1", "2"]

    active_player = adapter.get_active_player()
    adapter.observe(active_player)
    observed_frame = adapter.get_last_frame()
    assert observed_frame["observer_player_id"] == "player_0"

    adapter.apply(ArenaAction(player=active_player, move="1", raw="1"))
    advanced_frame = adapter.get_last_frame()
    assert advanced_frame["active_player_id"] == "player_1"
    assert advanced_frame["move_count"] == 1
