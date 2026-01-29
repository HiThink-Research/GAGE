from typing import Any, Dict, List

from gage_eval.role.arena.games.pettingzoo.env import PettingZooAecArenaEnvironment
from gage_eval.role.arena.types import ArenaAction


class DummyActionSpace:
    def __init__(self, n: int) -> None:
        self.n = n

    def contains(self, item: int) -> bool:
        return isinstance(item, int) and 0 <= item < self.n


class FakeAecEnv:
    def __init__(self, max_steps: int = 4) -> None:
        self.possible_agents = ["agent_0", "agent_1"]
        self.agents = list(self.possible_agents)
        self._action_spaces = {agent: DummyActionSpace(3) for agent in self.agents}
        self._max_steps = max_steps
        self._step_count = 0
        self.reset()

    def reset(self, seed: Any = None, options: Any = None) -> None:
        self._step_count = 0
        self.agent_selection = self.agents[0]
        self.terminations = {agent: False for agent in self.agents}
        self.truncations = {agent: False for agent in self.agents}
        self.rewards = {agent: 0.0 for agent in self.agents}
        self.infos = {agent: {} for agent in self.agents}

    def action_space(self, agent: str) -> DummyActionSpace:
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
        agent = self.agent_selection
        if not self.terminations[agent] and not self.truncations[agent]:
            self.rewards[agent] = 1.0
        self._step_count += 1
        if self._step_count >= self._max_steps:
            for agent_id in self.agents:
                self.terminations[agent_id] = True
        idx = self.agents.index(agent)
        self.agent_selection = self.agents[(idx + 1) % len(self.agents)]


def test_pettingzoo_env_basic_loop():
    env = FakeAecEnv(max_steps=4)
    adapter = PettingZooAecArenaEnvironment(
        env=env,
        player_ids=["player_0", "player_1"],
        illegal_policy={"retry": 0, "on_fail": "loss"},
    )
    adapter.reset()
    active_player = adapter.get_active_player()
    assert active_player == "player_0"

    observation = adapter.observe(active_player)
    assert observation.legal_moves == ["0", "1", "2"]

    result = adapter.apply(ArenaAction(player=active_player, move="1", raw="1"))
    assert result is None

    for _ in range(10):
        active_player = adapter.get_active_player()
        adapter.observe(active_player)
        result = adapter.apply(ArenaAction(player=active_player, move="0", raw="0"))
        if result is not None:
            break

    assert result is not None
    assert result.move_count >= 1
    assert len(result.move_log) == result.move_count


def test_pettingzoo_env_illegal_action_loss():
    env = FakeAecEnv(max_steps=2)
    adapter = PettingZooAecArenaEnvironment(
        env=env,
        player_ids=["player_0", "player_1"],
        illegal_policy={"retry": 0, "on_fail": "loss"},
    )
    adapter.reset()
    active_player = adapter.get_active_player()
    adapter.observe(active_player)
    result = adapter.apply(ArenaAction(player=active_player, move="99", raw="99"))
    assert result is not None
    assert result.result in {"loss", "draw"}
