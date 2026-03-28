from typing import Any, Dict, List

from gage_eval.role.arena.games.pettingzoo.env import PettingZooAecArenaEnvironment
from gage_eval.role.arena.types import ArenaAction


class DummyActionSpace:
    def __init__(self, n: int) -> None:
        self.n = n

    def contains(self, item: int) -> bool:
        return isinstance(item, int) and 0 <= item < self.n


class FakeAecEnv:
    def __init__(self, max_steps: int = 4, action_n: int = 3) -> None:
        self.possible_agents = ["agent_0", "agent_1"]
        self.agents = list(self.possible_agents)
        self._action_spaces = {agent: DummyActionSpace(action_n) for agent in self.agents}
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


class FakeRgbFrame:
    shape = (2, 2, 3)
    dtype = "uint8"

    def tobytes(self) -> bytes:
        return bytes([0, 64, 128, 255]) * 3


class FakeRgbAecEnv(FakeAecEnv):
    def observe(self, agent: str) -> Dict[str, Any]:
        return {"observation": FakeRgbFrame(), "agent": agent, "step": self._step_count}


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
    assert observation.prompt is not None
    assert "Legal moves:" in observation.prompt.instruction
    assert observation.prompt.payload.get("env_id") == ""

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


def test_pettingzoo_env_disable_action_meanings_uses_numeric_moves():
    env = FakeAecEnv(max_steps=2, action_n=6)
    adapter = PettingZooAecArenaEnvironment(
        env=env,
        env_id="pettingzoo.atari.space_invaders_v2",
        player_ids=["player_0", "player_1"],
        illegal_policy={"retry": 0, "on_fail": "loss"},
        use_action_meanings=False,
    )
    adapter.reset()
    active_player = adapter.get_active_player()
    observation = adapter.observe(active_player)
    assert observation.legal_moves[:3] == ["0", "1", "2"]
    assert "NOOP" not in observation.legal_moves
    assert observation.prompt is not None


def test_pettingzoo_env_observation_includes_inline_image_when_rgb_obs_available(monkeypatch):
    env = FakeRgbAecEnv(max_steps=2, action_n=6)
    adapter = PettingZooAecArenaEnvironment(
        env=env,
        env_id="pettingzoo.atari.space_invaders_v2",
        player_ids=["player_0", "player_1"],
        illegal_policy={"retry": 0, "on_fail": "loss"},
    )
    monkeypatch.setattr(
        adapter,
        "_build_image_data_url",
        lambda frame: "data:image/png;base64,Zm9v",
    )

    adapter.reset()
    observation = adapter.observe(adapter.get_active_player())

    assert observation.view is not None
    assert observation.view["image"]["data_url"] == "data:image/png;base64,Zm9v"
    assert observation.view["image"]["shape"] == [2, 2, 3]
