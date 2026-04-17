from typing import Any, Dict

from gage_eval.game_kits.aec_env_game.pettingzoo.action_codec import (
    DiscreteActionCodec,
)
from gage_eval.game_kits.aec_env_game.pettingzoo.environment import (
    PettingZooAecArenaEnvironment,
)
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


class RecordingAecEnv(FakeAecEnv):
    def __init__(self, max_steps: int = 4, action_n: int = 3) -> None:
        self.actions: list[tuple[str, Any]] = []
        super().__init__(max_steps=max_steps, action_n=action_n)

    def reset(self, seed: Any = None, options: Any = None) -> None:
        super().reset(seed=seed, options=options)
        self.actions = []

    def step(self, action: Any) -> None:
        self.actions.append((str(self.agent_selection), action))
        super().step(action)


class FakeRgbFrame:
    shape = (2, 2, 3)
    dtype = "uint8"

    def tobytes(self) -> bytes:
        return bytes([0, 64, 128, 255]) * 3


class FakeRgbAecEnv(FakeAecEnv):
    def observe(self, agent: str) -> Dict[str, Any]:
        return {"observation": FakeRgbFrame(), "agent": agent, "step": self._step_count}


class FakeStaleLastAecEnv(FakeAecEnv):
    def __init__(self) -> None:
        super().__init__(max_steps=4, action_n=3)
        self._last_obs_step = 0

    def last(self):
        agent = self.agent_selection
        reward = self.rewards[agent]
        termination = self.terminations[agent]
        truncation = self.truncations[agent]
        info = self.infos[agent]
        return {"agent": agent, "step": self._last_obs_step}, reward, termination, truncation, info


class ScalarLikeReward:
    def __init__(self, value: float) -> None:
        self.value = value

    def __float__(self) -> float:
        return float(self.value)

    def item(self) -> float:
        return float(self.value)

    def __str__(self) -> str:
        return str(self.value)


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


def test_pettingzoo_env_uses_gamekit_owned_discrete_action_codec():
    env = FakeAecEnv(max_steps=2, action_n=4)
    adapter = PettingZooAecArenaEnvironment(
        env=env,
        env_id="pettingzoo.atari.space_invaders_v2",
        player_ids=["player_0", "player_1"],
        illegal_policy={"retry": 0, "on_fail": "loss"},
        use_action_meanings=False,
    )

    assert isinstance(adapter._codec, DiscreteActionCodec)
    assert adapter._codec.__class__.__module__ == (
        "gage_eval.game_kits.aec_env_game.pettingzoo.action_codec"
    )


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


def test_pettingzoo_env_normalizes_scalar_like_reward_to_python_float():
    env = FakeAecEnv(max_steps=3, action_n=3)
    adapter = PettingZooAecArenaEnvironment(
        env=env,
        player_ids=["player_0", "player_1"],
        illegal_policy={"retry": 0, "on_fail": "loss"},
    )
    active_agent = env.agent_selection
    env.rewards[active_agent] = ScalarLikeReward(1.5)

    active_player = adapter.get_active_player()
    observation = adapter.observe(active_player)
    frame = adapter.get_last_frame()

    assert observation.metadata["reward"] == 1.5
    assert isinstance(observation.metadata["reward"], float)
    assert frame["reward"] == 1.5
    assert isinstance(frame["reward"], float)

    adapter.apply(ArenaAction(player=active_player, move="1", raw="1"))
    result = adapter.build_result(result="completed", reason="manual")

    assert result.move_log[0]["reward"] == 1.5
    assert isinstance(result.move_log[0]["reward"], float)


def test_pettingzoo_env_captures_fresh_observation_when_last_is_stale():
    env = FakeStaleLastAecEnv()
    adapter = PettingZooAecArenaEnvironment(
        env=env,
        player_ids=["player_0", "player_1"],
        illegal_policy={"retry": 0, "on_fail": "loss"},
        include_raw_obs=True,
    )

    active_player = adapter.get_active_player()
    adapter.apply(ArenaAction(player=active_player, move="1", raw="1"))
    observation = adapter.observe(adapter.get_active_player())

    assert observation.metadata["raw_obs"]["step"] == env._step_count  # noqa: SLF001


def test_pettingzoo_env_apply_repeats_human_action_and_auto_noops_dummy_turns():
    env = RecordingAecEnv(max_steps=20, action_n=3)
    adapter = PettingZooAecArenaEnvironment(
        env=env,
        player_ids=["pilot_0", "pilot_1"],
        illegal_policy={"retry": 0, "on_fail": "loss"},
        use_action_meanings=False,
        action_schema={"hold_ticks_min": 1, "hold_ticks_max": 8, "hold_ticks_default": 4},
        auto_noop_player_ids=["pilot_1"],
    )

    result = adapter.apply(
        ArenaAction(
            player="pilot_0",
            move="1",
            raw='{"move":"1","hold_ticks":3}',
            metadata={"hold_ticks": 3},
        )
    )

    assert result is None
    assert env.actions == [
        ("agent_0", 1),
        ("agent_1", 0),
        ("agent_0", 1),
        ("agent_1", 0),
        ("agent_0", 1),
        ("agent_1", 0),
    ]
    assert adapter.get_active_player() == "pilot_0"
    assert adapter._move_count == 1  # noqa: SLF001
    assert adapter._move_log[0]["hold_ticks"] == 3  # noqa: SLF001


def test_pettingzoo_env_uses_action_schema_default_and_exposes_schema_metadata():
    env = RecordingAecEnv(max_steps=20, action_n=3)
    adapter = PettingZooAecArenaEnvironment(
        env=env,
        player_ids=["pilot_0", "pilot_1"],
        illegal_policy={"retry": 0, "on_fail": "loss"},
        use_action_meanings=False,
        action_schema={"hold_ticks_min": 1, "hold_ticks_max": 8, "hold_ticks_default": 4},
        auto_noop_player_ids=["pilot_1"],
    )

    observation = adapter.observe("pilot_0")
    assert observation.metadata["action_schema_config"] == {
        "hold_ticks_min": 1,
        "hold_ticks_max": 8,
        "hold_ticks_default": 4,
    }
    assert (
        observation.context["action_schema_config"]
        == observation.metadata["action_schema_config"]
    )

    adapter.apply(ArenaAction(player="pilot_0", move="1", raw="1"))

    assert len(env.actions) == 8
    assert env.actions[0] == ("agent_0", 1)
    assert env.actions[-1] == ("agent_1", 0)
    assert adapter._move_log[0]["hold_ticks"] == 4  # noqa: SLF001
