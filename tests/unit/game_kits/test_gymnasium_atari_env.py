from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from gage_eval.game_kits.gymnasium_atari.envs.space_invaders import (
    SpaceInvadersGymEnvironment,
)
from gage_eval.role.arena.types import ArenaAction


class _DiscreteActionSpace:
    def __init__(self, n: int) -> None:
        self.n = n

    def contains(self, item: int) -> bool:
        return isinstance(item, int) and 0 <= item < self.n


class _FakeGymSpaceInvaders:
    action_space = _DiscreteActionSpace(4)
    render_mode = "rgb_array"

    def __init__(self) -> None:
        self.actions: list[int] = []
        self._step = 0
        self.unwrapped = self

    def get_action_meanings(self) -> list[str]:
        return ["NOOP", "FIRE", "RIGHT", "LEFT"]

    def reset(self, *, seed: int | None = None, options: object | None = None):
        del seed, options
        self.actions = []
        self._step = 0
        return self._frame(), {"reset": True}

    def step(self, action: int):
        self.actions.append(action)
        self._step += 1
        terminated = self._step >= 99
        return self._frame(), float(action), terminated, False, {"step": self._step}

    def render(self):
        return self._frame()

    def close(self) -> None:
        return None

    def _frame(self):
        frame = np.zeros((8, 8, 3), dtype=np.uint8)
        frame[:, :, 0] = self._step
        frame[:, :, 1] = 80
        frame[2:6, 2:6, 2] = 220
        return frame


@dataclass(frozen=True)
class _PlayerSpec:
    player_id: str = "pilot_0"
    display_name: str = "pilot_0"
    player_kind: str = "human"


def test_apply_repeats_single_player_action_for_hold_ticks() -> None:
    backend = _FakeGymSpaceInvaders()
    env = SpaceInvadersGymEnvironment(
        env=backend,
        env_id="ALE/SpaceInvaders-v5",
        player_specs=(_PlayerSpec(),),
        action_schema={
            "hold_ticks_min": 1,
            "hold_ticks_max": 8,
            "hold_ticks_default": 4,
        },
        max_cycles=20,
    )

    result = env.apply(
        ArenaAction(
            player="pilot_0",
            move="FIRE",
            raw="FIRE",
            metadata={"hold_ticks": 3},
        )
    )

    assert result is None
    assert backend.actions == [1, 1, 1]
    assert env.get_last_frame()["move_count"] == 1
    assert env.get_last_frame()["metadata"]["tick"] == 3
    assert env.get_last_frame()["metadata"]["action_schema_config"] == {
        "hold_ticks_min": 1,
        "hold_ticks_max": 8,
        "hold_ticks_default": 4,
    }


def test_tick_idle_advances_backend_without_recording_player_decision() -> None:
    backend = _FakeGymSpaceInvaders()
    env = SpaceInvadersGymEnvironment(
        env=backend,
        env_id="ALE/SpaceInvaders-v5",
        player_specs=(_PlayerSpec(),),
        max_cycles=20,
    )

    result = env.tick_idle(frames=2, move="NOOP")

    assert result is None
    assert backend.actions == [0, 0]
    assert env.get_last_frame()["move_count"] == 0
    assert env.get_last_frame()["metadata"]["tick"] == 2
    assert env.get_last_frame()["last_move"] is None
    assert env.observe("pilot_0").last_move is None


def test_observe_exposes_action_meanings_and_frame_payload() -> None:
    env = SpaceInvadersGymEnvironment(
        env=_FakeGymSpaceInvaders(),
        env_id="ALE/SpaceInvaders-v5",
        player_specs=(_PlayerSpec(),),
        max_cycles=20,
    )

    observation = env.observe("pilot_0")

    assert observation.active_player == "pilot_0"
    assert observation.legal_moves == ["NOOP", "FIRE", "RIGHT", "LEFT"]
    assert observation.legal_actions == {"items": ["NOOP", "FIRE", "RIGHT", "LEFT"]}
    assert observation.context == {
        "mode": "turn",
        "step": 0,
        "tick": 0,
        "action_schema_config": {
            "hold_ticks_min": 1,
            "hold_ticks_max": 1,
            "hold_ticks_default": 1,
        },
    }
    frame = env.get_last_frame()
    assert frame["active_player_id"] == "pilot_0"
    assert frame["legal_moves"] == ["NOOP", "FIRE", "RIGHT", "LEFT"]
    assert frame["_rgb"].shape == (8, 8, 3)


def test_observe_leaves_frame_encoding_to_game_session_visual_pipeline() -> None:
    env = SpaceInvadersGymEnvironment(
        env=_FakeGymSpaceInvaders(),
        env_id="ALE/SpaceInvaders-v5",
        player_specs=(_PlayerSpec(),),
        max_cycles=20,
    )

    observation = env.observe("pilot_0")

    assert observation.view == {"text": observation.board_text}
    assert env.get_last_frame()["_rgb"].shape == (8, 8, 3)
