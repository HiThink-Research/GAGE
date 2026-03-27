from __future__ import annotations

from dataclasses import dataclass
from dataclasses import field
from typing import Any

from gage_eval.role.arena.resources.handles import RuntimeHandle


class _RuntimeFrame:
    def __init__(self, marker: int) -> None:
        self.shape = (1, 1, 3)
        self.dtype = "uint8"
        self._bytes = bytes([marker % 256, 0, 0])

    def tobytes(self) -> bytes:
        return self._bytes


class _ViZDoomRuntimeBackend:
    def __init__(self, *, max_rounds: int = 3) -> None:
        self.max_rounds = max(1, int(max_rounds))
        self.closed = False
        self.terminated = False
        self.reaped = False
        self._round = 0
        self._shots = {0: 0, 1: 0}
        self._last_actions = {0: 0, 1: 0}
        self._pov_frames = {0: _RuntimeFrame(11), 1: _RuntimeFrame(22)}
        self._view = None

    def configure(self, *, max_rounds: int | None = None) -> None:
        if max_rounds is not None:
            self.max_rounds = max(1, int(max_rounds))

    def reset(self, seed: int | None = None) -> dict[int, dict[str, Any]]:
        self._round = 0
        self._shots = {0: 0, 1: 0}
        self._last_actions = {0: 0, 1: 0}
        self._pov_frames = {0: _RuntimeFrame(11), 1: _RuntimeFrame(22)}
        return {
            0: {"HEALTH": 100.0, "AMMO": 8, "seed": seed, "round": 0},
            1: {"HEALTH": 100.0, "AMMO": 8, "seed": seed, "round": 0},
        }

    def close(self) -> None:
        self.closed = True

    def terminate(self) -> None:
        self.terminated = True

    def reap(self) -> None:
        self.reaped = True

    def set_view(self, view: str) -> None:
        self._view = view

    def get_pov_frames(self) -> dict[int, _RuntimeFrame]:
        return dict(self._pov_frames)

    def step(
        self,
        actions: dict[int, int],
    ) -> tuple[dict[int, dict[str, Any]], dict[int, float], bool, dict[str, Any]]:
        self._round += 1
        self._last_actions = {0: int(actions.get(0, 0)), 1: int(actions.get(1, 0))}
        for player_index, move_id in self._last_actions.items():
            if move_id == 1:
                self._shots[player_index] += 1

        self._pov_frames = {
            0: _RuntimeFrame(11 + self._round),
            1: _RuntimeFrame(22 + self._round),
        }

        done = self._round >= self.max_rounds
        winner_index = 0 if self._shots[0] >= self._shots[1] else 1
        info: dict[str, Any] = {
            "round": self._round,
            "shots": dict(self._shots),
            "actions": dict(self._last_actions),
            "view": self._view,
        }
        if done:
            info["outcome"] = "p0_win" if winner_index == 0 else "p1_win"

        observations = {
            0: {
                "HEALTH": 100.0 - self._round,
                "ARMOR": 50.0,
                "last_action": self._last_actions[0],
                "shots": self._shots[0],
            },
            1: {
                "HEALTH": 100.0 - self._round,
                "ARMOR": 50.0,
                "last_action": self._last_actions[1],
                "shots": self._shots[1],
            },
        }
        rewards = {
            0: 1.0 if done and winner_index == 0 else 0.0,
            1: 1.0 if done and winner_index == 1 else 0.0,
        }
        return observations, rewards, done, info


class _RetroRuntimeBackend:
    buttons = ["LEFT", "RIGHT", "UP", "DOWN", "A", "B", "START", "SELECT"]

    def __init__(self, *, max_ticks: int = 4) -> None:
        self.max_ticks = max(1, int(max_ticks))
        self.closed = False
        self.terminated = False
        self.reaped = False
        self._tick = 0
        self._progress = 0

    def configure(self, *, max_ticks: int | None = None) -> None:
        if max_ticks is not None:
            self.max_ticks = max(1, int(max_ticks))

    def reset(self, seed: int | None = None):
        self._tick = 0
        self._progress = 0
        return _RuntimeFrame(31), {"tick": 0, "x": 0, "score": 0, "seed": seed}

    def step(self, payload):
        self._tick += 1
        pressed = {
            self.buttons[index]: bool(value)
            for index, value in enumerate(payload)
            if index < len(self.buttons)
        }
        self._progress += 1
        if pressed.get("RIGHT"):
            self._progress += 2
        if pressed.get("B"):
            self._progress += 1
        if pressed.get("A"):
            self._progress += 1
        terminated = self._tick >= self.max_ticks
        reward = float(self._progress)
        info = {
            "tick": self._tick,
            "x": self._progress,
            "score": self._progress * 10,
            "win": terminated,
        }
        return _RuntimeFrame(31 + self._tick), reward, terminated, False, info

    def close(self) -> None:
        self.closed = True

    def terminate(self) -> None:
        self.terminated = True

    def reap(self) -> None:
        self.reaped = True


@dataclass
class RuntimeLease:
    resource_spec: object = field(default_factory=dict)
    backend_kind: str | None = None
    backend: Any | None = None
    closed: bool = False
    terminated: bool = False
    reaped: bool = False

    def configure_vizdoom(self, *, max_rounds: int | None = None) -> Any:
        if self.backend_kind not in (None, "vizdoom"):
            raise TypeError("runtime lease is not configured for ViZDoom")
        if not isinstance(self.backend, _ViZDoomRuntimeBackend):
            self.backend = _ViZDoomRuntimeBackend(max_rounds=max_rounds or 3)
        else:
            self.backend.configure(max_rounds=max_rounds)
        self.backend_kind = "vizdoom"
        return self.backend

    def configure_retro(self, *, max_ticks: int | None = None) -> Any:
        if self.backend_kind not in (None, "retro"):
            raise TypeError("runtime lease is not configured for retro")
        if not isinstance(self.backend, _RetroRuntimeBackend):
            self.backend = _RetroRuntimeBackend(max_ticks=max_ticks or 4)
        else:
            self.backend.configure(max_ticks=max_ticks)
        self.backend_kind = "retro"
        return self.backend

    def adopt_backend(self, backend: Any, *, backend_kind: str | None = None) -> Any:
        self.backend = backend
        if backend_kind is not None:
            self.backend_kind = backend_kind
        elif self.backend_kind is None:
            self.backend_kind = "adopted"
        return backend

    def close(self) -> None:
        if self.closed:
            return
        self.closed = True
        backend = self.backend
        if backend is not None and hasattr(backend, "close"):
            backend.close()

    def terminate(self) -> None:
        if self.terminated:
            return
        self.terminated = True
        backend = self.backend
        if backend is not None and hasattr(backend, "terminate"):
            backend.terminate()

    def reap(self) -> None:
        if self.reaped:
            return
        self.reaped = True
        backend = self.backend
        if backend is not None and hasattr(backend, "reap"):
            backend.reap()


@dataclass
class RuntimeBridge:
    runtime: RuntimeHandle | None = None
    resource_spec: object = field(default_factory=dict)


def build_runtime_bridge(
    *,
    resource_spec: object,
    runtime: RuntimeHandle | None = None,
) -> RuntimeBridge:
    return RuntimeBridge(runtime=runtime, resource_spec=resource_spec)


def attach_runtime_resources(target: object, resources: object) -> object:
    setattr(target, "_arena_resources", resources)
    setattr(target, "_resource_spec", getattr(resources, "resource_spec", None))
    setattr(target, "_runtime_handle", getattr(resources, "game_runtime", None))
    setattr(target, "_runtime_bridge", getattr(resources, "game_bridge", None))
    return target
