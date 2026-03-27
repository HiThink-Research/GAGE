from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any

from gage_eval.game_kits.real_time_game.backend_mode import normalize_backend_mode
from gage_eval.role.arena.resources.runtime_bridge import attach_runtime_resources
from gage_eval.role.arena.games.retro.retro_env import StableRetroArenaEnvironment


def _default_replay_dir(name: str) -> str:
    return str(Path(tempfile.gettempdir()) / "gage_eval_gamearena" / name)


class _StubRetroFrame:
    def __init__(self, marker: int) -> None:
        self.shape = (1, 1, 3)
        self.dtype = "uint8"
        self._bytes = bytes([marker % 256, 0, 0])

    def tobytes(self) -> bytes:
        return self._bytes


class _StubRetroBackend:
    buttons = ["LEFT", "RIGHT", "UP", "DOWN", "A", "B", "START", "SELECT"]

    def __init__(self, *, max_ticks: int) -> None:
        self._max_ticks = max(1, int(max_ticks))
        self._tick = 0
        self._progress = 0

    def reset(self, seed: int | None = None):
        self._tick = 0
        self._progress = 0
        return _StubRetroFrame(31), {"tick": 0, "x": 0, "score": 0, "seed": seed}

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
        terminated = self._tick >= self._max_ticks
        reward = float(self._progress)
        info = {
            "tick": self._tick,
            "x": self._progress,
            "score": self._progress * 10,
            "win": terminated,
        }
        return _StubRetroFrame(31 + self._tick), reward, terminated, False, info


class RetroMarioEnvironment(StableRetroArenaEnvironment):
    def __init__(
        self,
        *,
        backend_mode: str,
        stub_max_ticks: int,
        runtime_backend: Any | None = None,
        **kwargs: Any,
    ) -> None:
        self._backend_mode = normalize_backend_mode(backend_mode)
        self._stub_max_ticks = max(1, int(stub_max_ticks))
        self._runtime_backend = runtime_backend
        super().__init__(**kwargs)

    def _make_env(self):
        if self._runtime_backend is not None:
            return self._runtime_backend
        if self._backend_mode == "dummy":
            return _StubRetroBackend(max_ticks=self._stub_max_ticks)
        return super()._make_env()

    @classmethod
    def from_runtime(cls, *, sample, resolved, resources, player_specs):
        defaults = {
            **dict(resolved.game_kit.defaults),
            **dict(resolved.env_spec.defaults),
            **dict(sample.runtime_overrides or {}),
        }
        player_ids = [str(getattr(player, "player_id")) for player in player_specs]
        player_names = {
            str(getattr(player, "player_id")): str(getattr(player, "display_name"))
            for player in player_specs
        }
        runtime = getattr(resources, "game_runtime", None)
        backend_mode = normalize_backend_mode(defaults.get("backend_mode", "real"))
        runtime_backend = None
        if backend_mode == "dummy" and runtime is not None and hasattr(runtime, "configure_retro"):
            runtime_backend = runtime.configure_retro(
                max_ticks=int(defaults.get("stub_max_ticks", 4))
            )
        environment = cls(
            backend_mode=backend_mode,
            stub_max_ticks=int(defaults.get("stub_max_ticks", 4)),
            runtime_backend=runtime_backend,
            game=str(defaults.get("game", "SuperMarioBros3-Nes-v0")),
            state=defaults.get("state"),
            default_state=str(defaults.get("default_state", "Start")),
            rom_path=defaults.get("rom_path"),
            player_ids=player_ids or ["player_0"],
            player_names=player_names,
            runtime_policy=str(defaults.get("runtime_policy", "persistent")),
            display_mode=str(defaults.get("display_mode", "headless")),
            record_bk2=bool(defaults.get("record_bk2", False)),
            record_dir=defaults.get("record_dir"),
            record_filename=defaults.get("record_filename"),
            record_path=defaults.get("record_path"),
            action_mapping=defaults.get("action_mapping"),
            legal_moves=defaults.get(
                "legal_moves",
                ("noop", "right", "right_run", "right_jump", "right_run_jump", "jump"),
            ),
            info_feeder=defaults.get("info_feeder"),
            action_schema=defaults.get("action_schema"),
            token_budget=int(defaults.get("token_budget", 200)),
            frame_stride=int(defaults.get("frame_stride", 1)),
            snapshot_stride=int(defaults.get("snapshot_stride", 1)),
            obs_image=bool(defaults.get("obs_image", False)),
            replay_output_dir=str(
                defaults.get("replay_output_dir")
                or _default_replay_dir("retro_platformer")
            ),
            replay_filename=defaults.get("replay_filename"),
            frame_output_dir=defaults.get("frame_output_dir"),
            run_id=defaults.get("run_id"),
            sample_id=defaults.get("sample_id"),
            seed=int(defaults["seed"]) if defaults.get("seed") is not None else None,
        )
        try:
            environment.reset()
        finally:
            if backend_mode != "dummy" and runtime is not None:
                backend = getattr(environment, "_retro_env", None)
                if backend is not None and hasattr(runtime, "adopt_backend"):
                    runtime.adopt_backend(backend, backend_kind="retro")
        return attach_runtime_resources(environment, resources)


def build_retro_mario_environment(*, sample, resolved, resources, player_specs) -> Any:
    return RetroMarioEnvironment.from_runtime(
        sample=sample,
        resolved=resolved,
        resources=resources,
        player_specs=player_specs,
    )
