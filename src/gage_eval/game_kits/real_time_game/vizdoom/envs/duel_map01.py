from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any, Sequence

from gage_eval.game_kits.real_time_game.backend_mode import normalize_backend_mode
from gage_eval.role.arena.resources.runtime_bridge import attach_runtime_resources
from gage_eval.role.arena.games.vizdoom.env import (
    DEFAULT_ACTION_LABELS,
    ViZDoomArenaEnvironment,
)


def _default_replay_dir(name: str) -> str:
    return str(Path(tempfile.gettempdir()) / "gage_eval_gamearena" / name)


class _StubFrame:
    def __init__(self, marker: int) -> None:
        self.shape = (1, 1, 3)
        self.dtype = "uint8"
        self._bytes = bytes([marker % 256, 0, 0])

    def tobytes(self) -> bytes:
        return self._bytes


class _StubViZDoomBackend:
    def __init__(self, *, max_rounds: int) -> None:
        self._max_rounds = max(1, int(max_rounds))
        self._round = 0
        self._shots = {0: 0, 1: 0}
        self._last_actions = {0: 0, 1: 0}
        self._pov_frames = {0: _StubFrame(11), 1: _StubFrame(22)}
        self._view = None

    def reset(self, seed: int | None = None) -> dict[int, dict[str, Any]]:
        self._round = 0
        self._shots = {0: 0, 1: 0}
        self._last_actions = {0: 0, 1: 0}
        self._pov_frames = {0: _StubFrame(11), 1: _StubFrame(22)}
        return {
            0: {"HEALTH": 100.0, "AMMO": 8, "seed": seed, "round": 0},
            1: {"HEALTH": 100.0, "AMMO": 8, "seed": seed, "round": 0},
        }

    def close(self) -> None:
        return None

    def set_view(self, view: str) -> None:
        self._view = view

    def get_pov_frames(self) -> dict[int, _StubFrame]:
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
            0: _StubFrame(11 + self._round),
            1: _StubFrame(22 + self._round),
        }

        done = self._round >= self._max_rounds
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


class DuelMap01Environment(ViZDoomArenaEnvironment):
    def __init__(
        self,
        *,
        backend_mode: str,
        stub_max_rounds: int,
        runtime_backend: Any | None = None,
        **kwargs: Any,
    ) -> None:
        self._backend_mode = normalize_backend_mode(backend_mode)
        self._stub_max_rounds = max(1, int(stub_max_rounds))
        self._runtime_backend = runtime_backend
        self._turn_player_id: str | None = None
        self._turn_player_index = 0
        super().__init__(**kwargs)

    def reset(self) -> None:
        super().reset()
        self._turn_player_index = 0
        self._turn_player_id = None

    def get_active_player(self) -> str:
        if not self._player_ids:
            return "p0"
        if self._turn_player_id is None:
            index = self._turn_player_index % len(self._player_ids)
            self._turn_player_id = self._player_ids[index]
        return self._turn_player_id

    def apply(self, action):
        result = super().apply(action)
        if self._player_ids:
            current_player = self._turn_player_id or str(action.player)
            try:
                current_index = self._player_ids.index(current_player)
            except ValueError:
                current_index = self._turn_player_index % len(self._player_ids)
            self._turn_player_index = (current_index + 1) % len(self._player_ids)
        self._turn_player_id = None
        return result

    def _build_env(self, cfg):
        if self._runtime_backend is not None:
            return self._runtime_backend
        if self._backend_mode == "dummy":
            return _StubViZDoomBackend(max_rounds=self._stub_max_rounds)
        return super()._build_env(cfg)

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
        raw_action_labels = defaults.get("action_labels")
        action_labels = (
            tuple(str(label) for label in raw_action_labels)
            if raw_action_labels is not None
            else tuple(DEFAULT_ACTION_LABELS)
        )
        runtime = getattr(resources, "game_runtime", None)
        backend_mode = normalize_backend_mode(defaults.get("backend_mode", "real"))
        runtime_backend = None
        if backend_mode == "dummy" and runtime is not None and hasattr(runtime, "configure_vizdoom"):
            runtime_backend = runtime.configure_vizdoom(
                max_rounds=int(defaults.get("stub_max_rounds", 3))
            )
        environment = cls(
            backend_mode=backend_mode,
            stub_max_rounds=int(defaults.get("stub_max_rounds", 3)),
            runtime_backend=runtime_backend,
            player_ids=player_ids,
            player_names=player_names,
            start_player_id=defaults.get("start_player_id") or (player_ids[0] if player_ids else None),
            use_single_process=bool(defaults.get("use_single_process", False)),
            render_mode=defaults.get("render_mode"),
            pov_view=defaults.get("pov_view"),
            show_automap=bool(defaults.get("show_automap", False)),
            automap_scale=int(defaults.get("automap_scale", 3)),
            automap_follow=bool(defaults.get("automap_follow", False)),
            automap_stride=int(defaults.get("automap_stride", 1)),
            show_pov=bool(defaults.get("show_pov", False)),
            capture_pov=bool(defaults.get("capture_pov", False)),
            pov_stride=int(defaults.get("pov_stride", 1)),
            allow_respawn=bool(defaults.get("allow_respawn", False)),
            respawn_grace_steps=int(defaults.get("respawn_grace_steps", 0)),
            no_attack_seconds=float(defaults.get("no_attack_seconds", 0.0)),
            max_steps=int(defaults.get("max_steps", 16)),
            action_repeat=int(defaults.get("action_repeat", 1)),
            sleep_s=float(defaults.get("sleep_s", 0.0)),
            port=defaults.get("port"),
            config_path=defaults.get("config_path"),
            replay_output_dir=str(
                defaults.get("replay_output_dir")
                or _default_replay_dir("vizdoom")
            ),
            game_id=str(defaults.get("game_id", "vizdoom_multi_duel_map01")),
            tick_rate_hz=defaults.get("tick_rate_hz"),
            frame_stride=int(defaults.get("frame_stride", 1)),
            time_source=str(defaults.get("time_source", "wall_clock")),
            obs_image=bool(defaults.get("obs_image", False)),
            obs_image_history_len=int(defaults.get("obs_image_history_len", 1)),
            replay_in_env=bool(defaults.get("replay_in_env", True)),
            action_labels=action_labels,
            allow_partial_actions=bool(defaults.get("allow_partial_actions", False)),
            reset_retry_count=int(defaults.get("reset_retry_count", 1)),
            death_check_warmup_steps=int(defaults.get("death_check_warmup_steps", 0)),
        )
        try:
            environment.reset()
        finally:
            if backend_mode != "dummy" and runtime is not None:
                backend = getattr(environment, "_env", None)
                if backend is not None and hasattr(runtime, "adopt_backend"):
                    runtime.adopt_backend(backend, backend_kind="vizdoom")
        return attach_runtime_resources(environment, resources)


def build_duel_map01_environment(*, sample, resolved, resources, player_specs) -> Any:
    return DuelMap01Environment.from_runtime(
        sample=sample,
        resolved=resolved,
        resources=resources,
        player_specs=player_specs,
    )
