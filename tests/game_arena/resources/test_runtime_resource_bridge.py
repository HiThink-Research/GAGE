from __future__ import annotations

from types import SimpleNamespace

import pytest

from gage_eval.role.arena.core.arena_core import GameArenaCore
from gage_eval.role.arena.core.types import ArenaSample
from gage_eval.role.arena.resources.control import ArenaResourceControl
from gage_eval.game_kits.aec_env_game.pettingzoo.kit import (
    build_pettingzoo_game_kit,
)
from gage_eval.game_kits.board_game.gomoku.kit import (
    build_gomoku_game_kit,
)
from gage_eval.game_kits.board_game.tictactoe.kit import (
    build_tictactoe_game_kit,
)
from gage_eval.game_kits.phase_card_game.doudizhu.kit import (
    build_doudizhu_game_kit,
)
from gage_eval.game_kits.phase_card_game.mahjong.kit import (
    build_mahjong_game_kit,
)
from gage_eval.game_kits.real_time_game.retro_platformer.kit import (
    build_retro_platformer_game_kit,
)
from gage_eval.game_kits.real_time_game.vizdoom.kit import (
    build_vizdoom_game_kit,
)
from gage_eval.game_kits.board_game.gomoku.envs.gomoku_standard import (
    build_gomoku_standard_environment,
)
from gage_eval.game_kits.board_game.tictactoe.envs.tictactoe_standard import (
    build_tictactoe_standard_environment,
)
from gage_eval.game_kits.aec_env_game.pettingzoo.envs.space_invaders import (
    build_space_invaders_environment,
)
from gage_eval.game_kits.real_time_game.vizdoom.envs.duel_map01 import (
    build_duel_map01_environment,
)
from gage_eval.game_kits.real_time_game.retro_platformer.envs.retro_mario import (
    build_retro_mario_environment,
)
from gage_eval.game_kits.real_time_game.retro_platformer.environment import (
    StableRetroArenaEnvironment,
)
from gage_eval.game_kits.real_time_game.vizdoom.environment import ViZDoomArenaEnvironment
from gage_eval.game_kits.phase_card_game.doudizhu.envs.classic_3p import (
    build_classic_3p_environment,
)
from gage_eval.game_kits.phase_card_game.mahjong.envs.riichi_4p import (
    build_riichi_4p_environment,
)


def _build_resolved_from_kit(
    kit,
    *,
    env_factory,
    env_defaults_overrides: dict[str, object] | None = None,
):
    env_spec = kit.env_catalog[0]
    defaults = dict(env_spec.defaults)
    defaults["env_factory"] = env_factory
    defaults.update(env_defaults_overrides or {})
    return SimpleNamespace(
        game_kit=SimpleNamespace(
            kit_id=kit.kit_id,
            defaults=dict(kit.defaults),
            seat_spec=dict(kit.seat_spec),
        ),
        env_spec=SimpleNamespace(
            env_id=env_spec.env_id,
            defaults=defaults,
        ),
        resource_spec=env_spec.resource_spec,
        scheduler=SimpleNamespace(run=lambda session: None, defaults={}),
        players=(
            {"player_id": "p0", "display_name": "P0"},
            {"player_id": "p1", "display_name": "P1"},
            {"player_id": "p2", "display_name": "P2"},
            {"player_id": "p3", "display_name": "P3"},
        ),
        observation_workflow=None,
    )


def _build_sample(*, game_kit: str, env: str) -> ArenaSample:
    return ArenaSample(
        game_kit=game_kit,
        env=env,
        players=(
            {"player_id": "p0", "display_name": "P0"},
            {"player_id": "p1", "display_name": "P1"},
            {"player_id": "p2", "display_name": "P2"},
            {"player_id": "p3", "display_name": "P3"},
        ),
    )


def test_arena_resource_control_allocate_materializes_runtime_bridge() -> None:
    resources = ArenaResourceControl().allocate(
        {"env_id": "gomoku_standard", "family": "gomoku"}
    )

    assert resources.resource_spec == {"env_id": "gomoku_standard", "family": "gomoku"}
    assert resources.game_runtime is not None
    assert resources.game_bridge is not None
    assert resources.game_bridge.runtime is resources.game_runtime
    assert resources.game_bridge.resource_spec == resources.resource_spec


def test_arena_resource_control_release_cleans_allocated_runtime_handle() -> None:
    resources = ArenaResourceControl().allocate(
        {"env_id": "gomoku_standard", "family": "gomoku"}
    )
    runtime = resources.game_runtime

    ArenaResourceControl().release(resources)

    assert runtime is not None
    assert getattr(runtime, "closed", False) is True
    assert getattr(runtime, "terminated", False) is True
    assert getattr(runtime, "reaped", False) is True


@pytest.mark.parametrize(
    ("resource_spec"),
    [
        {"env_id": "duel_map01", "family": "vizdoom"},
        {"env_id": "retro_mario", "family": "retro"},
    ],
)
def test_arena_resource_control_allocate_leaves_realtime_backend_unbound(
    resource_spec: dict[str, object],
) -> None:
    resources = ArenaResourceControl().allocate(resource_spec)

    assert resources.game_runtime is not None
    assert resources.game_runtime.backend is None
    assert resources.game_runtime.backend_kind is None


@pytest.mark.parametrize(
    ("kit_builder", "env_builder", "game_kit", "env_id", "env_defaults_overrides"),
    [
        (
            build_gomoku_game_kit,
            build_gomoku_standard_environment,
            "gomoku",
            "gomoku_standard",
            {},
        ),
        (
            build_tictactoe_game_kit,
            build_tictactoe_standard_environment,
            "tictactoe",
            "tictactoe_standard",
            {},
        ),
        (
            build_pettingzoo_game_kit,
            build_space_invaders_environment,
            "pettingzoo",
            "space_invaders",
            {},
        ),
        (
            build_vizdoom_game_kit,
            build_duel_map01_environment,
            "vizdoom",
            "duel_map01",
            {"backend_mode": "dummy"},
        ),
        (
            build_retro_platformer_game_kit,
            build_retro_mario_environment,
            "retro_platformer",
            "retro_mario",
            {"backend_mode": "dummy"},
        ),
        (
            build_doudizhu_game_kit,
            build_classic_3p_environment,
            "doudizhu",
            "classic_3p",
            {},
        ),
        (
            build_mahjong_game_kit,
            build_riichi_4p_environment,
            "mahjong",
            "riichi_4p",
            {},
        ),
    ],
)
def test_env_builders_preserve_runtime_resources(
    kit_builder,
    env_builder,
    game_kit: str,
    env_id: str,
    env_defaults_overrides: dict[str, object],
) -> None:
    kit = kit_builder()
    resolved = _build_resolved_from_kit(
        kit,
        env_factory=env_builder,
        env_defaults_overrides=env_defaults_overrides,
    )
    sample = _build_sample(game_kit=game_kit, env=env_id)
    resources = ArenaResourceControl().allocate(resolved.resource_spec)
    assert resources.game_runtime is not None
    assert resources.game_runtime.backend is None
    player_specs = tuple(
        SimpleNamespace(player_id=f"p{index}", display_name=f"P{index}")
        for index in range(len(kit.seat_spec.get("seats", ())))
    )

    env = env_builder(
        sample=sample,
        resolved=resolved,
        resources=resources,
        player_specs=player_specs,
    )

    assert getattr(env, "_runtime_bridge") is resources.game_bridge
    assert getattr(env, "_runtime_handle") is resources.game_runtime
    assert getattr(env, "_resource_spec") == resources.resource_spec
    if env_id == "duel_map01":
        assert getattr(env, "_env") is resources.game_runtime.backend
    if env_id == "retro_mario":
        assert getattr(env, "_retro_env") is resources.game_runtime.backend


@pytest.mark.parametrize(
    ("kit_builder", "env_builder", "game_kit", "env_id", "player_count"),
    [
        (
            build_vizdoom_game_kit,
            build_duel_map01_environment,
            "vizdoom",
            "duel_map01",
            2,
        ),
        (
            build_retro_platformer_game_kit,
            build_retro_mario_environment,
            "retro_platformer",
            "retro_mario",
            1,
        ),
    ],
)
def test_realtime_env_builders_accept_legacy_stub_backend_mode(
    kit_builder,
    env_builder,
    game_kit: str,
    env_id: str,
    player_count: int,
) -> None:
    kit = kit_builder()
    resolved = _build_resolved_from_kit(
        kit,
        env_factory=env_builder,
        env_defaults_overrides={"backend_mode": "stub"},
    )
    sample = _build_sample(game_kit=game_kit, env=env_id)
    resources = ArenaResourceControl().allocate(resolved.resource_spec)
    player_specs = tuple(
        SimpleNamespace(player_id=f"p{index}", display_name=f"P{index}")
        for index in range(player_count)
    )

    env = env_builder(
        sample=sample,
        resolved=resolved,
        resources=resources,
        player_specs=player_specs,
    )

    assert env is not None
    assert getattr(env, "_backend_mode") == "dummy"
    assert resources.game_runtime is not None
    assert getattr(resources.game_runtime, "backend", None) is not None


class _FakeViZDoomBackend:
    def __init__(self) -> None:
        self.closed = False
        self.terminated = False
        self.reaped = False
        self.close_calls = 0
        self.terminate_calls = 0
        self.reap_calls = 0
        self._pov_frames = {0: SimpleNamespace(shape=(1, 1, 3), dtype="uint8"), 1: SimpleNamespace(shape=(1, 1, 3), dtype="uint8")}

    def reset(self, seed=None):
        del seed
        return (
            {
                0: {"HEALTH": 100.0, "AMMO": 8},
                1: {"HEALTH": 100.0, "AMMO": 8},
            }
        )

    def close(self) -> None:
        self.closed = True
        self.close_calls += 1

    def terminate(self) -> None:
        self.terminated = True
        self.terminate_calls += 1

    def reap(self) -> None:
        self.reaped = True
        self.reap_calls += 1

    def set_view(self, view: str) -> None:
        self.view = view

    def get_pov_frames(self):
        return dict(self._pov_frames)

    def step(self, actions):
        del actions
        return (
            {
                0: {"HEALTH": 99.0, "AMMO": 8},
                1: {"HEALTH": 99.0, "AMMO": 8},
            },
            {0: 0.0, 1: 0.0},
            False,
            {"outcome": None},
        )


class _FakeRetroBackend:
    buttons = ["LEFT", "RIGHT", "UP", "DOWN", "A", "B", "START", "SELECT"]

    def __init__(self) -> None:
        self.closed = False
        self.terminated = False
        self.reaped = False
        self.close_calls = 0
        self.terminate_calls = 0
        self.reap_calls = 0

    def reset(self, seed=None):
        del seed
        return SimpleNamespace(shape=(1, 1, 3), dtype="uint8"), {"tick": 0, "win": False}

    def step(self, payload):
        del payload
        return (
            SimpleNamespace(shape=(1, 1, 3), dtype="uint8"),
            0.0,
            False,
            False,
            {"win": False},
        )

    def close(self) -> None:
        self.closed = True
        self.close_calls += 1

    def terminate(self) -> None:
        self.terminated = True
        self.terminate_calls += 1

    def reap(self) -> None:
        self.reaped = True
        self.reap_calls += 1


def test_realtime_vizdoom_real_path_adopts_actual_backend(monkeypatch) -> None:
    fake_backend = _FakeViZDoomBackend()

    monkeypatch.setattr(ViZDoomArenaEnvironment, "_build_env", lambda self, cfg: fake_backend)

    kit = build_vizdoom_game_kit()
    env_spec = kit.env_catalog[0]
    resolved = _build_resolved_from_kit(
        kit,
        env_factory=build_duel_map01_environment,
        env_defaults_overrides={"backend_mode": "real"},
    )
    resources = ArenaResourceControl().allocate(resolved.resource_spec)
    env = build_duel_map01_environment(
        sample=_build_sample(game_kit="vizdoom", env="duel_map01"),
        resolved=resolved,
        resources=resources,
        player_specs=(SimpleNamespace(player_id="p0", display_name="P0"), SimpleNamespace(player_id="p1", display_name="P1")),
    )

    assert env._env is fake_backend
    assert resources.game_runtime.backend is fake_backend
    assert resources.game_runtime.backend_kind == "vizdoom"

    ArenaResourceControl().release(resources)

    assert fake_backend.closed is True
    assert fake_backend.terminated is True
    assert fake_backend.reaped is True


def test_realtime_vizdoom_real_path_adopts_backend_even_if_reset_fails(monkeypatch) -> None:
    fake_backend = _FakeViZDoomBackend()

    monkeypatch.setattr(ViZDoomArenaEnvironment, "_build_env", lambda self, cfg: fake_backend)
    monkeypatch.setattr(
        ViZDoomArenaEnvironment,
        "reset",
        lambda self: (_ for _ in ()).throw(RuntimeError("reset failed")),
    )

    kit = build_vizdoom_game_kit()
    resolved = _build_resolved_from_kit(
        kit,
        env_factory=build_duel_map01_environment,
        env_defaults_overrides={"backend_mode": "real"},
    )
    resources = ArenaResourceControl().allocate(resolved.resource_spec)

    with pytest.raises(RuntimeError, match="reset failed"):
        build_duel_map01_environment(
            sample=_build_sample(game_kit="vizdoom", env="duel_map01"),
            resolved=resolved,
            resources=resources,
            player_specs=(
                SimpleNamespace(player_id="p0", display_name="P0"),
                SimpleNamespace(player_id="p1", display_name="P1"),
            ),
        )

    assert resources.game_runtime.backend is fake_backend
    assert resources.game_runtime.backend_kind == "vizdoom"

    ArenaResourceControl().release(resources)

    assert fake_backend.close_calls == 1
    assert fake_backend.terminate_calls == 1
    assert fake_backend.reap_calls == 1


def test_realtime_retro_real_path_adopts_actual_backend(monkeypatch) -> None:
    fake_backend = _FakeRetroBackend()

    monkeypatch.setattr(StableRetroArenaEnvironment, "_make_env", lambda self: fake_backend)

    kit = build_retro_platformer_game_kit()
    resolved = _build_resolved_from_kit(
        kit,
        env_factory=build_retro_mario_environment,
        env_defaults_overrides={"backend_mode": "real"},
    )
    resources = ArenaResourceControl().allocate(resolved.resource_spec)
    env = build_retro_mario_environment(
        sample=_build_sample(game_kit="retro_platformer", env="retro_mario"),
        resolved=resolved,
        resources=resources,
        player_specs=(SimpleNamespace(player_id="player_0", display_name="player_0"),),
    )

    assert env._retro_env is fake_backend
    assert resources.game_runtime.backend is fake_backend
    assert resources.game_runtime.backend_kind == "retro"

    ArenaResourceControl().release(resources)

    assert fake_backend.closed is True
    assert fake_backend.terminated is True
    assert fake_backend.reaped is True


def test_realtime_retro_real_path_adopts_backend_even_if_reset_fails(monkeypatch) -> None:
    fake_backend = _FakeRetroBackend()

    monkeypatch.setattr(StableRetroArenaEnvironment, "_make_env", lambda self: fake_backend)

    def failing_reset(self):
        self._ensure_env()
        raise RuntimeError("reset failed")

    monkeypatch.setattr(StableRetroArenaEnvironment, "reset", failing_reset)

    kit = build_retro_platformer_game_kit()
    resolved = _build_resolved_from_kit(
        kit,
        env_factory=build_retro_mario_environment,
        env_defaults_overrides={"backend_mode": "real"},
    )
    resources = ArenaResourceControl().allocate(resolved.resource_spec)

    with pytest.raises(RuntimeError, match="reset failed"):
        build_retro_mario_environment(
            sample=_build_sample(game_kit="retro_platformer", env="retro_mario"),
            resolved=resolved,
            resources=resources,
            player_specs=(SimpleNamespace(player_id="player_0", display_name="player_0"),),
        )

    assert resources.game_runtime.backend is fake_backend
    assert resources.game_runtime.backend_kind == "retro"

    ArenaResourceControl().release(resources)

    assert fake_backend.close_calls == 1
    assert fake_backend.terminate_calls == 1
    assert fake_backend.reap_calls == 1


def test_runtime_lease_cleanup_is_idempotent() -> None:
    resources = ArenaResourceControl().allocate({"env_id": "duel_map01", "family": "vizdoom"})
    fake_backend = _FakeViZDoomBackend()
    resources.game_runtime.adopt_backend(fake_backend, backend_kind="vizdoom")

    control = ArenaResourceControl()
    control.release(resources)
    control.release(resources)

    assert fake_backend.close_calls == 1
    assert fake_backend.terminate_calls == 1
    assert fake_backend.reap_calls == 1


@pytest.mark.parametrize(
    ("kit_builder", "env_builder", "game_kit", "env_id", "player_count"),
    [
        (
            build_vizdoom_game_kit,
            build_duel_map01_environment,
            "vizdoom",
            "duel_map01",
            2,
        ),
        (
            build_retro_platformer_game_kit,
            build_retro_mario_environment,
            "retro_platformer",
            "retro_mario",
            1,
        ),
    ],
)
def test_realtime_game_arena_core_reuses_runtime_backend_and_releases_it(
    kit_builder,
    env_builder,
    game_kit: str,
    env_id: str,
    player_count: int,
) -> None:
    kit = kit_builder()
    env_spec = kit.env_catalog[0]
    captured: dict[str, object] = {}

    def capturing_env_factory(*, sample, resolved, resources, player_specs):
        captured["resources"] = resources
        return env_builder(
            sample=sample,
            resolved=resolved,
            resources=resources,
            player_specs=player_specs,
        )

    resolved = SimpleNamespace(
        game_kit=SimpleNamespace(
            kit_id=kit.kit_id,
            defaults=dict(kit.defaults),
            seat_spec=dict(kit.seat_spec),
        ),
        env_spec=SimpleNamespace(
            env_id=env_spec.env_id,
            defaults={
                **dict(env_spec.defaults),
                "env_factory": capturing_env_factory,
                "backend_mode": "dummy",
            },
        ),
        resource_spec=env_spec.resource_spec,
        scheduler=SimpleNamespace(run=lambda session: None, defaults={}),
        players=tuple(
            {"player_id": f"p{index}", "display_name": f"P{index}"}
            for index in range(player_count)
        ),
        observation_workflow=None,
    )

    class FakeResolver:
        def resolve(self, sample):
            captured["sample"] = sample
            return resolved

    class FakeOutputWriter:
        def finalize(self, session):
            captured["session"] = session
            return {"tick": session.tick, "step": session.step}

    core = GameArenaCore(
        resolver=FakeResolver(),
        resource_control=ArenaResourceControl(),
        output_writer=FakeOutputWriter(),
    )
    sample = ArenaSample(
        game_kit=game_kit,
        env=env_id,
        players=tuple(
            {"player_id": f"p{index}", "display_name": f"P{index}"}
            for index in range(player_count)
        ),
    )

    output = core.run_sample(sample)

    runtime = captured["resources"].game_runtime
    assert output["tick"] >= 0
    assert runtime.backend is not None
    if env_id == "duel_map01":
        assert getattr(captured["session"].environment, "_env") is runtime.backend
    if env_id == "retro_mario":
        assert getattr(captured["session"].environment, "_retro_env") is runtime.backend
    assert runtime.backend.closed is True
    assert runtime.backend.terminated is True
    assert runtime.backend.reaped is True


def test_game_arena_core_allocates_builds_and_releases_runtime_resources() -> None:
    kit = build_gomoku_game_kit()
    env_spec = kit.env_catalog[0]
    captured: dict[str, object] = {}

    def capturing_env_factory(*, sample, resolved, resources, player_specs):
        captured["sample"] = sample
        captured["resolved"] = resolved
        captured["resources"] = resources
        captured["player_specs"] = player_specs
        return build_gomoku_standard_environment(
            sample=sample,
            resolved=resolved,
            resources=resources,
            player_specs=player_specs,
        )

    resolved = SimpleNamespace(
        game_kit=SimpleNamespace(
            kit_id=kit.kit_id,
            defaults=dict(kit.defaults),
            seat_spec=dict(kit.seat_spec),
        ),
        env_spec=SimpleNamespace(
            env_id=env_spec.env_id,
            defaults={**dict(env_spec.defaults), "env_factory": capturing_env_factory},
        ),
        resource_spec=env_spec.resource_spec,
        scheduler=SimpleNamespace(run=lambda session: None, defaults={}),
        players=(
            {"player_id": "Black", "display_name": "Black"},
            {"player_id": "White", "display_name": "White"},
        ),
        observation_workflow=None,
    )

    class FakeResolver:
        def resolve(self, sample):
            captured["resolved_sample"] = sample
            return resolved

    class FakeOutputWriter:
        def finalize(self, session):
            captured["session"] = session
            return {"tick": session.tick, "step": session.step}

    core = GameArenaCore(
        resolver=FakeResolver(),
        resource_control=ArenaResourceControl(),
        output_writer=FakeOutputWriter(),
    )
    sample = ArenaSample(
        game_kit=kit.kit_id,
        env=env_spec.env_id,
        players=(
            {"player_id": "Black", "display_name": "Black"},
            {"player_id": "White", "display_name": "White"},
        ),
    )

    output = core.run_sample(sample)

    assert output == {"tick": 0, "step": 0}
    assert captured["resolved_sample"] is sample
    assert captured["resources"].game_bridge is not None
    assert captured["resources"].game_bridge.runtime is captured["resources"].game_runtime
    assert getattr(captured["resources"].game_runtime, "closed", False) is True
    assert getattr(captured["resources"].game_runtime, "terminated", False) is True
    assert getattr(captured["resources"].game_runtime, "reaped", False) is True
    assert getattr(captured["session"].environment, "_runtime_bridge") is captured["resources"].game_bridge
