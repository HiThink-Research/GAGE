from __future__ import annotations

import importlib
from pathlib import Path

import pytest
import yaml

from gage_eval.role.arena.core.types import ArenaSample


REPO_ROOT = Path(__file__).resolve().parents[3]


def test_runtime_binding_resolver_prefers_sample_env_over_default(fake_game_kit_registry) -> None:
    runtime_binding = importlib.import_module("gage_eval.game_kits.runtime_binding")

    resolver = runtime_binding.RuntimeBindingResolver(game_kits=fake_game_kit_registry)
    sample = ArenaSample(
        game_kit="arc_suite_v1",
        env="case_002",
        players=({"seat": "alpha", "player_kind": "human"},),
    )

    resolved = resolver.resolve(sample)

    assert resolved.game_kit.kit_id == "arc_suite_v1"
    assert resolved.env_spec.env_id == "case_002"
    assert resolved.resource_spec == {"env_id": "case_002", "source": "sample"}
    assert resolved.players == sample.players
    assert resolved.observation_workflow.workflow_id == "noop_observation_v1"


def test_runtime_binding_resolver_uses_default_env_when_sample_env_missing(fake_game_kit_registry) -> None:
    runtime_binding = importlib.import_module("gage_eval.game_kits.runtime_binding")

    resolver = runtime_binding.RuntimeBindingResolver(game_kits=fake_game_kit_registry)
    sample = ArenaSample(game_kit="arc_suite_v1", env=None)

    resolved = resolver.resolve(sample)

    assert resolved.env_spec.env_id == "case_001"
    assert resolved.resource_spec == {"env_id": "case_001", "source": "default"}


def test_runtime_binding_resolver_honors_suite_level_env_override(fake_game_kit_registry) -> None:
    runtime_binding = importlib.import_module("gage_eval.game_kits.runtime_binding")

    resolver = runtime_binding.RuntimeBindingResolver(game_kits=fake_game_kit_registry)
    sample = ArenaSample(
        game_kit="arc_suite_v1",
        env=None,
        runtime_overrides={"env": "case_002"},
    )

    resolved = resolver.resolve(sample)

    assert resolved.env_spec.env_id == "case_002"
    assert resolved.resource_spec == {"env_id": "case_002", "source": "sample"}


def test_runtime_binding_resolver_defaults_single_env_kit_without_explicit_env(fake_game_kit_registry) -> None:
    runtime_binding = importlib.import_module("gage_eval.game_kits.runtime_binding")

    resolver = runtime_binding.RuntimeBindingResolver(game_kits=fake_game_kit_registry)
    sample = ArenaSample(game_kit="gomoku", env=None)

    resolved = resolver.resolve(sample)

    assert resolved.env_spec.env_id == "gomoku_standard"
    assert resolved.resource_spec == {"env_id": "gomoku_standard"}


def test_runtime_binding_resolver_parses_scheduler_owned_queued_command_v1_policy() -> None:
    runtime_binding = importlib.import_module("gage_eval.game_kits.runtime_binding")
    registry_module = importlib.import_module("gage_eval.game_kits.registry")

    sample = ArenaSample(
        game_kit="openra",
        env="ra_skirmish_1v1",
        players=(
            {
                "seat": "player_0",
                "player_id": "player_0",
                "player_kind": "human",
                "driver_params": {
                    "input_semantics": "queued_command",
                    "tick_interval_ms": 50,
                    "timeout_ms": 50,
                },
            },
        ),
        runtime_overrides={
            "runtime_binding_policy_config": {
                "mode": "scheduler_owned_human_realtime",
                "activation_scope": "pure_human_only",
                "input_model": "queued_command",
                "input_transport": "realtime_ws",
                "tick_interval_ms": 50,
                "max_commands_per_tick": 4,
                "max_command_queue_size": 128,
                "bridge_stall_timeout_ms": 2000,
            }
        },
    )

    resolved = runtime_binding.RuntimeBindingResolver(game_kits=registry_module.GameKitRegistry()).resolve(
        sample
    )

    profile = resolved.runtime_profile.realtime_human_control
    assert profile is not None
    assert profile.max_commands_per_tick == 4
    assert profile.max_command_queue_size == 128
    assert profile.bridge_stall_timeout_ms == 2000
    assert profile.fallback_move is None


@pytest.mark.parametrize(
    ("config_relpath", "expected_driver_params"),
    [
            (
                "config/custom/retro_mario/retro_mario_human_visual_gamekit.yaml",
                {
                    "input_semantics": "continuous_state",
                    "stateful_actions": True,
                    "tick_interval_ms": 16,
                    "timeout_ms": 16,
                    "timeout_fallback_move": "noop",
                },
            ),
            (
                "config/custom/vizdoom/vizdoom_human_visual_gamekit.yaml",
                {
                    "input_semantics": "continuous_state",
                    "stateful_actions": True,
                    "tick_interval_ms": 16,
                    "timeout_ms": 16,
                    "timeout_fallback_move": "0",
                },
            ),
            (
                "config/custom/openra/openra_ra_skirmish_native_pure_human_visual.yaml",
                {
                    "input_semantics": "queued_command",
                    "tick_interval_ms": 50,
                    "timeout_ms": 50,
                    "timeout_fallback_move": "noop",
                },
            ),
    ],
)
def test_runtime_binding_resolver_preserves_human_driver_params_from_config(
    config_relpath: str,
    expected_driver_params: dict[str, object],
) -> None:
    runtime_binding = importlib.import_module("gage_eval.game_kits.runtime_binding")
    registry_module = importlib.import_module("gage_eval.game_kits.registry")

    payload = yaml.safe_load(
        (REPO_ROOT / config_relpath).read_text(encoding="utf-8")
    )
    params = payload["role_adapters"][0]["params"]
    sample = ArenaSample(
        game_kit=str(params["game_kit"]),
        env=str(params["env"]),
        players=tuple(dict(player) for player in params["players"]),
        runtime_overrides=dict(params["runtime_overrides"]),
    )
    resolver = runtime_binding.RuntimeBindingResolver(game_kits=registry_module.GameKitRegistry())

    resolved = resolver.resolve(sample)

    assert resolved.scheduler.binding_id == "real_time_tick/default"
    assert resolved.player_bindings[0].driver_id == "player_driver/human_local_input"
    assert resolved.player_bindings[0].driver_params == expected_driver_params


@pytest.mark.parametrize(
    (
        "config_relpath",
        "expected_semantics",
        "expected_tick_ms",
        "expected_pure_human",
        "expected_low_latency",
    ),
    [
        (
            "config/custom/retro_mario/retro_mario_human_visual_gamekit.yaml",
            "continuous_state",
            16,
            True,
            True,
        ),
        (
            "config/custom/vizdoom/vizdoom_human_visual_gamekit.yaml",
            "continuous_state",
            16,
            False,
            False,
        ),
        (
            "config/custom/openra/openra_ra_skirmish_native_pure_human_visual.yaml",
            "queued_command",
            50,
            True,
            True,
        ),
    ],
)
def test_runtime_binding_resolver_builds_realtime_runtime_profile_from_config(
    config_relpath: str,
    expected_semantics: str,
    expected_tick_ms: int,
    expected_pure_human: bool,
    expected_low_latency: bool,
) -> None:
    runtime_binding = importlib.import_module("gage_eval.game_kits.runtime_binding")
    registry_module = importlib.import_module("gage_eval.game_kits.registry")

    payload = yaml.safe_load(
        (REPO_ROOT / config_relpath).read_text(encoding="utf-8")
    )
    params = payload["role_adapters"][0]["params"]
    sample = ArenaSample(
        game_kit=str(params["game_kit"]),
        env=str(params["env"]),
        players=tuple(dict(player) for player in params["players"]),
        runtime_overrides=dict(params["runtime_overrides"]),
    )
    resolver = runtime_binding.RuntimeBindingResolver(game_kits=registry_module.GameKitRegistry())

    resolved = resolver.resolve(sample)

    assert resolved.runtime_profile is not None
    assert resolved.runtime_profile.scheduler_binding == "real_time_tick/default"
    assert resolved.runtime_profile.scheduler_family == "real_time_tick"
    assert resolved.runtime_profile.tick_interval_ms == expected_tick_ms
    assert resolved.runtime_profile.pure_human_realtime is expected_pure_human
    assert resolved.runtime_profile.supports_low_latency_realtime_input is expected_low_latency
    assert resolved.runtime_profile.supports_realtime_input_websocket is expected_low_latency

    human_profile = resolved.runtime_profile.human_realtime_inputs[0]
    assert human_profile.player_id == resolved.player_bindings[0].player_id
    assert human_profile.semantics == expected_semantics
    assert human_profile.tick_interval_ms == expected_tick_ms

    if expected_pure_human:
        assert resolved.runtime_profile.scheduler_owns_realtime_clock is True
        assert resolved.runtime_profile.realtime_human_control is not None
        assert resolved.runtime_profile.realtime_human_control.mode == "scheduler_owned_human_realtime"
        assert resolved.runtime_profile.realtime_human_control.tick_interval_ms == expected_tick_ms
    else:
        assert resolved.runtime_profile.scheduler_owns_realtime_clock is False
        assert resolved.runtime_profile.realtime_human_control is None


def test_runtime_binding_resolver_requires_explicit_scheduler_owned_policy_for_realtime_input_capabilities() -> None:
    runtime_binding = importlib.import_module("gage_eval.game_kits.runtime_binding")
    registry_module = importlib.import_module("gage_eval.game_kits.registry")

    payload = yaml.safe_load(
        (REPO_ROOT / "config/custom/vizdoom/vizdoom_human_visual_gamekit.yaml").read_text(
            encoding="utf-8"
        )
    )
    params = payload["role_adapters"][0]["params"]
    runtime_overrides = dict(params["runtime_overrides"])
    runtime_overrides.pop("runtime_binding_policy_config", None)
    sample = ArenaSample(
        game_kit=str(params["game_kit"]),
        env=str(params["env"]),
        players=tuple(dict(player) for player in params["players"]),
        runtime_overrides=runtime_overrides,
    )
    resolver = runtime_binding.RuntimeBindingResolver(game_kits=registry_module.GameKitRegistry())

    resolved = resolver.resolve(sample)

    assert resolved.runtime_profile is not None
    assert resolved.runtime_profile.realtime_human_control is None
    assert resolved.runtime_profile.scheduler_owns_realtime_clock is False
    assert resolved.runtime_profile.supports_low_latency_realtime_input is False
    assert resolved.runtime_profile.supports_realtime_input_websocket is False


def test_runtime_binding_resolver_uses_clone_local_scheduler_binding(fake_game_kit_registry) -> None:
    runtime_binding = importlib.import_module("gage_eval.game_kits.runtime_binding")

    resolver = runtime_binding.RuntimeBindingResolver(game_kits=fake_game_kit_registry)
    sample = ArenaSample(game_kit="clone_scheduler_kit_v1", env=None)

    resolved = resolver.resolve(sample)

    assert resolved.scheduler.binding_id == "clone_only_scheduler_v1"
    assert resolved.scheduler.defaults == {"max_ticks": 99}


def test_runtime_binding_resolver_honors_flat_scheduler_override(fake_game_kit_registry) -> None:
    runtime_binding = importlib.import_module("gage_eval.game_kits.runtime_binding")

    resolver = runtime_binding.RuntimeBindingResolver(game_kits=fake_game_kit_registry)
    sample = ArenaSample(
        game_kit="gomoku",
        env=None,
        scheduler="clone_only_scheduler_v1",
    )

    resolved = resolver.resolve(sample)

    assert resolved.scheduler.binding_id == "clone_only_scheduler_v1"
    assert resolved.scheduler.defaults == {"max_ticks": 99}


def test_runtime_binding_resolver_honors_runtime_scheduler_override(
    fake_game_kit_registry,
) -> None:
    runtime_binding = importlib.import_module("gage_eval.game_kits.runtime_binding")

    resolver = runtime_binding.RuntimeBindingResolver(game_kits=fake_game_kit_registry)
    sample = ArenaSample(
        game_kit="gomoku",
        env=None,
        runtime_overrides={"scheduler": "clone_only_scheduler_v1"},
    )

    resolved = resolver.resolve(sample)

    assert resolved.scheduler.binding_id == "clone_only_scheduler_v1"
    assert resolved.scheduler.defaults == {"max_ticks": 99}


def test_runtime_binding_resolver_honors_runtime_scheduler_binding_alias(
    fake_game_kit_registry,
) -> None:
    runtime_binding = importlib.import_module("gage_eval.game_kits.runtime_binding")

    resolver = runtime_binding.RuntimeBindingResolver(game_kits=fake_game_kit_registry)
    sample = ArenaSample(
        game_kit="gomoku",
        env=None,
        runtime_overrides={"scheduler_binding": "clone_only_scheduler_v1"},
    )

    resolved = resolver.resolve(sample)

    assert resolved.scheduler.binding_id == "clone_only_scheduler_v1"
    assert resolved.scheduler.defaults == {"max_ticks": 99}


def test_runtime_binding_resolver_prefers_flat_scheduler_over_runtime_override(
    fake_game_kit_registry,
) -> None:
    runtime_binding = importlib.import_module("gage_eval.game_kits.runtime_binding")

    resolver = runtime_binding.RuntimeBindingResolver(game_kits=fake_game_kit_registry)
    sample = ArenaSample(
        game_kit="gomoku",
        env=None,
        scheduler="turn/default",
        runtime_overrides={"scheduler": "clone_only_scheduler_v1"},
    )

    resolved = resolver.resolve(sample)

    assert resolved.scheduler.binding_id == "turn/default"
    assert resolved.scheduler.defaults == {"max_ticks": 256}


@pytest.mark.parametrize(
    "legacy_spec",
    [
        {"seat": "player_0", "type": "backend", "ref": "qwen_backend"},
        {"seat": "enemy_bot", "player_kind": "dummy", "moves": ["RIGHT"]},
    ],
)
def test_runtime_binding_resolver_rejects_legacy_player_fields(
    fake_game_kit_registry,
    legacy_spec: dict[str, object],
) -> None:
    runtime_binding = importlib.import_module("gage_eval.game_kits.runtime_binding")

    resolver = runtime_binding.RuntimeBindingResolver(game_kits=fake_game_kit_registry)
    sample = ArenaSample(
        game_kit="retro_platformer_players_v1",
        env="retro_mario",
        players=(legacy_spec,),
    )

    with pytest.raises(ValueError, match="legacy player fields"):
        resolver.resolve(sample)


def test_runtime_binding_resolver_raises_for_unknown_env(fake_game_kit_registry) -> None:
    runtime_binding = importlib.import_module("gage_eval.game_kits.runtime_binding")

    resolver = runtime_binding.RuntimeBindingResolver(game_kits=fake_game_kit_registry)
    sample = ArenaSample(game_kit="arc_suite_v1", env="missing_case")

    with pytest.raises(KeyError, match="missing_case"):
        resolver.resolve(sample)


def test_runtime_binding_resolver_supports_spec_backed_game_kit(fake_game_kit_registry) -> None:
    runtime_binding = importlib.import_module("gage_eval.game_kits.runtime_binding")

    resolver = runtime_binding.RuntimeBindingResolver(game_kits=fake_game_kit_registry)
    sample = ArenaSample(game_kit="arena/default", env=None)

    resolved = resolver.resolve(sample)

    assert resolved.game_kit.kit_id == "arena/default"
    assert resolved.env_spec.env_id == "arena/default"
    assert resolved.scheduler.binding_id == "turn/default"


@pytest.mark.parametrize(
    ("env_id", "expected_resource_spec", "expected_content_refs"),
    [
        (
            "ra_map01",
            {"env_id": "ra_map01", "family": "openra"},
            {
                "mod": "mod/openra/ra",
                "map": "map/openra/ra_map01",
            },
        ),
        (
            "ra_skirmish_1v1",
            {"env_id": "ra_skirmish_1v1", "family": "openra", "mod": "ra"},
            {
                "mod": "mod/openra/ra",
                "map": "map/openra/ra/marigold-town.oramap",
            },
        ),
        (
            "cnc_mission_gdi01",
            {"env_id": "cnc_mission_gdi01", "family": "openra", "mod": "cnc"},
            {
                "mod": "mod/openra/cnc",
                "map": "map/openra/cnc/gdi01",
                "script": "script/openra/gdi01.lua",
            },
        ),
        (
            "d2k_skirmish_1v1",
            {"env_id": "d2k_skirmish_1v1", "family": "openra", "mod": "d2k"},
            {
                "mod": "mod/openra/d2k",
                "map": "map/openra/d2k/chin-rock.oramap",
            },
        ),
    ],
)
def test_runtime_binding_resolver_resolves_openra_realtime_kit(
    env_id: str,
    expected_resource_spec: dict[str, str],
    expected_content_refs: dict[str, str],
) -> None:
    runtime_binding = importlib.import_module("gage_eval.game_kits.runtime_binding")
    registry_module = importlib.import_module("gage_eval.game_kits.registry")

    resolver = runtime_binding.RuntimeBindingResolver(game_kits=registry_module.GameKitRegistry())
    sample = ArenaSample(
        game_kit="openra",
        env=env_id,
        runtime_overrides={"backend_mode": "dummy"},
    )

    resolved = resolver.resolve(sample)

    assert resolved.game_kit.kit_id == "openra"
    assert resolved.env_spec.env_id == env_id
    assert resolved.scheduler.binding_id == "real_time_tick/default"
    assert resolved.resource_spec == expected_resource_spec
    assert resolved.visualization_spec is not None
    assert resolved.visualization_spec.spec_id == "arena/visualization/openra_rts_v1"
    assert resolved.visualization_spec.plugin_id == "arena.visualization.openra.rts_v1"
    assert resolved.input_mapper == (
        "gage_eval.game_kits.real_time_game.openra.input_mapper.OpenRAInputMapper"
    )
    for key, value in expected_content_refs.items():
        assert resolved.game_content_refs[key] == value


def test_runtime_binding_resolver_resolves_support_workflow(fake_game_kit_registry) -> None:
    runtime_binding = importlib.import_module("gage_eval.game_kits.runtime_binding")
    support_registry_module = importlib.import_module("gage_eval.role.arena.support.registry")

    resolver = runtime_binding.RuntimeBindingResolver(
        game_kits=fake_game_kit_registry,
        support_workflows=support_registry_module.SupportWorkflowRegistry(
            registry_view=fake_game_kit_registry.registry_view
        ),
    )
    sample = ArenaSample(game_kit="gomoku", env=None)

    resolved = resolver.resolve(sample)

    assert resolved.support_workflow.workflow_id == "arena/default"
    assert resolved.support_workflow.units_by_hook
    assert resolved.support_workflow.units_by_hook


def test_runtime_binding_resolver_surfaces_extension_refs_with_explicit_precedence() -> None:
    contracts = importlib.import_module("gage_eval.game_kits.contracts")
    registry_module = importlib.import_module("gage_eval.game_kits.registry")
    runtime_binding = importlib.import_module("gage_eval.game_kits.runtime_binding")
    registry_core = importlib.import_module("gage_eval.registry")

    clone = registry_core.registry.clone()
    clone.register(
        "visualization_specs",
        "display/sample",
        contracts.GameVisualizationSpec(
            spec_id="display/sample",
            plugin_id="arena.visualization.tests.sample",
            visual_kind="board",
        ),
        desc="Sample-level game display for resolver precedence tests",
    )
    clone.register(
        "game_kits",
        "binding_surface_v1",
        contracts.GameKit(
            kit_id="binding_surface_v1",
            family="suite",
            scheduler_binding="turn/default",
            support_workflow="arena/default",
            observation_workflow="noop_observation_v1",
            env_catalog=(
                contracts.EnvSpec(
                    env_id="env_v1",
                    kit_id="binding_surface_v1",
                    resource_spec={"env_id": "env_v1"},
                    game_content_refs={
                        "kit_only": "content/kit",
                        "env_only": "content/env",
                        "shared": "env",
                    },
                    runtime_binding_policy="policy/env",
                    game_display="display/env",
                    replay_viewer="viewer/env",
                    parser="parser/env",
                    renderer="renderer/env",
                    replay_policy="replay/env",
                    input_mapper="input/env",
                ),
            ),
            default_env="env_v1",
            seat_spec={"seats": ("alpha", "beta")},
            game_content_refs={
                "kit_only": "content/kit",
                "shared": "kit",
            },
            runtime_binding_policy="policy/kit",
            game_display="display/kit",
            replay_viewer="viewer/kit",
            parser="parser/kit",
            renderer="renderer/kit",
            replay_policy="replay/kit",
            input_mapper="input/kit",
        ),
        desc="Game kit with explicit extension refs for precedence tests",
    )

    resolver = runtime_binding.RuntimeBindingResolver(
        game_kits=registry_module.GameKitRegistry(registry_view=clone)
    )
    sample = ArenaSample(
        game_kit="binding_surface_v1",
        env="env_v1",
        runtime_overrides={
            "game_display": "display/sample",
            "replay_viewer": "viewer/sample",
            "parser": "parser/sample",
            "game_content_refs": {
                "sample_only": "content/sample",
                "shared": "sample",
            },
        },
    )

    resolved = resolver.resolve(sample)

    assert resolved.runtime_binding_policy == "policy/env"
    assert resolved.game_display == "display/sample"
    assert resolved.replay_viewer == "viewer/sample"
    assert resolved.parser == "parser/sample"
    assert resolved.renderer == "renderer/env"
    assert resolved.replay_policy == "replay/env"
    assert resolved.input_mapper == "input/env"
    assert resolved.game_content_refs == {
        "kit_only": "content/kit",
        "env_only": "content/env",
        "sample_only": "content/sample",
        "shared": "sample",
    }
    assert resolved.visualization_spec.spec_id == "display/sample"
