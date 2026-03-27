from __future__ import annotations

import importlib

import pytest

from gage_eval.role.arena.core.types import ArenaSample


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
