from __future__ import annotations

from gage_eval.registry import import_asset_from_manifest, registry


def test_gamearena_registry_kinds_declared() -> None:
    expected = {
        "game_kits",
        "scheduler_bindings",
        "support_workflows",
        "support_units",
        "observation_workflows",
        "visualization_specs",
        "game_content_assets",
        "player_drivers",
    }
    assert expected.issubset(set(registry.kinds()))


def test_gamearena_default_specs_use_new_driver_defaults() -> None:
    from gage_eval.role.arena.schedulers.specs import DEFAULT_SCHEDULER_BINDING
    from gage_eval.role.arena.support.specs import (
        DEFAULT_AGENT_PLAYER_DRIVER,
        DEFAULT_HUMAN_PLAYER_DRIVER,
        DEFAULT_LLM_PLAYER_DRIVER,
        DEFAULT_OBSERVATION_WORKFLOW,
        DEFAULT_PLAYER_DRIVER,
        DEFAULT_SUPPORT_UNIT,
        DEFAULT_VISUALIZATION_SPEC,
    )

    assert DEFAULT_SCHEDULER_BINDING.scheduler_impl == "turn"
    assert DEFAULT_SUPPORT_UNIT.impl.startswith("placeholder://")
    assert DEFAULT_OBSERVATION_WORKFLOW.impl.startswith("placeholder://")
    assert DEFAULT_VISUALIZATION_SPEC.renderer_impl.startswith("placeholder://")
    assert DEFAULT_PLAYER_DRIVER.driver_id == "player_driver/dummy"
    assert DEFAULT_PLAYER_DRIVER.impl == "dummy"
    assert DEFAULT_HUMAN_PLAYER_DRIVER.impl == "human_local_input"
    assert DEFAULT_LLM_PLAYER_DRIVER.impl == "llm_backend"
    assert DEFAULT_AGENT_PLAYER_DRIVER.impl == "agent_role_stub"


def test_gamearena_manifest_bootstrap_entries_importable() -> None:
    clone = registry.clone()
    expected_bootstrap = {
        "game_kits": "arena/default",
        "scheduler_bindings": "turn/default",
        "support_workflows": "arena/default",
        "support_units": "arena/default",
        "observation_workflows": "arena/default",
        "visualization_specs": "arena/default",
        "game_content_assets": "arena/default_content",
        "player_drivers": "player_driver/dummy",
    }

    imported_kinds: set[str] = set()
    with registry.route_to(clone):
        for kind, name in expected_bootstrap.items():
            report = import_asset_from_manifest(kind, name, registry=clone, source="unit-test")
            assert report.ok
            assert len(report.imported) == 1
            imported = report.imported[0]
            assert imported.kind == kind
            assert imported.name == name
            clone.entry(kind, name)
            imported_kinds.add(imported.kind)

    assert imported_kinds == set(expected_bootstrap)
