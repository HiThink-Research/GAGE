from __future__ import annotations

import importlib

import pytest

from gage_eval.role.arena.core.types import ArenaSample


EXPECTED_VISUALIZATION_SPECS = {
    "gomoku": {
        "env": "gomoku_standard",
        "spec_id": "arena/visualization/gomoku_board_v1",
        "plugin_id": "arena.visualization.gomoku.board_v1",
        "visual_kind": "board",
        "supported_observers": ("player", "global"),
    },
    "tictactoe": {
        "env": "tictactoe_standard",
        "spec_id": "arena/visualization/tictactoe_board_v1",
        "plugin_id": "arena.visualization.tictactoe.board_v1",
        "visual_kind": "board",
        "supported_observers": ("player", "global"),
    },
    "doudizhu": {
        "env": "classic_3p",
        "spec_id": "arena/visualization/doudizhu_table_v1",
        "plugin_id": "arena.visualization.doudizhu.table_v1",
        "visual_kind": "table",
        "supported_observers": ("player", "global"),
    },
    "mahjong": {
        "env": "riichi_4p",
        "spec_id": "arena/visualization/mahjong_table_v1",
        "plugin_id": "arena.visualization.mahjong.table_v1",
        "visual_kind": "table",
        "supported_observers": ("player", "global"),
    },
    "pettingzoo": {
        "env": "space_invaders",
        "spec_id": "arena/visualization/pettingzoo_frame_v1",
        "plugin_id": "arena.visualization.pettingzoo.frame_v1",
        "visual_kind": "frame",
        "supported_observers": ("player", "global"),
    },
    "vizdoom": {
        "env": "duel_map01",
        "spec_id": "arena/visualization/vizdoom_frame_v1",
        "plugin_id": "arena.visualization.vizdoom.frame_v1",
        "visual_kind": "frame",
        "supported_observers": ("player", "camera"),
    },
    "retro_platformer": {
        "env": "retro_mario",
        "spec_id": "arena/visualization/retro_platformer_frame_v1",
        "plugin_id": "arena.visualization.retro_platformer.frame_v1",
        "visual_kind": "frame",
        "supported_observers": ("player", "camera"),
    },
}


def _resolve_visualization_spec(kit_id: str) -> object:
    runtime_binding = importlib.import_module("gage_eval.game_kits.runtime_binding")
    game_kits = importlib.import_module("gage_eval.game_kits.registry").GameKitRegistry()
    sample = ArenaSample(game_kit=kit_id, env=EXPECTED_VISUALIZATION_SPECS[kit_id]["env"])
    resolver = runtime_binding.RuntimeBindingResolver(game_kits=game_kits)
    return resolver.resolve(sample)


@pytest.mark.parametrize(("kit_id", "expected"), EXPECTED_VISUALIZATION_SPECS.items())
def test_visualization_spec_resolution_uses_explicit_spec_id(
    kit_id: str,
    expected: dict[str, str],
) -> None:
    resolved = _resolve_visualization_spec(kit_id)

    assert resolved.game_kit.visualization_spec == expected["spec_id"]
    assert resolved.visualization_spec.spec_id == expected["spec_id"]
    assert resolved.visualization_spec.plugin_id == expected["plugin_id"]
    assert resolved.visualization_spec.visual_kind == expected["visual_kind"]
    assert tuple(resolved.visualization_spec.observer_schema["supported_modes"]) == expected[
        "supported_observers"
    ]
    assert resolved.visualization_spec.action_schema["action_metadata"] == {
        "descriptor": "placeholder"
    }


def test_visualization_spec_plugin_ids_are_unique_across_current_games() -> None:
    plugin_ids = {
        _resolve_visualization_spec(kit_id).visualization_spec.plugin_id
        for kit_id in EXPECTED_VISUALIZATION_SPECS
    }

    assert len(plugin_ids) == len(EXPECTED_VISUALIZATION_SPECS)


def test_mahjong_visualization_spec_declares_structured_table_scene_extensions() -> None:
    resolved = _resolve_visualization_spec("mahjong")
    rules = resolved.visualization_spec.scene_projection_rules

    assert rules["default_layout"] == "four-seat"
    assert rules["scene_contract"] == {
        "table": {
            "seat_extensions": ["meldGroups", "drawTile", "hand.drawTile"],
            "center_extensions": ["discardLanes"],
            "status_extensions": ["lastDiscard"],
        }
    }


def test_visualization_spec_kinds_cover_expected_stage_set() -> None:
    kinds = {
        _resolve_visualization_spec(kit_id).visualization_spec.visual_kind
        for kit_id in EXPECTED_VISUALIZATION_SPECS
    }

    assert kinds == {"board", "table", "frame"}


def test_legacy_arena_default_visualization_spec_adapts_through_clone() -> None:
    registry_module = importlib.import_module("gage_eval.registry")
    support_specs = importlib.import_module("gage_eval.role.arena.support.specs")
    visualization_specs = importlib.import_module("gage_eval.game_kits.visualization_specs")

    clone = registry_module.registry.clone()
    with registry_module.registry.route_to(clone):
        importlib.reload(support_specs)

    resolved = visualization_specs.VisualizationSpecRegistry(registry_view=clone).build(
        "arena/default"
    )

    assert resolved.spec_id == "arena/default"
    assert resolved.renderer_impl == support_specs.DEFAULT_VISUALIZATION_SPEC.renderer_impl
    assert resolved.visual_kind == "frame"
