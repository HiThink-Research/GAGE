from __future__ import annotations

import importlib
import sys

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
        "supported_observers": ("global", "spectator", "camera", "player"),
    },
    "mahjong": {
        "env": "riichi_4p",
        "spec_id": "arena/visualization/mahjong_table_v1",
        "plugin_id": "arena.visualization.mahjong.table_v1",
        "visual_kind": "table",
        "supported_observers": ("global", "spectator", "camera", "player"),
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
    if kit_id == "doudizhu":
        assert resolved.visualization_spec.action_schema["action_metadata"] == {
            "descriptor": "doudizhu_table_actions_v1",
            "legal_action_source": "scene.legalActions",
            "selection_source": "table.seats[].hand.cards",
            "typed_actions": ["play_cards", "pass"],
        }
    elif kit_id == "mahjong":
        assert resolved.visualization_spec.action_schema["action_metadata"] == {
            "descriptor": "mahjong_table_actions_v1",
            "legal_action_source": "scene.legalActions",
            "selection_source": "table.seats[].hand.cards",
            "typed_actions": ["discard_tile", "call_meld", "declare_win", "pass"],
        }
    else:
        assert resolved.visualization_spec.action_schema["action_metadata"] == {
            "descriptor": "placeholder"
        }


def test_visualization_spec_plugin_ids_are_unique_across_current_games() -> None:
    plugin_ids = {
        _resolve_visualization_spec(kit_id).visualization_spec.plugin_id
        for kit_id in EXPECTED_VISUALIZATION_SPECS
    }

    assert len(plugin_ids) == len(EXPECTED_VISUALIZATION_SPECS)


def test_visualization_resolution_does_not_load_legacy_arena_game_modules() -> None:
    legacy_prefix = ".".join(("gage_eval", "role", "arena", "games"))
    for module_name in list(sys.modules):
        if module_name == legacy_prefix or module_name.startswith(f"{legacy_prefix}."):
            sys.modules.pop(module_name, None)

    _resolve_visualization_spec("gomoku")
    _resolve_visualization_spec("tictactoe")

    assert not any(
        module_name == legacy_prefix or module_name.startswith(f"{legacy_prefix}.")
        for module_name in sys.modules
    )


def test_mahjong_visualization_spec_declares_structured_table_scene_extensions() -> None:
    resolved = _resolve_visualization_spec("mahjong")
    rules = resolved.visualization_spec.scene_projection_rules

    assert rules["default_layout"] == "four-seat"
    assert rules["scene_contract"] == {
        "table": {
            "seat_extensions": ["meldGroups", "drawTile", "hand.drawTile"],
            "center_extensions": ["history", "discardLanes"],
            "status_extensions": [
                "privateViewPlayerId",
                "lastDiscard",
                "winner",
                "result",
                "resultReason",
                "remainingTiles",
            ],
            "panel_extensions": ["chatLog", "events", "trace"],
        }
    }
    assert resolved.visualization_spec.observer_schema["descriptor"] == (
        "mahjong_table_observer_schema_v1"
    )
    assert resolved.visualization_spec.observer_schema["default_mode"] == "spectator"
    assert resolved.visualization_spec.observer_schema["host_selection"] == {
        "default_mode": "spectator",
        "player_requires_observer_id": True,
        "supports_observer_switching": True,
    }
    assert resolved.visualization_spec.observer_schema["mode_semantics"] == {
        "global": {"label": "Global table", "private_hand_visible": False},
        "spectator": {"label": "Spectator", "private_hand_visible": False},
        "camera": {"label": "Broadcast camera", "private_hand_visible": False},
        "player": {"label": "Player seat", "private_hand_visible": True},
    }
    assert resolved.visualization_spec.action_schema["action_types"] == [
        {
            "id": "discard_tile",
            "label": "Discard tile",
            "payload": {"actionText": "<tile code>"},
        },
        {
            "id": "call_meld",
            "label": "Call meld",
            "payload": {"actionText": "Pong|Chow|Kong"},
        },
        {
            "id": "declare_win",
            "label": "Declare win",
            "payload": {"actionText": "Hu"},
        },
        {
            "id": "pass",
            "label": "Pass",
            "payload": {"actionText": "Pass"},
        },
    ]
    assert resolved.visualization_spec.action_schema["selection_model"] == {
        "source": "hand.cards",
        "draw_tile_source": "hand.drawTile",
        "supports_multi_select": False,
        "confirm_required": True,
    }
    assert resolved.visualization_spec.timeline_annotation_rules == {
        "descriptor": "mahjong_table_timeline_v1",
        "focus_event_types": ["decision_window_open", "action_intent", "result"],
        "event_labels": {
            "decision_window_open": "Discard or call window",
            "action_intent": "Mahjong action submitted",
            "result": "Hand result",
        },
        "annotations": [
            {
                "id": "discard_commit",
                "label": "Discard selected",
                "match_action_type": "discard_tile",
            },
            {
                "id": "call_declared",
                "label": "Call declared",
                "match_action_regex": "^(pong|chow|kong|gong)$",
            },
            {"id": "win_declared", "label": "Win declared", "match_action": "hu"},
            {"id": "hand_result", "label": "Hand result", "match_event": "result"},
        ],
    }


def test_doudizhu_visualization_spec_declares_structured_table_semantics() -> None:
    resolved = _resolve_visualization_spec("doudizhu")
    rules = resolved.visualization_spec.scene_projection_rules

    assert rules["default_layout"] == "three-seat"
    assert rules["scene_contract"] == {
        "table": {
            "seat_extensions": ["role", "playedCards", "hand.maskedCount"],
            "center_extensions": ["history"],
            "status_extensions": ["privateViewPlayerId", "landlordId"],
            "panel_extensions": ["chatLog", "events", "trace"],
        }
    }
    assert resolved.visualization_spec.action_schema["action_types"] == [
        {
            "id": "play_cards",
            "label": "Play cards",
            "payload": {"actionText": "<card pattern>"},
        },
        {
            "id": "pass",
            "label": "Pass",
            "payload": {"actionText": "pass"},
        },
    ]
    assert resolved.visualization_spec.action_schema["selection_model"] == {
        "source": "hand.cards",
        "supports_multi_select": True,
        "pass_action_text": "pass",
    }
    assert resolved.visualization_spec.timeline_annotation_rules == {
        "descriptor": "doudizhu_table_timeline_v1",
        "focus_event_types": ["decision_window_open", "action_intent", "result"],
        "event_labels": {
            "decision_window_open": "Turn opens",
            "action_intent": "Action submitted",
            "result": "Round result",
        },
        "annotations": [
            {"id": "pass_chain", "label": "Pass chain", "match_action": "pass"},
            {"id": "lead_play", "label": "Lead play", "match_event": "decision_window_open"},
            {"id": "round_result", "label": "Round result", "match_event": "result"},
        ],
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
