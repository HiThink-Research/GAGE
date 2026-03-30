"""Mahjong visualization spec."""

from __future__ import annotations

from gage_eval.game_kits.contracts import GameVisualizationSpec
from gage_eval.registry import registry

VISUALIZATION_SPEC_ID = "arena/visualization/mahjong_table_v1"
VISUALIZATION_PLUGIN_ID = "arena.visualization.mahjong.table_v1"
VISUAL_KIND = "table"
RENDERER_IMPL = "placeholder://arena/visualization/mahjong/renderer"

SCENE_PROJECTION_RULES = {
    "impl": "builtin://arena/visualization/table_scene_projection_v1",
    "spec_id": VISUALIZATION_SPEC_ID,
    "plugin_id": VISUALIZATION_PLUGIN_ID,
    "visual_kind": VISUAL_KIND,
    "kit_id": "mahjong",
    "table_game": "mahjong",
    "seat_count": 4,
    "default_layout": "four-seat",
    "scene_contract": {
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
    },
}
ACTION_SCHEMA = {
    "descriptor": "mahjong_table_action_schema_v1",
    "action_metadata": {
        "descriptor": "mahjong_table_actions_v1",
        "legal_action_source": "scene.legalActions",
        "selection_source": "table.seats[].hand.cards",
        "typed_actions": ["discard_tile", "call_meld", "declare_win", "pass"],
    },
    "action_types": [
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
    ],
    "selection_model": {
        "source": "hand.cards",
        "draw_tile_source": "hand.drawTile",
        "supports_multi_select": False,
        "confirm_required": True,
    },
}
OBSERVER_SCHEMA = {
    "descriptor": "mahjong_table_observer_schema_v1",
    "supported_modes": ["global", "spectator", "camera", "player"],
    "default_mode": "spectator",
    "host_selection": {
        "default_mode": "spectator",
        "player_requires_observer_id": True,
        "supports_observer_switching": True,
    },
    "mode_semantics": {
        "global": {"label": "Global table", "private_hand_visible": False},
        "spectator": {"label": "Spectator", "private_hand_visible": False},
        "camera": {"label": "Broadcast camera", "private_hand_visible": False},
        "player": {"label": "Player seat", "private_hand_visible": True},
    },
}
TIMELINE_ANNOTATION_RULES = {
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

VISUALIZATION_SPEC = GameVisualizationSpec(
    spec_id=VISUALIZATION_SPEC_ID,
    plugin_id=VISUALIZATION_PLUGIN_ID,
    visual_kind=VISUAL_KIND,
    renderer_impl=RENDERER_IMPL,
    scene_projection_rules=SCENE_PROJECTION_RULES,
    action_schema=ACTION_SCHEMA,
    observer_schema=OBSERVER_SCHEMA,
    timeline_annotation_rules=TIMELINE_ANNOTATION_RULES,
)

registry.register(
    "visualization_specs",
    VISUALIZATION_SPEC_ID,
    VISUALIZATION_SPEC,
    desc="Mahjong table visualization spec",
)

__all__ = [
    "VISUALIZATION_SPEC",
    "VISUALIZATION_SPEC_ID",
    "VISUALIZATION_PLUGIN_ID",
    "VISUAL_KIND",
    "RENDERER_IMPL",
    "SCENE_PROJECTION_RULES",
    "ACTION_SCHEMA",
    "OBSERVER_SCHEMA",
    "TIMELINE_ANNOTATION_RULES",
]
