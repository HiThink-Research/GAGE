"""Doudizhu visualization spec."""

from __future__ import annotations

from gage_eval.game_kits.contracts import GameVisualizationSpec
from gage_eval.game_kits.visualization_specs import build_placeholder_descriptor
from gage_eval.registry import registry

VISUALIZATION_SPEC_ID = "arena/visualization/doudizhu_table_v1"
VISUALIZATION_PLUGIN_ID = "arena.visualization.doudizhu.table_v1"
VISUAL_KIND = "table"
RENDERER_IMPL = "placeholder://arena/visualization/doudizhu/renderer"

SCENE_PROJECTION_RULES = {
    "impl": "builtin://arena/visualization/table_scene_projection_v1",
    "spec_id": VISUALIZATION_SPEC_ID,
    "plugin_id": VISUALIZATION_PLUGIN_ID,
    "visual_kind": VISUAL_KIND,
    "kit_id": "doudizhu",
    "table_game": "doudizhu",
    "seat_count": 3,
    "default_layout": "three-seat",
}
ACTION_SCHEMA = {
    "descriptor": "doudizhu_table_action_schema_v1",
    "action_metadata": {
        "descriptor": "doudizhu_table_actions_v1",
        "legal_action_source": "scene.legalActions",
        "selection_source": "table.seats[].hand.cards",
        "typed_actions": ["play_cards", "pass"],
    },
    "action_types": [
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
    ],
    "selection_model": {
        "source": "hand.cards",
        "supports_multi_select": True,
        "pass_action_text": "pass",
    },
}
OBSERVER_SCHEMA = {
    **build_placeholder_descriptor(
        spec_id=VISUALIZATION_SPEC_ID,
        plugin_id=VISUALIZATION_PLUGIN_ID,
        visual_kind=VISUAL_KIND,
        kit_id="doudizhu",
        channel="observer_schema",
    ),
    "supported_modes": ["global", "spectator", "camera", "player"],
}
TIMELINE_ANNOTATION_RULES = {
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

SCENE_PROJECTION_RULES["scene_contract"] = {
    "table": {
        "seat_extensions": ["role", "playedCards", "hand.maskedCount"],
        "center_extensions": ["history"],
        "status_extensions": ["privateViewPlayerId", "landlordId"],
        "panel_extensions": ["chatLog", "events", "trace"],
    }
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
    desc="Doudizhu table visualization spec",
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
