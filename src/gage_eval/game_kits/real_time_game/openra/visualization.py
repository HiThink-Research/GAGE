"""OpenRA RTS visualization spec."""

from __future__ import annotations

from gage_eval.game_kits.contracts import GameVisualizationSpec
from gage_eval.game_kits.visualization_specs import build_placeholder_descriptor
from gage_eval.registry import registry

VISUALIZATION_SPEC_ID = "arena/visualization/openra_rts_v1"
VISUALIZATION_PLUGIN_ID = "arena.visualization.openra.rts_v1"
VISUAL_KIND = "rts"
RENDERER_IMPL = "placeholder://arena/visualization/openra/renderer"

SCENE_PROJECTION_RULES = {
    "impl": "builtin://arena/visualization/openra_rts_scene_projection_v1",
    "spec_id": VISUALIZATION_SPEC_ID,
    "plugin_id": VISUALIZATION_PLUGIN_ID,
    "visual_kind": VISUAL_KIND,
    "kit_id": "openra",
    "frame_title": "OpenRA RTS",
    "default_stream_id": "main",
    "default_fit": "contain",
}
ACTION_SCHEMA = {
    **build_placeholder_descriptor(
        spec_id=VISUALIZATION_SPEC_ID,
        plugin_id=VISUALIZATION_PLUGIN_ID,
        visual_kind=VISUAL_KIND,
        kit_id="openra",
        channel="action_schema",
    ),
    "action_metadata": {
        "descriptor": "openra_rts_actions_v1",
        "legal_action_source": "scene.legalActions",
        "selection_source": "scene.selection.units",
        "typed_actions": [
            "select_units",
            "issue_command",
            "queue_production",
            "camera_pan",
        ],
    },
}
OBSERVER_SCHEMA = {
    **build_placeholder_descriptor(
        spec_id=VISUALIZATION_SPEC_ID,
        plugin_id=VISUALIZATION_PLUGIN_ID,
        visual_kind=VISUAL_KIND,
        kit_id="openra",
        channel="observer_schema",
    ),
    "supported_modes": ["player", "spectator", "camera"],
}
TIMELINE_ANNOTATION_RULES = build_placeholder_descriptor(
    spec_id=VISUALIZATION_SPEC_ID,
    plugin_id=VISUALIZATION_PLUGIN_ID,
    visual_kind=VISUAL_KIND,
    kit_id="openra",
    channel="timeline_annotations",
)

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
    desc="OpenRA RTS visualization spec",
)

__all__ = [
    "ACTION_SCHEMA",
    "OBSERVER_SCHEMA",
    "RENDERER_IMPL",
    "SCENE_PROJECTION_RULES",
    "TIMELINE_ANNOTATION_RULES",
    "VISUALIZATION_PLUGIN_ID",
    "VISUALIZATION_SPEC",
    "VISUALIZATION_SPEC_ID",
    "VISUAL_KIND",
]
