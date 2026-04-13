"""PettingZoo visualization spec."""

from __future__ import annotations

from gage_eval.game_kits.contracts import GameVisualizationSpec
from gage_eval.game_kits.visualization_specs import build_placeholder_descriptor
from gage_eval.registry import registry

VISUALIZATION_SPEC_ID = "arena/visualization/pettingzoo_frame_v1"
VISUALIZATION_PLUGIN_ID = "arena.visualization.pettingzoo.frame_v1"
VISUAL_KIND = "frame"
RENDERER_IMPL = "placeholder://arena/visualization/pettingzoo/renderer"

SCENE_PROJECTION_RULES = {
    "impl": "builtin://arena/visualization/frame_scene_projection_v1",
    "spec_id": VISUALIZATION_SPEC_ID,
    "plugin_id": VISUALIZATION_PLUGIN_ID,
    "visual_kind": VISUAL_KIND,
    "kit_id": "pettingzoo",
    "frame_game": "pettingzoo",
    "frame_title": "PettingZoo Frame",
    "default_stream_id": "arena",
    "default_fit": "contain",
}
ACTION_SCHEMA = {
    **build_placeholder_descriptor(
        spec_id=VISUALIZATION_SPEC_ID,
        plugin_id=VISUALIZATION_PLUGIN_ID,
        visual_kind=VISUAL_KIND,
        kit_id="pettingzoo",
        channel="action_schema",
    ),
    "action_metadata": {"descriptor": "placeholder"},
}
OBSERVER_SCHEMA = {
    **build_placeholder_descriptor(
        spec_id=VISUALIZATION_SPEC_ID,
        plugin_id=VISUALIZATION_PLUGIN_ID,
        visual_kind=VISUAL_KIND,
        kit_id="pettingzoo",
        channel="observer_schema",
    ),
    "supported_modes": ["player", "global"],
}
TIMELINE_ANNOTATION_RULES = build_placeholder_descriptor(
    spec_id=VISUALIZATION_SPEC_ID,
    plugin_id=VISUALIZATION_PLUGIN_ID,
    visual_kind=VISUAL_KIND,
    kit_id="pettingzoo",
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
    desc="PettingZoo frame visualization spec",
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
