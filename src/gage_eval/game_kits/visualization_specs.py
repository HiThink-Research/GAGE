"""Visualization spec contracts and registry helpers."""

from __future__ import annotations

from typing import Any

import gage_eval.role.arena.support.specs as _support_visualization_specs  # noqa: F401
from gage_eval.game_kits.contracts import GameVisualizationSpec
from gage_eval.registry import registry


def _is_spec_like(asset: Any, *, attrs: tuple[str, ...]) -> bool:
    return all(hasattr(asset, attr) for attr in attrs)


def build_placeholder_descriptor(
    *,
    spec_id: str,
    plugin_id: str,
    visual_kind: str,
    kit_id: str,
    channel: str,
) -> dict[str, object]:
    return {
        "impl": f"placeholder://arena/visualization/{kit_id}/{channel}",
        "spec_id": spec_id,
        "plugin_id": plugin_id,
        "visual_kind": visual_kind,
        "kit_id": kit_id,
        "channel": channel,
    }


def build_placeholder_visualization_spec(
    *,
    spec_id: str,
    plugin_id: str,
    visual_kind: str,
    kit_id: str,
) -> GameVisualizationSpec:
    """Create a stable placeholder visualization spec for a game kit."""

    renderer_impl = f"placeholder://arena/visualization/{kit_id}/renderer"
    return GameVisualizationSpec(
        spec_id=spec_id,
        plugin_id=plugin_id,
        visual_kind=visual_kind,
        renderer_impl=renderer_impl,
        scene_projection_rules=build_placeholder_descriptor(
            spec_id=spec_id,
            plugin_id=plugin_id,
            visual_kind=visual_kind,
            kit_id=kit_id,
            channel="scene_projection",
        ),
        action_schema=build_placeholder_descriptor(
            spec_id=spec_id,
            plugin_id=plugin_id,
            visual_kind=visual_kind,
            kit_id=kit_id,
            channel="action_schema",
        ),
        observer_schema=build_placeholder_descriptor(
            spec_id=spec_id,
            plugin_id=plugin_id,
            visual_kind=visual_kind,
            kit_id=kit_id,
            channel="observer_schema",
        ),
        timeline_annotation_rules=build_placeholder_descriptor(
            spec_id=spec_id,
            plugin_id=plugin_id,
            visual_kind=visual_kind,
            kit_id=kit_id,
            channel="timeline_annotations",
        ),
    )


def register_visualization_spec(
    *,
    spec_id: str,
    plugin_id: str,
    visual_kind: str,
    kit_id: str,
    desc: str,
) -> GameVisualizationSpec:
    """Register a placeholder visualization spec in the registry."""

    spec = build_placeholder_visualization_spec(
        spec_id=spec_id,
        plugin_id=plugin_id,
        visual_kind=visual_kind,
        kit_id=kit_id,
    )
    registry.register("visualization_specs", spec_id, spec, desc=desc)
    return spec


def _materialize_legacy_visualization_spec(
    asset: Any,
    *,
    spec_id: str,
) -> GameVisualizationSpec:
    renderer_impl = str(getattr(asset, "renderer_impl", "")).strip() or None
    defaults = getattr(asset, "defaults", {}) or {}
    visual_kind = str(defaults.get("visual_kind") or "frame")
    plugin_id = f"arena.visualization.{str(spec_id).replace('/', '.')}"
    return GameVisualizationSpec(
        spec_id=str(getattr(asset, "spec_id")),
        plugin_id=plugin_id,
        visual_kind=visual_kind,
        renderer_impl=renderer_impl,
        scene_projection_rules={
            "impl": renderer_impl or f"placeholder://arena/visualization/{spec_id}/scene_projection",
            "spec_id": str(getattr(asset, "spec_id")),
            "source": "legacy_visualization_spec",
        },
        action_schema={
            "impl": f"placeholder://arena/visualization/{spec_id}/action_schema",
            "spec_id": str(getattr(asset, "spec_id")),
            "source": "legacy_visualization_spec",
        },
        observer_schema={
            "impl": f"placeholder://arena/visualization/{spec_id}/observer_schema",
            "spec_id": str(getattr(asset, "spec_id")),
            "source": "legacy_visualization_spec",
        },
        timeline_annotation_rules={
            "impl": f"placeholder://arena/visualization/{spec_id}/timeline_annotations",
            "spec_id": str(getattr(asset, "spec_id")),
            "source": "legacy_visualization_spec",
        },
    )


def _materialize_visualization_spec(
    asset: Any,
    *,
    spec_id: str,
) -> GameVisualizationSpec:
    if isinstance(asset, GameVisualizationSpec):
        return asset

    if callable(asset):
        spec = asset()
        if isinstance(spec, GameVisualizationSpec):
            return spec
        return _materialize_visualization_spec(spec, spec_id=spec_id)

    if _is_spec_like(
        asset,
        attrs=(
            "spec_id",
            "plugin_id",
            "visual_kind",
            "renderer_impl",
            "scene_projection_rules",
            "action_schema",
            "observer_schema",
            "timeline_annotation_rules",
        ),
    ):
        return GameVisualizationSpec(
            spec_id=str(getattr(asset, "spec_id")),
            plugin_id=str(getattr(asset, "plugin_id")),
            visual_kind=str(getattr(asset, "visual_kind")),
            renderer_impl=getattr(asset, "renderer_impl"),
            scene_projection_rules=dict(getattr(asset, "scene_projection_rules") or {}),
            action_schema=dict(getattr(asset, "action_schema") or {}),
            observer_schema=dict(getattr(asset, "observer_schema") or {}),
            timeline_annotation_rules=dict(
                getattr(asset, "timeline_annotation_rules") or {}
            ),
        )

    if _is_spec_like(asset, attrs=("spec_id", "renderer_impl", "defaults")):
        return _materialize_legacy_visualization_spec(asset, spec_id=spec_id)

    raise TypeError(
        f"Visualization spec '{spec_id}' must be a GameVisualizationSpec or callable "
        f"returning one (got '{type(asset).__name__}')"
    )


class VisualizationSpecRegistry:
    """Build registered visualization specs into runtime contracts."""

    def __init__(self, *, registry_view=None) -> None:
        self._registry = registry_view or registry
        self.registry_view = self._registry

    def build(self, spec_id: str) -> GameVisualizationSpec:
        asset = self._registry.get("visualization_specs", spec_id)
        return _materialize_visualization_spec(asset, spec_id=spec_id)


__all__ = [
    "GameVisualizationSpec",
    "VisualizationSpecRegistry",
    "build_placeholder_visualization_spec",
    "build_placeholder_descriptor",
    "register_visualization_spec",
]
