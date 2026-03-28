"""Core game-kit contracts."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class EnvSpec:
    """Concrete runtime environment bound to a game kit."""

    env_id: str
    kit_id: str
    resource_spec: object = field(default_factory=dict)
    scheduler_binding: str | None = None
    observation_workflow: str | None = None
    game_content_refs: dict[str, str] = field(default_factory=dict)
    defaults: dict[str, object] = field(default_factory=dict)


@dataclass(frozen=True)
class GameKit:
    """Game-family level runtime contract."""

    kit_id: str
    family: str
    scheduler_binding: str
    observation_workflow: str
    env_catalog: tuple[EnvSpec, ...] = ()
    default_env: str | None = None
    seat_spec: dict[str, object] = field(default_factory=dict)
    resource_spec: object = field(default_factory=dict)
    support_workflow: str | None = None
    visualization_spec: str | None = None
    player_driver: str | None = None
    content_asset: str | None = None
    defaults: dict[str, object] = field(default_factory=dict)


@dataclass(frozen=True)
class GameVisualizationSpec:
    """Resolved visualization contract for a game kit."""

    spec_id: str
    plugin_id: str
    visual_kind: str
    renderer_impl: str | None = None
    scene_projection_rules: dict[str, object] = field(default_factory=dict)
    action_schema: dict[str, object] = field(default_factory=dict)
    observer_schema: dict[str, object] = field(default_factory=dict)
    timeline_annotation_rules: dict[str, object] = field(default_factory=dict)


@dataclass(frozen=True)
class ResolvedRuntimeBinding:
    """Resolved runtime resources for a concrete sample."""

    game_kit: GameKit
    env_spec: EnvSpec
    scheduler: object
    resource_spec: object
    visualization_spec: GameVisualizationSpec | None = None
    players: tuple[dict[str, object], ...] = ()
    player_bindings: tuple[object, ...] = ()
    player_driver_registry: Any | None = None
    observation_workflow: object = None
    support_workflow: object = None
