"""Core game-kit contracts."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal


RealtimeInputSemantics = Literal["continuous_state", "queued_command"]


@dataclass(frozen=True)
class EnvSpec:
    """Concrete runtime environment bound to a game kit."""

    env_id: str
    kit_id: str
    resource_spec: object = field(default_factory=dict)
    scheduler_binding: str | None = None
    observation_workflow: str | None = None
    game_content_refs: dict[str, str] = field(default_factory=dict)
    runtime_binding_policy: str | None = None
    game_display: str | None = None
    replay_viewer: str | None = None
    parser: str | None = None
    renderer: str | None = None
    replay_policy: str | None = None
    input_mapper: str | None = None
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
    game_content_refs: dict[str, str] = field(default_factory=dict)
    runtime_binding_policy: str | None = None
    game_display: str | None = None
    replay_viewer: str | None = None
    parser: str | None = None
    renderer: str | None = None
    replay_policy: str | None = None
    input_mapper: str | None = None
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
class HumanRealtimeInputProfile:
    """Resolved realtime input semantics for one bound human player."""

    player_id: str
    semantics: RealtimeInputSemantics
    tick_interval_ms: int | None = None
    timeout_ms: int | None = None
    timeout_fallback_move: str | None = None


@dataclass(frozen=True)
class RealtimeHumanControlProfile:
    """Explicit scheduler-owned realtime control policy for a session."""

    mode: str
    activation_scope: str
    input_model: RealtimeInputSemantics
    tick_interval_ms: int
    input_transport: str | None = None
    frame_output_hz: int | None = None
    artifact_sampling_mode: str | None = None
    fallback_move: str | None = None
    max_commands_per_tick: int | None = None
    max_command_queue_size: int | None = None
    command_stale_after_ms: int | None = None
    queue_overflow_policy: str | None = None
    bridge_stall_timeout_ms: int | None = None
    bridge_abort_timeout_ms: int | None = None


@dataclass(frozen=True)
class ResolvedRuntimeProfile:
    """Resolved runtime profile derived from scheduler, players, and env defaults."""

    scheduler_binding: str
    scheduler_family: str
    tick_interval_ms: int | None = None
    pure_human_realtime: bool = False
    scheduler_owns_realtime_clock: bool = False
    supports_low_latency_realtime_input: bool = False
    supports_realtime_input_websocket: bool = False
    human_realtime_inputs: tuple[HumanRealtimeInputProfile, ...] = ()
    realtime_human_control: RealtimeHumanControlProfile | None = None

    def input_profile_for(self, player_id: str) -> HumanRealtimeInputProfile | None:
        normalized_player_id = str(player_id)
        for profile in self.human_realtime_inputs:
            if profile.player_id == normalized_player_id:
                return profile
        return None

    def uses_scheduler_owned_human_realtime(self) -> bool:
        profile = self.realtime_human_control
        if profile is None:
            return self.scheduler_owns_realtime_clock and self.pure_human_realtime
        return (
            profile.mode == "scheduler_owned_human_realtime"
            and profile.activation_scope == "pure_human_only"
            and self.scheduler_owns_realtime_clock
        )


@dataclass(frozen=True)
class ResolvedRuntimeBinding:
    """Resolved runtime resources for a concrete sample."""

    game_kit: GameKit
    env_spec: EnvSpec
    scheduler: object
    resource_spec: object
    runtime_binding_policy: str | None = None
    game_display: str | None = None
    replay_viewer: str | None = None
    parser: str | None = None
    renderer: str | None = None
    replay_policy: str | None = None
    input_mapper: str | None = None
    game_content_refs: dict[str, str] = field(default_factory=dict)
    visualization_spec: GameVisualizationSpec | None = None
    players: tuple[dict[str, object], ...] = ()
    player_bindings: tuple[object, ...] = ()
    player_driver_registry: Any | None = None
    observation_workflow: object = None
    support_workflow: object = None
    runtime_profile: ResolvedRuntimeProfile | None = None
