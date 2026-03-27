"""Support and observation specs for GameArena."""

from __future__ import annotations

from dataclasses import dataclass, field

import gage_eval.role.arena.support.registry as _support_registry  # noqa: F401
from gage_eval.registry import registry


@dataclass(frozen=True)
class SupportWorkflowSpec:
    workflow_id: str
    unit_ids: tuple[str, ...] = ()
    defaults: dict[str, object] = field(default_factory=dict)


@dataclass(frozen=True)
class SupportUnitSpec:
    unit_id: str
    impl: str
    defaults: dict[str, object] = field(default_factory=dict)


@dataclass(frozen=True)
class ObservationWorkflowSpec:
    workflow_id: str
    impl: str
    defaults: dict[str, object] = field(default_factory=dict)


@dataclass(frozen=True)
class VisualizationSpec:
    spec_id: str
    renderer_impl: str
    defaults: dict[str, object] = field(default_factory=dict)


@dataclass(frozen=True)
class PlayerDriverSpec:
    driver_id: str
    family: str
    impl: str
    defaults: dict[str, object] = field(default_factory=dict)


DEFAULT_SUPPORT_WORKFLOW = SupportWorkflowSpec(
    workflow_id="arena/default",
    unit_ids=("arena/default",),
)
DEFAULT_SUPPORT_UNIT = SupportUnitSpec(
    unit_id="arena/default",
    impl="placeholder://arena/support_units/default",
)
DEFAULT_OBSERVATION_WORKFLOW = ObservationWorkflowSpec(
    workflow_id="arena/default",
    impl="placeholder://arena/observation_workflows/default",
)
DEFAULT_VISUALIZATION_SPEC = VisualizationSpec(
    spec_id="arena/default",
    renderer_impl="placeholder://arena/visualization_specs/default",
)
DEFAULT_DUMMY_PLAYER_DRIVER = PlayerDriverSpec(
    driver_id="player_driver/dummy",
    family="dummy",
    impl="dummy",
)
DEFAULT_HUMAN_PLAYER_DRIVER = PlayerDriverSpec(
    driver_id="player_driver/human_local_input",
    family="human",
    impl="human_local_input",
)
DEFAULT_LLM_PLAYER_DRIVER = PlayerDriverSpec(
    driver_id="player_driver/llm_backend",
    family="llm",
    impl="llm_backend",
)
DEFAULT_AGENT_PLAYER_DRIVER = PlayerDriverSpec(
    driver_id="player_driver/agent_role_stub",
    family="agent",
    impl="agent_role_stub",
)
DEFAULT_PLAYER_DRIVER = DEFAULT_DUMMY_PLAYER_DRIVER


def register_runtime_assets(*, registry_target=None) -> None:
    target = registry_target or registry
    _support_registry.register_runtime_assets(registry_target=target)
    target.register(
        "support_units",
        "arena/default",
        DEFAULT_SUPPORT_UNIT,
        desc="Default support unit spec for GameArena",
    )
    target.register(
        "observation_workflows",
        "arena/default",
        DEFAULT_OBSERVATION_WORKFLOW,
        desc="Default observation workflow spec for GameArena",
    )
    target.register(
        "visualization_specs",
        "arena/default",
        DEFAULT_VISUALIZATION_SPEC,
        desc="Default visualization spec for GameArena",
    )
    target.register(
        "player_drivers",
        DEFAULT_DUMMY_PLAYER_DRIVER.driver_id,
        DEFAULT_DUMMY_PLAYER_DRIVER,
        desc="Default dummy player driver spec for GameArena",
    )
    target.register(
        "player_drivers",
        DEFAULT_HUMAN_PLAYER_DRIVER.driver_id,
        DEFAULT_HUMAN_PLAYER_DRIVER,
        desc="Default local human input driver spec for GameArena",
    )
    target.register(
        "player_drivers",
        DEFAULT_LLM_PLAYER_DRIVER.driver_id,
        DEFAULT_LLM_PLAYER_DRIVER,
        desc="Default llm backend driver spec for GameArena",
    )
    target.register(
        "player_drivers",
        DEFAULT_AGENT_PLAYER_DRIVER.driver_id,
        DEFAULT_AGENT_PLAYER_DRIVER,
        desc="Default agent stub driver spec for GameArena",
    )


register_runtime_assets(registry_target=registry)
