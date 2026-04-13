"""Game kit and content specs for GameArena."""

from __future__ import annotations

from dataclasses import dataclass, field

from gage_eval.registry import registry


@dataclass(frozen=True)
class GameKitSpec:
    kit_id: str
    scheduler_binding: str
    support_workflow: str
    observation_workflow: str
    visualization_spec: str
    player_driver: str
    content_asset: str
    defaults: dict[str, object] = field(default_factory=dict)


@dataclass(frozen=True)
class GameContentAssetSpec:
    asset_id: str
    kind: str
    location: str
    defaults: dict[str, object] = field(default_factory=dict)


DEFAULT_GAME_CONTENT_ASSET = GameContentAssetSpec(
    asset_id="arena/default_content",
    kind="placeholder",
    location="memory://arena/default",
)
DEFAULT_GAME_KIT = GameKitSpec(
    kit_id="arena/default",
    scheduler_binding="turn/default",
    support_workflow="arena/default",
    observation_workflow="arena/default",
    visualization_spec="arena/default",
    player_driver="arena/default",
    content_asset=DEFAULT_GAME_CONTENT_ASSET.asset_id,
)


registry.register(
    "game_content_assets",
    "arena/default_content",
    DEFAULT_GAME_CONTENT_ASSET,
    desc="Default content asset spec for GameArena",
)
registry.register(
    "game_kits",
    "arena/default",
    DEFAULT_GAME_KIT,
    desc="Default game kit spec for GameArena",
)
