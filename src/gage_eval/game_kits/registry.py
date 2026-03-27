"""Game-kit registry helpers."""

from __future__ import annotations

from typing import Any

from gage_eval.game_kits.aec_env_game.pettingzoo.kit import (
    build_pettingzoo_game_kit,
)
from gage_eval.game_kits.content import DEFAULT_GAME_CONTENT_ASSET, DEFAULT_GAME_KIT
from gage_eval.game_kits.board_game.gomoku.kit import build_gomoku_game_kit
from gage_eval.game_kits.board_game.tictactoe.kit import build_tictactoe_game_kit
from gage_eval.game_kits.contracts import EnvSpec, GameKit
from gage_eval.game_kits.observation import ObservationWorkflowRegistry
from gage_eval.game_kits.phase_card_game.doudizhu.kit import (
    build_doudizhu_game_kit,
)
from gage_eval.game_kits.phase_card_game.mahjong.kit import (
    build_mahjong_game_kit,
)
from gage_eval.game_kits.real_time_game.retro_platformer.kit import (
    build_retro_platformer_game_kit,
)
from gage_eval.game_kits.real_time_game.vizdoom.kit import (
    build_vizdoom_game_kit,
)
from gage_eval.registry import registry


def _materialize_game_kit(asset: Any, *, kit_id: str) -> GameKit:
    if isinstance(asset, GameKit):
        return asset

    if callable(asset):
        kit = asset()
        if isinstance(kit, GameKit):
            return kit
        return _materialize_game_kit(kit, kit_id=kit_id)

    try:
        from gage_eval.game_kits.content import GameKitSpec
    except Exception:  # pragma: no cover - defensive import fallback
        GameKitSpec = None  # type: ignore[assignment]

    if GameKitSpec is not None and isinstance(asset, GameKitSpec):
        family = kit_id.split("/", 1)[0] if "/" in kit_id else kit_id
        runtime_resource_spec = {
            "content_asset": asset.content_asset,
            **dict(asset.defaults),
        }
        default_env = EnvSpec(
            env_id=asset.kit_id,
            kit_id=asset.kit_id,
            resource_spec=runtime_resource_spec,
            scheduler_binding=asset.scheduler_binding,
            observation_workflow=asset.observation_workflow,
            game_content_refs={"content_asset": asset.content_asset},
            defaults=dict(asset.defaults),
        )
        return GameKit(
            kit_id=asset.kit_id,
            family=family,
            scheduler_binding=asset.scheduler_binding,
            observation_workflow=asset.observation_workflow,
            env_catalog=(default_env,),
            default_env=default_env.env_id,
            seat_spec={},
            resource_spec=runtime_resource_spec,
            support_workflow=asset.support_workflow,
            visualization_spec=asset.visualization_spec,
            player_driver=asset.player_driver,
            content_asset=asset.content_asset,
            defaults=dict(asset.defaults),
        )

    raise TypeError(
        f"Game kit '{kit_id}' must be a 'GameKit' or callable returning one "
        f"(got '{type(asset).__name__}')"
    )


class GameKitRegistry:
    """Resolve registry-managed game-kit assets into runtime contracts."""

    def __init__(self, *, registry_view=None) -> None:
        self._registry = registry_view or registry
        self.registry_view = self._registry

    def build(self, kit_id: str) -> GameKit:
        asset = self._registry.get("game_kits", kit_id)
        return _materialize_game_kit(asset, kit_id=kit_id)

    def get(self, kit_id: str) -> GameKit:
        return self.build(kit_id)


def register_runtime_assets(*, registry_target=None) -> None:
    target = registry_target or registry
    target.register(
        "game_content_assets",
        DEFAULT_GAME_CONTENT_ASSET.asset_id,
        DEFAULT_GAME_CONTENT_ASSET,
        desc="Default content asset spec for GameArena",
    )
    target.register(
        "game_kits",
        DEFAULT_GAME_KIT.kit_id,
        DEFAULT_GAME_KIT,
        desc="Default game kit spec for GameArena",
    )
    target.register(
        "game_kits",
        "gomoku",
        build_gomoku_game_kit,
        desc="GameArena Gomoku kit",
        tags=("gamekit", "board_game", "gomoku"),
    )
    target.register(
        "game_kits",
        "tictactoe",
        build_tictactoe_game_kit,
        desc="GameArena Tic-Tac-Toe kit",
        tags=("gamekit", "board_game", "tictactoe"),
    )
    target.register(
        "game_kits",
        "pettingzoo",
        build_pettingzoo_game_kit,
        desc="GameArena PettingZoo kit",
        tags=("gamekit", "aec_env_game", "pettingzoo"),
    )
    target.register(
        "game_kits",
        "doudizhu",
        build_doudizhu_game_kit,
        desc="GameArena 斗地主 phase-card kit",
        tags=("gamekit", "phase_card_game", "doudizhu"),
    )
    target.register(
        "game_kits",
        "mahjong",
        build_mahjong_game_kit,
        desc="GameArena 麻将 phase-card kit",
        tags=("gamekit", "phase_card_game", "mahjong"),
    )
    target.register(
        "game_kits",
        "vizdoom",
        build_vizdoom_game_kit,
        desc="GameArena ViZDoom realtime kit",
        tags=("gamekit", "real_time_game", "vizdoom"),
    )
    target.register(
        "game_kits",
        "retro_platformer",
        build_retro_platformer_game_kit,
        desc="GameArena retro platformer realtime kit",
        tags=("gamekit", "real_time_game", "retro"),
    )


register_runtime_assets(registry_target=registry)


__all__ = ["GameKitRegistry", "ObservationWorkflowRegistry", "register_runtime_assets"]
