"""Game-kit registry helpers."""

from __future__ import annotations

from dataclasses import replace
from typing import Any

from gage_eval.game_kits.aec_env_game.pettingzoo.kit import (
    build_pettingzoo_game_kit,
)
from gage_eval.game_kits.aec_env_game.pettingzoo.visualization import (
    VISUALIZATION_SPEC as PETTINGZOO_VISUALIZATION_SPEC,
    VISUALIZATION_SPEC_ID as PETTINGZOO_VISUALIZATION_SPEC_ID,
)
from gage_eval.game_kits.content import DEFAULT_GAME_CONTENT_ASSET, DEFAULT_GAME_KIT
from gage_eval.game_kits.gymnasium_atari.kit import build_gymnasium_atari_game_kit
from gage_eval.game_kits.board_game.gomoku.kit import build_gomoku_game_kit
from gage_eval.game_kits.board_game.gomoku.visualization import (
    VISUALIZATION_SPEC as GOMOKU_VISUALIZATION_SPEC,
    VISUALIZATION_SPEC_ID as GOMOKU_VISUALIZATION_SPEC_ID,
)
from gage_eval.game_kits.board_game.tictactoe.kit import build_tictactoe_game_kit
from gage_eval.game_kits.board_game.tictactoe.visualization import (
    VISUALIZATION_SPEC as TICTACTOE_VISUALIZATION_SPEC,
    VISUALIZATION_SPEC_ID as TICTACTOE_VISUALIZATION_SPEC_ID,
)
from gage_eval.game_kits.contracts import EnvSpec, GameKit
from gage_eval.game_kits.observation import ObservationWorkflowRegistry
from gage_eval.game_kits.visualization_specs import VisualizationSpecRegistry
from gage_eval.game_kits.phase_card_game.doudizhu.kit import (
    build_doudizhu_game_kit,
)
from gage_eval.game_kits.phase_card_game.doudizhu.visualization import (
    VISUALIZATION_SPEC as DOUDIZHU_VISUALIZATION_SPEC,
    VISUALIZATION_SPEC_ID as DOUDIZHU_VISUALIZATION_SPEC_ID,
)
from gage_eval.game_kits.phase_card_game.mahjong.kit import (
    build_mahjong_game_kit,
)
from gage_eval.game_kits.phase_card_game.mahjong.visualization import (
    VISUALIZATION_SPEC as MAHJONG_VISUALIZATION_SPEC,
    VISUALIZATION_SPEC_ID as MAHJONG_VISUALIZATION_SPEC_ID,
)
from gage_eval.game_kits.real_time_game.retro_platformer.kit import (
    build_retro_platformer_game_kit,
)
from gage_eval.game_kits.real_time_game.retro_platformer.visualization import (
    VISUALIZATION_SPEC as RETRO_PLATFORMER_VISUALIZATION_SPEC,
    VISUALIZATION_SPEC_ID as RETRO_PLATFORMER_VISUALIZATION_SPEC_ID,
)
from gage_eval.game_kits.real_time_game.vizdoom.kit import (
    build_vizdoom_game_kit,
)
from gage_eval.game_kits.real_time_game.vizdoom.visualization import (
    VISUALIZATION_SPEC as VIZDOOM_VISUALIZATION_SPEC,
    VISUALIZATION_SPEC_ID as VIZDOOM_VISUALIZATION_SPEC_ID,
)
from gage_eval.registry import import_asset_from_manifest, registry


_RUNTIME_RENDERER_ASSETS = (
    "gomoku_board_v1",
    "tictactoe_board_v1",
)

_RUNTIME_VISUALIZATION_SPECS = (
    (
        GOMOKU_VISUALIZATION_SPEC_ID,
        GOMOKU_VISUALIZATION_SPEC,
        "Gomoku board visualization spec",
    ),
    (
        TICTACTOE_VISUALIZATION_SPEC_ID,
        TICTACTOE_VISUALIZATION_SPEC,
        "Tic-tac-toe board visualization spec",
    ),
    (
        DOUDIZHU_VISUALIZATION_SPEC_ID,
        DOUDIZHU_VISUALIZATION_SPEC,
        "斗地主 table visualization spec",
    ),
    (
        MAHJONG_VISUALIZATION_SPEC_ID,
        MAHJONG_VISUALIZATION_SPEC,
        "麻将 table visualization spec",
    ),
    (
        PETTINGZOO_VISUALIZATION_SPEC_ID,
        PETTINGZOO_VISUALIZATION_SPEC,
        "PettingZoo frame visualization spec",
    ),
    (
        VIZDOOM_VISUALIZATION_SPEC_ID,
        VIZDOOM_VISUALIZATION_SPEC,
        "ViZDoom frame visualization spec",
    ),
    (
        RETRO_PLATFORMER_VISUALIZATION_SPEC_ID,
        RETRO_PLATFORMER_VISUALIZATION_SPEC,
        "Retro platformer frame visualization spec",
    ),
)


def _coerce_optional_text(value: object) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _resolve_visualization_renderer_ref(
    *,
    registry_view,
    spec_id: str | None,
) -> str | None:
    resolved_spec_id = _coerce_optional_text(spec_id)
    if resolved_spec_id is None:
        return None
    try:
        spec = VisualizationSpecRegistry(registry_view=registry_view).build(resolved_spec_id)
    except (KeyError, TypeError):
        return None
    return _coerce_optional_text(spec.renderer_impl)


def _normalize_env_spec(env_spec: EnvSpec, *, registry_view) -> EnvSpec:
    renderer = _coerce_optional_text(env_spec.renderer) or _resolve_visualization_renderer_ref(
        registry_view=registry_view,
        spec_id=env_spec.game_display,
    )
    replay_viewer = _coerce_optional_text(env_spec.replay_viewer) or _coerce_optional_text(
        env_spec.game_display
    )
    return replace(
        env_spec,
        renderer=renderer,
        replay_viewer=replay_viewer,
        game_content_refs=dict(env_spec.game_content_refs or {}),
    )


def _normalize_game_kit(game_kit: GameKit, *, registry_view) -> GameKit:
    game_display = _coerce_optional_text(game_kit.game_display) or _coerce_optional_text(
        game_kit.visualization_spec
    )
    replay_viewer = _coerce_optional_text(game_kit.replay_viewer) or game_display
    visualization_spec = _coerce_optional_text(game_kit.visualization_spec) or game_display
    renderer = _coerce_optional_text(game_kit.renderer) or _resolve_visualization_renderer_ref(
        registry_view=registry_view,
        spec_id=game_display,
    )
    game_content_refs = dict(game_kit.game_content_refs or {})
    if game_kit.content_asset and "content_asset" not in game_content_refs:
        game_content_refs["content_asset"] = str(game_kit.content_asset)
    env_catalog = tuple(
        _normalize_env_spec(env_spec, registry_view=registry_view)
        for env_spec in game_kit.env_catalog
    )
    return replace(
        game_kit,
        env_catalog=env_catalog,
        visualization_spec=visualization_spec,
        game_content_refs=game_content_refs,
        game_display=game_display,
        replay_viewer=replay_viewer,
        renderer=renderer,
    )


def _materialize_game_kit(asset: Any, *, kit_id: str, registry_view) -> GameKit:
    if isinstance(asset, GameKit):
        return _normalize_game_kit(asset, registry_view=registry_view)

    if callable(asset):
        kit = asset()
        if isinstance(kit, GameKit):
            return _normalize_game_kit(kit, registry_view=registry_view)
        return _materialize_game_kit(kit, kit_id=kit_id, registry_view=registry_view)

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
            game_display=asset.visualization_spec,
            replay_viewer=asset.visualization_spec,
            defaults=dict(asset.defaults),
        )
        return _normalize_game_kit(
            GameKit(
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
            game_content_refs={"content_asset": asset.content_asset},
            game_display=asset.visualization_spec,
            replay_viewer=asset.visualization_spec,
            defaults=dict(asset.defaults),
            ),
            registry_view=registry_view,
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
        return _materialize_game_kit(asset, kit_id=kit_id, registry_view=self._registry)

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
        "gymnasium_atari",
        build_gymnasium_atari_game_kit,
        desc="GameArena Gymnasium Atari kit",
        tags=("gamekit", "gymnasium", "atari"),
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
    for asset_name in _RUNTIME_RENDERER_ASSETS:
        import_asset_from_manifest(
            "renderer_impls",
            asset_name,
            registry=target,
            source="gage_eval.game_kits.registry",
        )
    for spec_id, spec, desc in _RUNTIME_VISUALIZATION_SPECS:
        target.register(
            "visualization_specs",
            spec_id,
            spec,
            desc=desc,
        )


register_runtime_assets(registry_target=registry)


__all__ = ["GameKitRegistry", "ObservationWorkflowRegistry", "register_runtime_assets"]
