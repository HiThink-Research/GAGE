from __future__ import annotations

from gage_eval.game_kits.contracts import EnvSpec, GameKit
from gage_eval.game_kits.real_time_game.openra.envs import (
    build_cnc_mission_gdi01_environment,
    build_d2k_skirmish_1v1_environment,
    build_ra_map01_environment,
    build_ra_skirmish_1v1_environment,
)
from gage_eval.game_kits.real_time_game.openra.environment import OpenRAArenaEnvironment
from gage_eval.game_kits.real_time_game.openra.visualization import (
    VISUALIZATION_SPEC_ID,
)
from gage_eval.registry import registry

PARSER_PATH = "gage_eval.game_kits.real_time_game.openra.parser.OpenRAActionParser"
INPUT_MAPPER_PATH = "gage_eval.game_kits.real_time_game.openra.input_mapper.OpenRAInputMapper"
ENGINE_CONTENT_REFS = {"engine": "engine/openra-bleed"}
COMMON_ENV_DEFAULTS = {
    "backend_mode": "dummy",
    "stub_max_ticks": 6,
    "stream_id": "main",
    "viewport": {"width": 1280, "height": 720},
    "parser": PARSER_PATH,
    "input_mapper": INPUT_MAPPER_PATH,
}


def _build_env_spec(
    *,
    env_id: str,
    env_factory,
    mod_id: str,
    map_ref: str,
    script_ref: str | None = None,
    resource_spec: dict[str, str] | None = None,
) -> EnvSpec:
    game_content_refs = {
        "mod": f"mod/openra/{mod_id}",
        "map": map_ref,
    }
    if script_ref is not None:
        game_content_refs["script"] = script_ref
    return EnvSpec(
        env_id=env_id,
        kit_id="openra",
        resource_spec=resource_spec or {"env_id": env_id, "family": "openra", "mod": mod_id},
        game_content_refs=game_content_refs,
        parser=PARSER_PATH,
        input_mapper=INPUT_MAPPER_PATH,
        defaults={
            **COMMON_ENV_DEFAULTS,
            "env_factory": env_factory,
            "env_id": env_id,
            "mod_id": mod_id,
            "map_ref": map_ref,
        },
    )


@registry.asset(
    "game_kits",
    "openra",
    desc="GameArena OpenRA realtime kit",
    tags=("gamekit", "real_time_game", "openra", "rts"),
)
def build_openra_game_kit() -> GameKit:
    return GameKit(
        kit_id="openra",
        family="real_time_game",
        scheduler_binding="real_time_tick/default",
        observation_workflow="noop_observation_v1",
        visualization_spec=VISUALIZATION_SPEC_ID,
        parser=PARSER_PATH,
        input_mapper=INPUT_MAPPER_PATH,
        game_content_refs=dict(ENGINE_CONTENT_REFS),
        env_catalog=(
            _build_env_spec(
                env_id="ra_map01",
                env_factory=build_ra_map01_environment,
                mod_id="ra",
                map_ref="map/openra/ra_map01",
                resource_spec={"env_id": "ra_map01", "family": "openra"},
            ),
            _build_env_spec(
                env_id="ra_skirmish_1v1",
                env_factory=build_ra_skirmish_1v1_environment,
                mod_id="ra",
                map_ref="map/openra/ra/marigold-town.oramap",
            ),
            _build_env_spec(
                env_id="cnc_mission_gdi01",
                env_factory=build_cnc_mission_gdi01_environment,
                mod_id="cnc",
                map_ref="map/openra/cnc/gdi01",
                script_ref="script/openra/gdi01.lua",
            ),
            _build_env_spec(
                env_id="d2k_skirmish_1v1",
                env_factory=build_d2k_skirmish_1v1_environment,
                mod_id="d2k",
                map_ref="map/openra/d2k/chin-rock.oramap",
            ),
        ),
        default_env="ra_map01",
        seat_spec={"seats": ("player_0", "player_1")},
        defaults=dict(COMMON_ENV_DEFAULTS),
    )


__all__ = ["OpenRAArenaEnvironment", "build_openra_game_kit"]
