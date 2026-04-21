"""OpenRA environment catalog."""

from gage_eval.game_kits.real_time_game.openra.envs.cnc_mission_gdi01 import (
    build_cnc_mission_gdi01_environment,
)
from gage_eval.game_kits.real_time_game.openra.envs.d2k_skirmish_1v1 import (
    build_d2k_skirmish_1v1_environment,
)
from gage_eval.game_kits.real_time_game.openra.envs.ra_map01 import (
    build_ra_map01_environment,
)
from gage_eval.game_kits.real_time_game.openra.envs.ra_skirmish_1v1 import (
    build_ra_skirmish_1v1_environment,
)

__all__ = [
    "build_cnc_mission_gdi01_environment",
    "build_d2k_skirmish_1v1_environment",
    "build_ra_map01_environment",
    "build_ra_skirmish_1v1_environment",
]
