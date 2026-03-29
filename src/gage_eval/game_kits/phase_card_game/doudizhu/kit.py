from __future__ import annotations

from gage_eval.game_kits.contracts import EnvSpec, GameKit
from gage_eval.game_kits.phase_card_game.doudizhu.envs.classic_3p import (
    build_classic_3p_environment,
)
from gage_eval.game_kits.phase_card_game.doudizhu.envs.classic_3p_real import (
    build_classic_3p_real_environment,
)
from gage_eval.game_kits.phase_card_game.doudizhu.visualization import (
    VISUALIZATION_SPEC_ID,
)
from gage_eval.registry import registry


@registry.asset(
    "game_kits",
    "doudizhu",
    desc="GameArena 斗地主 phase-card kit",
    tags=("gamekit", "phase_card_game", "doudizhu"),
)
def build_doudizhu_game_kit() -> GameKit:
    return GameKit(
        kit_id="doudizhu",
        family="phase_card_game",
        scheduler_binding="turn/default",
        observation_workflow="noop_observation_v1",
        visualization_spec=VISUALIZATION_SPEC_ID,
        env_catalog=(
            EnvSpec(
                env_id="classic_3p",
                kit_id="doudizhu",
                resource_spec={"env_id": "classic_3p", "family": "doudizhu"},
                defaults={
                    "env_factory": build_classic_3p_environment,
                    "replay_filename": "doudizhu_classic_3p_replay.json",
                },
            ),
            EnvSpec(
                env_id="classic_3p_real",
                kit_id="doudizhu",
                resource_spec={"env_id": "classic_3p_real", "family": "doudizhu"},
                defaults={
                    "env_factory": build_classic_3p_real_environment,
                    "replay_filename": "doudizhu_classic_3p_real_replay.json",
                },
            ),
        ),
        default_env="classic_3p",
        seat_spec={"seats": ("landlord", "farmer_left", "farmer_right")},
        defaults={"replay_filename": "doudizhu_classic_3p_replay.json"},
    )
