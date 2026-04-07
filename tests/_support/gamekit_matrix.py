from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import yaml


REPO_ROOT = Path(__file__).resolve().parents[2]

HEADLESS_NO_HUMAN = "headless_no_human"
VISUAL_NO_HUMAN = "visual_no_human"
HUMAN_VISUAL = "human_visual"


@dataclass(frozen=True)
class GameKitConfigCase:
    relpath: str
    category: str
    game_kit: str
    env: str
    plugin_id: str | None = None
    live_scene_scheme: str | None = None

    @property
    def path(self) -> Path:
        return REPO_ROOT / self.relpath


PLUGIN_IDS_BY_GAME_KIT = {
    "gomoku": "arena.visualization.gomoku.board_v1",
    "tictactoe": "arena.visualization.tictactoe.board_v1",
    "doudizhu": "arena.visualization.doudizhu.table_v1",
    "mahjong": "arena.visualization.mahjong.table_v1",
    "pettingzoo": "arena.visualization.pettingzoo.frame_v1",
    "retro_platformer": "arena.visualization.retro_platformer.frame_v1",
    "vizdoom": "arena.visualization.vizdoom.frame_v1",
}


LIVE_GAMEKIT_CONFIG_CASES: tuple[GameKitConfigCase, ...] = (
    GameKitConfigCase(
        "config/custom/doudizhu/doudizhu_dummy_gamekit.yaml",
        HEADLESS_NO_HUMAN,
        "doudizhu",
        "classic_3p",
    ),
    GameKitConfigCase(
        "config/custom/doudizhu/doudizhu_llm_headless_gamekit.yaml",
        HEADLESS_NO_HUMAN,
        "doudizhu",
        "classic_3p",
    ),
    GameKitConfigCase(
        "config/custom/gomoku/gomoku_dummy_gamekit.yaml",
        HEADLESS_NO_HUMAN,
        "gomoku",
        "gomoku_standard",
    ),
    GameKitConfigCase(
        "config/custom/gomoku/gomoku_llm_headless_gamekit.yaml",
        HEADLESS_NO_HUMAN,
        "gomoku",
        "gomoku_standard",
    ),
    GameKitConfigCase(
        "config/custom/mahjong/mahjong_dummy_gamekit.yaml",
        HEADLESS_NO_HUMAN,
        "mahjong",
        "riichi_4p",
    ),
    GameKitConfigCase(
        "config/custom/mahjong/mahjong_llm_headless_gamekit.yaml",
        HEADLESS_NO_HUMAN,
        "mahjong",
        "riichi_4p",
    ),
    GameKitConfigCase(
        "config/custom/pettingzoo/space_invaders_dummy_gamekit.yaml",
        HEADLESS_NO_HUMAN,
        "pettingzoo",
        "space_invaders",
    ),
    GameKitConfigCase(
        "config/custom/pettingzoo/space_invaders_llm_headless_gamekit.yaml",
        HEADLESS_NO_HUMAN,
        "pettingzoo",
        "space_invaders",
    ),
    GameKitConfigCase(
        "config/custom/retro_mario/retro_mario_dummy_gamekit.yaml",
        HEADLESS_NO_HUMAN,
        "retro_platformer",
        "retro_mario",
    ),
    GameKitConfigCase(
        "config/custom/retro_mario/retro_mario_llm_headless_gamekit.yaml",
        HEADLESS_NO_HUMAN,
        "retro_platformer",
        "retro_mario",
    ),
    GameKitConfigCase(
        "config/custom/tictactoe/tictactoe_dummy_gamekit.yaml",
        HEADLESS_NO_HUMAN,
        "tictactoe",
        "tictactoe_standard",
    ),
    GameKitConfigCase(
        "config/custom/tictactoe/tictactoe_llm_headless_gamekit.yaml",
        HEADLESS_NO_HUMAN,
        "tictactoe",
        "tictactoe_standard",
    ),
    GameKitConfigCase(
        "config/custom/vizdoom/vizdoom_dummy_gamekit.yaml",
        HEADLESS_NO_HUMAN,
        "vizdoom",
        "duel_map01",
    ),
    GameKitConfigCase(
        "config/custom/vizdoom/vizdoom_llm_headless_gamekit.yaml",
        HEADLESS_NO_HUMAN,
        "vizdoom",
        "duel_map01",
    ),
    GameKitConfigCase(
        "config/custom/doudizhu/doudizhu_dummy_visual_gamekit.yaml",
        VISUAL_NO_HUMAN,
        "doudizhu",
        "classic_3p",
        plugin_id=PLUGIN_IDS_BY_GAME_KIT["doudizhu"],
    ),
    GameKitConfigCase(
        "config/custom/doudizhu/doudizhu_llm_visual_gamekit.yaml",
        VISUAL_NO_HUMAN,
        "doudizhu",
        "classic_3p",
        plugin_id=PLUGIN_IDS_BY_GAME_KIT["doudizhu"],
    ),
    GameKitConfigCase(
        "config/custom/gomoku/gomoku_dummy_visual_gamekit.yaml",
        VISUAL_NO_HUMAN,
        "gomoku",
        "gomoku_standard",
        plugin_id=PLUGIN_IDS_BY_GAME_KIT["gomoku"],
    ),
    GameKitConfigCase(
        "config/custom/gomoku/gomoku_llm_visual_gamekit.yaml",
        VISUAL_NO_HUMAN,
        "gomoku",
        "gomoku_standard",
        plugin_id=PLUGIN_IDS_BY_GAME_KIT["gomoku"],
    ),
    GameKitConfigCase(
        "config/custom/mahjong/mahjong_dummy_visual_gamekit.yaml",
        VISUAL_NO_HUMAN,
        "mahjong",
        "riichi_4p",
        plugin_id=PLUGIN_IDS_BY_GAME_KIT["mahjong"],
    ),
    GameKitConfigCase(
        "config/custom/mahjong/mahjong_llm_visual_gamekit.yaml",
        VISUAL_NO_HUMAN,
        "mahjong",
        "riichi_4p",
        plugin_id=PLUGIN_IDS_BY_GAME_KIT["mahjong"],
    ),
    GameKitConfigCase(
        "config/custom/pettingzoo/space_invaders_double_llm_visual_gamekit.yaml",
        VISUAL_NO_HUMAN,
        "pettingzoo",
        "space_invaders",
        plugin_id=PLUGIN_IDS_BY_GAME_KIT["pettingzoo"],
        live_scene_scheme="low_latency_channel",
    ),
    GameKitConfigCase(
        "config/custom/pettingzoo/space_invaders_double_llm_visual_low_latency_channel_gamekit.yaml",
        VISUAL_NO_HUMAN,
        "pettingzoo",
        "space_invaders",
        plugin_id=PLUGIN_IDS_BY_GAME_KIT["pettingzoo"],
        live_scene_scheme="http_pull",
    ),
    GameKitConfigCase(
        "config/custom/pettingzoo/space_invaders_dummy_visual_binary_stream_gamekit.yaml",
        VISUAL_NO_HUMAN,
        "pettingzoo",
        "space_invaders",
        plugin_id=PLUGIN_IDS_BY_GAME_KIT["pettingzoo"],
        live_scene_scheme="binary_stream",
    ),
    GameKitConfigCase(
        "config/custom/pettingzoo/space_invaders_dummy_visual_gamekit.yaml",
        VISUAL_NO_HUMAN,
        "pettingzoo",
        "space_invaders",
        plugin_id=PLUGIN_IDS_BY_GAME_KIT["pettingzoo"],
        live_scene_scheme="http_pull",
    ),
    GameKitConfigCase(
        "config/custom/pettingzoo/space_invaders_dummy_visual_low_latency_channel_gamekit.yaml",
        VISUAL_NO_HUMAN,
        "pettingzoo",
        "space_invaders",
        plugin_id=PLUGIN_IDS_BY_GAME_KIT["pettingzoo"],
        live_scene_scheme="low_latency_channel",
    ),
    GameKitConfigCase(
        "config/custom/pettingzoo/space_invaders_llm_visual_gamekit.yaml",
        VISUAL_NO_HUMAN,
        "pettingzoo",
        "space_invaders",
        plugin_id=PLUGIN_IDS_BY_GAME_KIT["pettingzoo"],
        live_scene_scheme="http_pull",
    ),
    GameKitConfigCase(
        "config/custom/retro_mario/retro_mario_llm_visual_gamekit.yaml",
        VISUAL_NO_HUMAN,
        "retro_platformer",
        "retro_mario",
        plugin_id=PLUGIN_IDS_BY_GAME_KIT["retro_platformer"],
        live_scene_scheme="http_pull",
    ),
    GameKitConfigCase(
        "config/custom/tictactoe/tictactoe_dummy_visual_gamekit.yaml",
        VISUAL_NO_HUMAN,
        "tictactoe",
        "tictactoe_standard",
        plugin_id=PLUGIN_IDS_BY_GAME_KIT["tictactoe"],
    ),
    GameKitConfigCase(
        "config/custom/tictactoe/tictactoe_llm_visual_gamekit.yaml",
        VISUAL_NO_HUMAN,
        "tictactoe",
        "tictactoe_standard",
        plugin_id=PLUGIN_IDS_BY_GAME_KIT["tictactoe"],
    ),
    GameKitConfigCase(
        "config/custom/vizdoom/vizdoom_llm_visual_gamekit.yaml",
        VISUAL_NO_HUMAN,
        "vizdoom",
        "duel_map01",
        plugin_id=PLUGIN_IDS_BY_GAME_KIT["vizdoom"],
        live_scene_scheme="http_pull",
    ),
    GameKitConfigCase(
        "config/custom/doudizhu/doudizhu_human_visual_acceptance_gamekit.yaml",
        HUMAN_VISUAL,
        "doudizhu",
        "classic_3p_real",
        plugin_id=PLUGIN_IDS_BY_GAME_KIT["doudizhu"],
    ),
    GameKitConfigCase(
        "config/custom/doudizhu/doudizhu_human_visual_gamekit.yaml",
        HUMAN_VISUAL,
        "doudizhu",
        "classic_3p_real",
        plugin_id=PLUGIN_IDS_BY_GAME_KIT["doudizhu"],
    ),
    GameKitConfigCase(
        "config/custom/gomoku/gomoku_human_visual_15x15_gamekit.yaml",
        HUMAN_VISUAL,
        "gomoku",
        "gomoku_standard",
        plugin_id=PLUGIN_IDS_BY_GAME_KIT["gomoku"],
    ),
    GameKitConfigCase(
        "config/custom/gomoku/gomoku_human_visual_gamekit.yaml",
        HUMAN_VISUAL,
        "gomoku",
        "gomoku_standard",
        plugin_id=PLUGIN_IDS_BY_GAME_KIT["gomoku"],
    ),
    GameKitConfigCase(
        "config/custom/mahjong/mahjong_human_visual_acceptance_gamekit.yaml",
        HUMAN_VISUAL,
        "mahjong",
        "riichi_4p_real",
        plugin_id=PLUGIN_IDS_BY_GAME_KIT["mahjong"],
    ),
    GameKitConfigCase(
        "config/custom/mahjong/mahjong_human_visual_gamekit.yaml",
        HUMAN_VISUAL,
        "mahjong",
        "riichi_4p_real",
        plugin_id=PLUGIN_IDS_BY_GAME_KIT["mahjong"],
    ),
    GameKitConfigCase(
        "config/custom/pettingzoo/space_invaders_human_visual_gamekit.yaml",
        HUMAN_VISUAL,
        "pettingzoo",
        "space_invaders",
        plugin_id=PLUGIN_IDS_BY_GAME_KIT["pettingzoo"],
        live_scene_scheme="http_pull",
    ),
    GameKitConfigCase(
        "config/custom/retro_mario/retro_mario_human_visual_gamekit.yaml",
        HUMAN_VISUAL,
        "retro_platformer",
        "retro_mario",
        plugin_id=PLUGIN_IDS_BY_GAME_KIT["retro_platformer"],
        live_scene_scheme="low_latency_channel",
    ),
    GameKitConfigCase(
        "config/custom/tictactoe/tictactoe_human_visual_gamekit.yaml",
        HUMAN_VISUAL,
        "tictactoe",
        "tictactoe_standard",
        plugin_id=PLUGIN_IDS_BY_GAME_KIT["tictactoe"],
    ),
    GameKitConfigCase(
        "config/custom/vizdoom/vizdoom_human_visual_gamekit.yaml",
        HUMAN_VISUAL,
        "vizdoom",
        "duel_map01",
        plugin_id=PLUGIN_IDS_BY_GAME_KIT["vizdoom"],
        live_scene_scheme="low_latency_channel",
    ),
)

REPLAY_ONLY_CONFIGS = (
    "config/custom/oneclick/replay_dummy/doudizhu_dummy_replay.yaml",
    "config/custom/oneclick/replay_dummy/gomoku_dummy_replay.yaml",
    "config/custom/oneclick/replay_dummy/mahjong_dummy_replay.yaml",
    "config/custom/oneclick/replay_dummy/pettingzoo_dummy_replay.yaml",
    "config/custom/oneclick/replay_dummy/tictactoe_dummy_replay.yaml",
)


LIVE_CASES_BY_CATEGORY = {
    HEADLESS_NO_HUMAN: tuple(case for case in LIVE_GAMEKIT_CONFIG_CASES if case.category == HEADLESS_NO_HUMAN),
    VISUAL_NO_HUMAN: tuple(case for case in LIVE_GAMEKIT_CONFIG_CASES if case.category == VISUAL_NO_HUMAN),
    HUMAN_VISUAL: tuple(case for case in LIVE_GAMEKIT_CONFIG_CASES if case.category == HUMAN_VISUAL),
}

LIVE_CASES_BY_RELPATH = {case.relpath: case for case in LIVE_GAMEKIT_CONFIG_CASES}


def iter_live_cases(*, category: str | None = None) -> tuple[GameKitConfigCase, ...]:
    if category is None:
        return LIVE_GAMEKIT_CONFIG_CASES
    return LIVE_CASES_BY_CATEGORY[category]


def load_config_payload(case_or_relpath: GameKitConfigCase | str) -> dict[str, object]:
    relpath = case_or_relpath.relpath if isinstance(case_or_relpath, GameKitConfigCase) else case_or_relpath
    payload = yaml.safe_load((REPO_ROOT / relpath).read_text(encoding="utf-8")) or {}
    if not isinstance(payload, dict):
        raise TypeError(f"Config {relpath} must load as a mapping")
    return payload


def load_primary_adapter_params(case_or_relpath: GameKitConfigCase | str) -> dict[str, object]:
    payload = load_config_payload(case_or_relpath)
    adapter = payload["role_adapters"][0]
    params = adapter.get("params") or {}
    if not isinstance(params, dict):
        raise TypeError(f"Primary adapter params for {case_or_relpath} must be a mapping")
    return params


def discover_live_gamekit_configs() -> tuple[str, ...]:
    return tuple(
        sorted(
            str(path.relative_to(REPO_ROOT))
            for path in (REPO_ROOT / "config/custom").rglob("*gamekit*.yaml")
        )
    )


def discover_replay_only_configs() -> tuple[str, ...]:
    return tuple(
        sorted(
            str(path.relative_to(REPO_ROOT))
            for path in (REPO_ROOT / "config/custom/oneclick/replay_dummy").glob("*.yaml")
        )
    )


def expected_live_gamekit_config_paths() -> tuple[str, ...]:
    return tuple(sorted(case.relpath for case in LIVE_GAMEKIT_CONFIG_CASES))


def human_visual_families() -> set[str]:
    return {case.game_kit for case in LIVE_CASES_BY_CATEGORY[HUMAN_VISUAL]}


def shipped_gamekit_families() -> set[str]:
    return {case.game_kit for case in LIVE_GAMEKIT_CONFIG_CASES}


def manual_human_visual_commands(python_bin: str) -> tuple[str, ...]:
    return tuple(
        f"cd {REPO_ROOT} && PYTHONPATH=src {python_bin} -m gage_eval.tools.config_checker --config {case.relpath} --materialize-runtime"
        for case in LIVE_CASES_BY_CATEGORY[HUMAN_VISUAL]
    )


def iter_case_paths(cases: Iterable[GameKitConfigCase]) -> tuple[str, ...]:
    return tuple(case.relpath for case in cases)
