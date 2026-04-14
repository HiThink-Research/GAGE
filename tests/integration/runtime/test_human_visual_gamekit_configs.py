from __future__ import annotations

from pathlib import Path

import pytest
import yaml


REPO_ROOT = Path(__file__).resolve().parents[3]
ARENA_MANIFEST_PATH = REPO_ROOT / "src/gage_eval/registry/manifests/arena.json"
_LEGACY_DDZ_RENDERER = "_".join(("doudizhu", "".join(("show", "down")), "v1"))


def _load_config(relpath: str) -> dict:
    return yaml.safe_load((REPO_ROOT / relpath).read_text(encoding="utf-8"))


def _load_arena_manifest() -> dict:
    return yaml.safe_load(ARENA_MANIFEST_PATH.read_text(encoding="utf-8"))


@pytest.mark.parametrize(
    ("relpath", "expected_overrides"),
    [
        (
            "config/custom/gomoku/gomoku_human_visual_gamekit.yaml",
            {
                "board_size": 3,
                "win_len": 3,
                "coord_scheme": "A1",
                "obs_image": True,
            },
        ),
        (
            "config/custom/gomoku/gomoku_human_visual_15x15_gamekit.yaml",
            {
                "board_size": 15,
                "win_len": 5,
                "max_turns": 225,
                "coord_scheme": "A1",
                "obs_image": True,
            },
        ),
    ],
)
def test_gomoku_human_visual_configs_route_human_input_to_black(
    relpath: str,
    expected_overrides: dict[str, object],
) -> None:
    payload = _load_config(relpath)
    params = payload["role_adapters"][0]["params"]
    visualizer = params["visualizer"]
    players = params["players"]

    assert params["game_kit"] == "gomoku"
    assert params["env"] == "gomoku_standard"
    assert params["human_input"] == {
        "enabled": True,
        "host": "127.0.0.1",
        "port": 0,
    }
    assert params["runtime_overrides"] == expected_overrides
    assert visualizer["enabled"] is True
    assert visualizer["launch_browser"] is True
    assert visualizer["mode"] == "arena_visual"
    assert visualizer["linger_after_finish_s"] == 15.0
    assert [player["seat"] for player in players] == ["black", "white"]
    assert [player["player_id"] for player in players] == ["Black", "White"]
    assert [player["player_kind"] for player in players] == ["human", "llm"]
    assert players[1]["backend_id"] == "local_qwen35_litellm_backend"


def test_tictactoe_human_visual_config_routes_human_input_to_x() -> None:
    payload = _load_config("config/custom/tictactoe/tictactoe_human_visual_gamekit.yaml")
    params = payload["role_adapters"][0]["params"]
    visualizer = params["visualizer"]
    players = params["players"]

    assert params["game_kit"] == "tictactoe"
    assert params["env"] == "tictactoe_standard"
    assert params["human_input"] == {
        "enabled": True,
        "host": "127.0.0.1",
        "port": 0,
    }
    assert params["runtime_overrides"] == {
        "coord_scheme": "ROW_COL",
    }
    assert visualizer["enabled"] is True
    assert visualizer["launch_browser"] is True
    assert visualizer["mode"] == "arena_visual"
    assert visualizer["linger_after_finish_s"] == 15.0
    assert [player["seat"] for player in players] == ["x", "o"]
    assert [player["player_id"] for player in players] == ["X", "O"]
    assert [player["player_kind"] for player in players] == ["human", "llm"]
    assert players[1]["backend_id"] == "local_qwen35_litellm_backend"


@pytest.mark.parametrize(
    "relpath",
    [
        "config/custom/doudizhu/doudizhu_human_visual_gamekit.yaml",
        "config/custom/doudizhu/doudizhu_human_visual_acceptance_gamekit.yaml",
    ],
)
def test_doudizhu_human_visual_config_uses_real_env_and_llm_opponents(relpath: str) -> None:
    payload = _load_config(relpath)
    params = payload["role_adapters"][0]["params"]
    players = params["players"]

    assert params["env"] == "classic_3p_real"
    assert params["runtime_overrides"]["chat_mode"] == "all"
    assert params["runtime_overrides"]["chat_every_n"] == 1
    assert payload["backends"][0]["backend_id"] == "local_qwen35_litellm_backend"
    assert players[0]["player_kind"] == "human"
    assert players[1]["player_kind"] == "llm"
    assert players[2]["player_kind"] == "llm"
    assert players[1]["backend_id"] == "local_qwen35_litellm_backend"
    assert players[2]["backend_id"] == "local_qwen35_litellm_backend"


@pytest.mark.parametrize(
    "relpath",
    [
        "config/custom/mahjong/mahjong_human_visual_gamekit.yaml",
        "config/custom/mahjong/mahjong_human_visual_acceptance_gamekit.yaml",
    ],
)
def test_mahjong_human_visual_config_uses_real_env_and_llm_opponents(relpath: str) -> None:
    payload = _load_config(relpath)
    params = payload["role_adapters"][0]["params"]
    players = params["players"]

    assert params["env"] == "riichi_4p_real"
    assert params["runtime_overrides"]["chat_mode"] == "all"
    assert params["runtime_overrides"]["chat_every_n"] == 1
    assert payload["backends"][0]["backend_id"] == "local_qwen35_litellm_backend"
    assert players[0]["player_kind"] == "human"
    assert [player["player_kind"] for player in players[1:]] == ["llm", "llm", "llm"]
    assert all(
        player["backend_id"] == "local_qwen35_litellm_backend"
        for player in players[1:]
    )


def test_phase_card_manifest_entries_point_to_gamekit_modules() -> None:
    manifest = _load_arena_manifest()
    entry_map = {entry["name"]: entry for entry in manifest["entries"]}

    assert "doudizhu_arena_parser_v1" not in entry_map
    assert _LEGACY_DDZ_RENDERER not in entry_map
    assert entry_map["doudizhu_v1"]["module"] == (
        "gage_eval.game_kits.phase_card_game.doudizhu.parsers.doudizhu"
    )
    assert entry_map["doudizhu_replay_v1"]["module"] == (
        "gage_eval.game_kits.phase_card_game.doudizhu.renderers.doudizhu"
    )
    assert entry_map["mahjong_rlcard_v1"]["module"] == (
        "gage_eval.game_kits.phase_card_game.mahjong.environment"
    )
    assert entry_map["mahjong_v1"]["module"] == (
        "gage_eval.game_kits.phase_card_game.mahjong.parsers.mahjong"
    )
    assert entry_map["mahjong_replay_v1"]["module"] == (
        "gage_eval.game_kits.phase_card_game.mahjong.renderers.mahjong"
    )


def test_doudizhu_dummy_replay_config_uses_gamekit_parser() -> None:
    payload = _load_config("config/custom/oneclick/replay_dummy/doudizhu_dummy_replay.yaml")
    arena_adapter = next(
        adapter for adapter in payload["role_adapters"] if adapter["adapter_id"] == "doudizhu_arena"
    )
    parser = arena_adapter["params"]["parser"]

    assert parser["impl"] == "doudizhu_v1"


@pytest.mark.parametrize(
    ("relpath", "expected_scheme"),
    [
        (
            "config/custom/pettingzoo/space_invaders_dummy_visual_gamekit.yaml",
            "http_pull",
        ),
        (
            "config/custom/pettingzoo/space_invaders_dummy_visual_binary_stream_gamekit.yaml",
            "binary_stream",
        ),
        (
            "config/custom/pettingzoo/space_invaders_dummy_visual_low_latency_channel_gamekit.yaml",
            "low_latency_channel",
        ),
    ],
)
def test_pettingzoo_dummy_visual_configs_cover_visual_schemes(relpath: str, expected_scheme: str) -> None:
    payload = _load_config(relpath)
    params = payload["role_adapters"][0]["params"]
    visualizer = params["visualizer"]
    players = params["players"]

    assert params["game_kit"] == "pettingzoo"
    assert params["env"] == "space_invaders"
    assert params["runtime_overrides"] == {
        "backend_mode": "real",
        "max_cycles": 500,
        "include_raw_obs": True,
        "use_action_meanings": True,
    }
    assert visualizer["enabled"] is True
    assert visualizer["launch_browser"] is True
    assert visualizer["mode"] == "arena_visual"
    assert visualizer["live_scene_scheme"] == expected_scheme
    assert visualizer["linger_after_finish_s"] == 15.0
    assert [player["player_kind"] for player in players] == ["dummy", "dummy"]
    assert players[0]["seat"] == "pilot_0"
    assert players[1]["seat"] == "pilot_1"
    assert players[0]["player_id"] == "pilot_0"
    assert players[1]["player_id"] == "pilot_1"
    assert players[0]["actions"] == ["FIRE", "RIGHT"]
    assert players[1]["actions"] == ["LEFT", "NOOP"]


def test_space_invaders_human_visual_config_uses_gymnasium_atari_single_player() -> None:
    payload = _load_config("config/custom/pettingzoo/space_invaders_human_visual_gamekit.yaml")
    params = payload["role_adapters"][0]["params"]
    visualizer = params["visualizer"]
    players = params["players"]

    assert params["game_kit"] == "gymnasium_atari"
    assert params["env"] == "space_invaders"
    assert params["human_input"] == {
        "enabled": True,
        "host": "127.0.0.1",
        "port": 0,
    }
    assert params["runtime_overrides"] == {
        "scheduler": "real_time_tick/default",
        "backend_mode": "real",
        "max_cycles": 7200,
        "max_steps": 7200,
        "max_ticks": 7200,
        "env_kwargs": {
            "env_id": "ALE/SpaceInvaders-v5",
            "frameskip": 2,
        },
        "action_schema": {
            "hold_ticks_min": 1,
            "hold_ticks_max": 4,
            "hold_ticks_default": 1,
        },
        "runtime_binding_policy_config": {
            "mode": "scheduler_owned_human_realtime",
            "activation_scope": "pure_human_only",
            "input_model": "queued_command",
            "input_transport": "realtime_ws",
            "tick_interval_ms": 33,
            "frame_output_hz": 30,
            "artifact_sampling_mode": "async_decimated_live",
            "snapshot_persist_stride": 3,
            "max_commands_per_tick": 1,
            "max_command_queue_size": 128,
            "queue_overflow_policy": "drop_oldest",
            "bridge_stall_timeout_ms": 2000,
        },
    }
    assert visualizer["enabled"] is True
    assert visualizer["launch_browser"] is True
    assert visualizer["mode"] == "arena_visual"
    assert visualizer["live_scene_scheme"] == "low_latency_channel"
    assert visualizer["linger_after_finish_s"] == 15.0
    assert players == [
        {
            "seat": "pilot_0",
            "player_id": "pilot_0",
            "player_kind": "human",
            "driver_params": {
                "input_semantics": "queued_command",
                "tick_interval_ms": 33,
            },
        }
    ]


def test_retro_mario_human_visual_config_routes_human_input_to_player_0() -> None:
    payload = _load_config("config/custom/retro_mario/retro_mario_human_visual_gamekit.yaml")
    params = payload["role_adapters"][0]["params"]
    visualizer = params["visualizer"]
    players = params["players"]

    assert params["game_kit"] == "retro_platformer"
    assert params["env"] == "retro_mario"
    assert params["human_input"] == {
        "enabled": True,
        "host": "127.0.0.1",
        "port": 0,
    }
    assert params["runtime_overrides"] == {
        "backend_mode": "real",
        "default_state": "Start",
        "display_mode": "headless",
        "obs_image": False,
        "frame_stride": 1,
        "legal_moves": [
            "noop",
            "left",
            "right",
            "up",
            "down",
            "jump",
            "run",
            "left_jump",
            "right_jump",
            "left_run",
            "right_run",
            "left_run_jump",
            "right_run_jump",
            "start",
            "select",
        ],
        "action_schema": {
            "hold_ticks_min": 1,
            "hold_ticks_max": 30,
            "hold_ticks_default": 10,
        },
        "max_turns": 7200,
        "runtime_binding_policy_config": {
            "mode": "scheduler_owned_human_realtime",
            "activation_scope": "pure_human_only",
            "input_model": "continuous_state",
            "input_transport": "realtime_ws",
            "tick_interval_ms": 16,
            "frame_output_hz": 60,
            "artifact_sampling_mode": "async_decimated_live",
            "snapshot_persist_stride": 3,
            "fallback_move": "noop",
        },
    }
    assert visualizer["enabled"] is True
    assert visualizer["launch_browser"] is True
    assert visualizer["mode"] == "arena_visual"
    assert visualizer["live_scene_scheme"] == "low_latency_channel"
    assert visualizer["linger_after_finish_s"] == 15.0
    assert players == [
        {
            "seat": "player_0",
            "player_id": "player_0",
            "player_kind": "human",
            "driver_params": {
                "input_semantics": "continuous_state",
                "stateful_actions": True,
                "tick_interval_ms": 16,
                "timeout_ms": 16,
                "timeout_fallback_move": "noop",
            },
        }
    ]


def test_vizdoom_human_visual_config_routes_human_input_to_p0() -> None:
    payload = _load_config("config/custom/vizdoom/vizdoom_human_visual_gamekit.yaml")
    params = payload["role_adapters"][0]["params"]
    visualizer = params["visualizer"]
    players = params["players"]

    assert params["game_kit"] == "vizdoom"
    assert params["env"] == "duel_map01"
    assert params["human_input"] == {
        "enabled": True,
        "host": "127.0.0.1",
        "port": 0,
    }
    assert params["runtime_overrides"] == {
        "backend_mode": "real",
        "show_pov": False,
        "capture_pov": True,
        "obs_image": False,
        "frame_stride": 1,
        "max_steps": 600,
        "action_repeat": 1,
        "sleep_s": 0.0,
        "allow_partial_actions": True,
        "allow_respawn": True,
        "respawn_grace_steps": 600,
        "reset_retry_count": 3,
        "death_check_warmup_steps": 8,
    }
    assert visualizer["enabled"] is True
    assert visualizer["launch_browser"] is True
    assert visualizer["mode"] == "arena_visual"
    assert visualizer["live_scene_scheme"] == "low_latency_channel"
    assert visualizer["linger_after_finish_s"] == 15.0
    assert players == [
        {
            "seat": "p0",
            "player_id": "p0",
            "player_kind": "human",
            "driver_params": {
                "input_semantics": "continuous_state",
                "stateful_actions": True,
                "tick_interval_ms": 16,
                "timeout_ms": 16,
                "timeout_fallback_move": "0",
            },
        },
        {
            "seat": "p1",
            "player_id": "p1",
            "player_kind": "dummy",
            "actions": ["3", "2", "3"],
        },
    ]


def test_pettingzoo_double_llm_visual_config_routes_both_seats_to_same_backend() -> None:
    payload = _load_config(
        "config/custom/pettingzoo/space_invaders_double_llm_visual_gamekit.yaml"
    )
    params = payload["role_adapters"][0]["params"]
    visualizer = params["visualizer"]
    players = params["players"]

    assert params["game_kit"] == "pettingzoo"
    assert params["env"] == "space_invaders"
    assert params["runtime_overrides"] == {
        "backend_mode": "real",
        "max_cycles": 200,
        "include_raw_obs": True,
        "use_action_meanings": True,
    }
    assert visualizer["enabled"] is True
    assert visualizer["launch_browser"] is True
    assert visualizer["mode"] == "arena_visual"
    assert visualizer["live_scene_scheme"] == "http_pull"
    assert visualizer["linger_after_finish_s"] == 15.0
    assert [player["seat"] for player in players] == ["pilot_0", "pilot_1"]
    assert [player["player_id"] for player in players] == ["pilot_0", "pilot_1"]
    assert [player["player_kind"] for player in players] == ["llm", "llm"]
    assert all(
        player["backend_id"] == "local_qwen35_litellm_backend"
        for player in players
    )


def test_pettingzoo_double_llm_low_latency_visual_config_routes_both_seats_to_same_backend() -> None:
    payload = _load_config(
        "config/custom/pettingzoo/space_invaders_double_llm_visual_low_latency_channel_gamekit.yaml"
    )
    params = payload["role_adapters"][0]["params"]
    visualizer = params["visualizer"]
    players = params["players"]

    assert params["game_kit"] == "pettingzoo"
    assert params["env"] == "space_invaders"
    assert params["runtime_overrides"] == {
        "backend_mode": "real",
        "max_cycles": 200,
        "include_raw_obs": True,
        "use_action_meanings": True,
    }
    assert visualizer["enabled"] is True
    assert visualizer["launch_browser"] is True
    assert visualizer["mode"] == "arena_visual"
    assert visualizer["live_scene_scheme"] == "low_latency_channel"
    assert visualizer["linger_after_finish_s"] == 15.0
    assert [player["seat"] for player in players] == ["pilot_0", "pilot_1"]
    assert [player["player_id"] for player in players] == ["pilot_0", "pilot_1"]
    assert [player["player_kind"] for player in players] == ["llm", "llm"]
    assert all(
        player["backend_id"] == "local_qwen35_litellm_backend"
        for player in players
    )
