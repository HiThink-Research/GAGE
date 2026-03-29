from __future__ import annotations

from pathlib import Path

import pytest
import yaml


REPO_ROOT = Path(__file__).resolve().parents[3]


def _load_config(relpath: str) -> dict:
    return yaml.safe_load((REPO_ROOT / relpath).read_text(encoding="utf-8"))


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
    assert payload["backends"][0]["backend_id"] == "local_qwen35_litellm_backend"
    assert players[0]["player_kind"] == "human"
    assert [player["player_kind"] for player in players[1:]] == ["llm", "llm", "llm"]
    assert all(
        player["backend_id"] == "local_qwen35_litellm_backend"
        for player in players[1:]
    )


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
        "max_cycles": 32,
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


def test_pettingzoo_human_visual_config_routes_human_input_to_pilot_0() -> None:
    payload = _load_config("config/custom/pettingzoo/space_invaders_human_visual_gamekit.yaml")
    params = payload["role_adapters"][0]["params"]
    visualizer = params["visualizer"]
    players = params["players"]

    assert params["game_kit"] == "pettingzoo"
    assert params["env"] == "space_invaders"
    assert params["human_input"] == {
        "enabled": True,
        "host": "127.0.0.1",
        "port": 0,
    }
    assert params["runtime_overrides"] == {
        "backend_mode": "real",
        "max_cycles": 8,
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
    assert [player["player_kind"] for player in players] == ["human", "dummy"]
    assert players[1]["actions"] == ["LEFT", "NOOP"]


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
        "max_cycles": 32,
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
        "max_cycles": 32,
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
