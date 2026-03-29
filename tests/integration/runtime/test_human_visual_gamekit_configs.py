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
