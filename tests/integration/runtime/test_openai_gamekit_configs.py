from __future__ import annotations

from pathlib import Path

import pytest
import yaml


REPO_ROOT = Path(__file__).resolve().parents[3]
CONFIG_ROOT = REPO_ROOT / "config/custom"
OPENAI_CONFIGS = tuple(sorted(CONFIG_ROOT.glob("**/*_openai_gamekit.yaml")))


@pytest.mark.parametrize("path", OPENAI_CONFIGS, ids=lambda path: str(path.relative_to(REPO_ROOT)))
def test_openai_gamekit_configs_use_environment_backends(path: Path) -> None:
    text = path.read_text(encoding="utf-8")
    payload = yaml.safe_load(text)

    assert "local_qwen35_litellm_backend" not in text
    assert "api_key: lmstudio" not in text
    assert "qwen/qwen3.5-9b" not in text

    backend = payload["backends"][0]
    config = backend["config"]
    assert backend["backend_id"] == "openai_litellm_backend"
    assert backend["type"] == "litellm"
    assert config["provider"] == "openai"
    assert config["api_base"] == "${OPENAI_API_BASE:-https://api.openai.com/v1}"
    assert config["api_key"] == "${OPENAI_API_KEY:?set OPENAI_API_KEY}"
    assert config["model"] == "${GAGE_GAME_ARENA_LLM_MODEL:-gpt-5.4}"

    players = payload["role_adapters"][0]["params"]["players"]
    llm_players = [player for player in players if player.get("player_kind") == "llm"]
    assert llm_players
    assert all(player["backend_id"] == "openai_litellm_backend" for player in llm_players)
