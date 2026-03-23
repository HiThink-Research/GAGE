from __future__ import annotations

from unittest.mock import MagicMock

from gage_eval.role.arena.players.llm_player import LLMPlayer
from gage_eval.role.arena.types import ArenaObservation


def test_llm_player_prompt_tolerates_magicmock_role_manager() -> None:
    parser = MagicMock()
    role_manager = MagicMock()

    player = LLMPlayer(
        name="player_0",
        adapter_id="dummy",
        role_manager=role_manager,
        sample={"messages": [], "metadata": {}},
        parser=parser,
    )

    observation = ArenaObservation(
        board_text="PettingZoo env: pettingzoo.atari.space_invaders_v2",
        legal_moves=["NOOP", "FIRE", "LEFT"],
        active_player="player_0",
        metadata={"env_id": "pettingzoo.atari.space_invaders_v2"},
        view={"text": "PettingZoo env: pettingzoo.atari.space_invaders_v2"},
        legal_actions={"items": ["NOOP", "FIRE", "LEFT"]},
    )

    prompt = player._format_observation(observation)

    assert "Legal moves:" in prompt


def test_llm_player_init_tolerates_magicmock_role_manager() -> None:
    parser = MagicMock()
    role_manager = MagicMock()

    player = LLMPlayer(
        name="player_0",
        adapter_id="dummy",
        role_manager=role_manager,
        sample={"messages": [], "metadata": {}},
        parser=parser,
    )

    assert player._emit_model_io_events is True
