from unittest.mock import MagicMock

from gage_eval.role.arena.players.llm_player import LLMPlayer
from gage_eval.role.arena.types import ArenaObservation


def test_llm_player_uses_pettingzoo_prompt():
    sample = {"messages": [], "metadata": {}}
    parser = MagicMock()
    role_manager = MagicMock()
    player = LLMPlayer(
        name="player_0",
        adapter_id="dummy",
        role_manager=role_manager,
        sample=sample,
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
    assert "NOOP" in prompt
    assert "Output ONLY the action label or id" in prompt
    assert "Current Board:" not in prompt
