from __future__ import annotations

from unittest.mock import MagicMock

from gage_eval.role.arena.players.llm_player import LLMPlayer


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
