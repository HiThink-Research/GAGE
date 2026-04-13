from __future__ import annotations

from unittest.mock import MagicMock

from gage_eval.role.arena.core.invocation import GameArenaInvocationContext
from gage_eval.role.arena.core.players import PlayerBindingSpec
from gage_eval.role.arena.player_drivers.llm_backend import LLMBackendDriver
from gage_eval.role.arena.types import ArenaObservation


def _build_llm_bound_player():
    backend = MagicMock()
    role_manager = MagicMock()
    role_manager.get_backend.return_value = backend

    driver = LLMBackendDriver(driver_id="player_driver/llm_backend", family="llm")
    spec = PlayerBindingSpec(
        seat="player_0",
        player_id="player_0",
        player_kind="llm",
        driver_id="player_driver/llm_backend",
        backend_id="dummy",
    )
    player = driver.bind(
        spec,
        invocation=GameArenaInvocationContext(
            adapter_id="arena",
            sample_id="sample-1",
            role_manager=role_manager,
            sample_payload={"messages": [], "metadata": {}},
        ),
    )
    return player, role_manager


def test_llm_backend_prompt_tolerates_magicmock_role_manager() -> None:
    player, _role_manager = _build_llm_bound_player()

    observation = ArenaObservation(
        board_text="PettingZoo env: pettingzoo.atari.space_invaders_v2",
        legal_moves=["NOOP", "FIRE", "LEFT"],
        active_player="player_0",
        metadata={"env_id": "pettingzoo.atari.space_invaders_v2"},
        view={"text": "PettingZoo env: pettingzoo.atari.space_invaders_v2"},
        legal_actions={"items": ["NOOP", "FIRE", "LEFT"]},
    )

    messages = player._build_messages(observation)

    assert "Legal moves:" in messages[-1]["content"]


def test_llm_backend_driver_init_tolerates_magicmock_role_manager() -> None:
    player, role_manager = _build_llm_bound_player()

    assert player.player_id == "player_0"
    role_manager.get_backend.assert_called_once_with("dummy")


def test_llm_backend_uses_first_legal_fallback_for_unparseable_response() -> None:
    player, _role_manager = _build_llm_bound_player()
    player._backend.invoke.return_value = {"answer": "not-a-legal-move"}

    action = player.next_action(
        ArenaObservation(
            board_text="PettingZoo env: pettingzoo.atari.space_invaders_v2",
            legal_moves=["NOOP", "FIRE", "LEFT"],
            active_player="player_0",
            metadata={"env_id": "pettingzoo.atari.space_invaders_v2"},
            view={"text": "PettingZoo env: pettingzoo.atari.space_invaders_v2"},
            legal_actions={"items": ["NOOP", "FIRE", "LEFT"]},
        )
    )

    assert action.move == "NOOP"
