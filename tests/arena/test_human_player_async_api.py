import time
from unittest.mock import MagicMock

import pytest

from gage_eval.role.arena.players.human_player import HumanPlayer
from gage_eval.role.arena.types import ArenaAction, ArenaObservation


def _wait_until(predicate, *, timeout_s: float = 1.0) -> None:
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        if predicate():
            return
        time.sleep(0.001)
    raise AssertionError("Timed out waiting for predicate")


def test_human_player_start_thinking_emits_latency_metadata(monkeypatch: pytest.MonkeyPatch):
    sample = {"messages": [], "metadata": {}}
    parser = MagicMock()
    role_manager = MagicMock()
    player = HumanPlayer(
        name="player_0",
        adapter_id="dummy",
        role_manager=role_manager,
        sample=sample,
        parser=parser,
    )

    def _fake_think(_: ArenaObservation) -> ArenaAction:
        return ArenaAction(player="player_0", move="noop", raw="noop")

    monkeypatch.setattr(player, "think", _fake_think)

    observation = ArenaObservation(
        board_text="board",
        legal_moves=["noop"],
        active_player="player_0",
        view={"text": "board"},
        legal_actions={"items": ["noop"]},
    )

    player.start_thinking(observation)
    _wait_until(player.has_action)
    action = player.pop_action()

    assert action.move == "noop"
    assert isinstance(action.metadata.get("latency_ms"), int)


def test_human_player_pop_action_raises_when_worker_failed(monkeypatch: pytest.MonkeyPatch):
    sample = {"messages": [], "metadata": {}}
    parser = MagicMock()
    role_manager = MagicMock()
    player = HumanPlayer(
        name="player_0",
        adapter_id="dummy",
        role_manager=role_manager,
        sample=sample,
        parser=parser,
    )

    def _boom(_: ArenaObservation) -> ArenaAction:
        raise RuntimeError("boom")

    monkeypatch.setattr(player, "think", _boom)

    observation = ArenaObservation(
        board_text="board",
        legal_moves=["noop"],
        active_player="player_0",
        view={"text": "board"},
        legal_actions={"items": ["noop"]},
    )

    player.start_thinking(observation)
    _wait_until(player.has_action)
    with pytest.raises(RuntimeError, match="failed to produce an action"):
        player.pop_action()

