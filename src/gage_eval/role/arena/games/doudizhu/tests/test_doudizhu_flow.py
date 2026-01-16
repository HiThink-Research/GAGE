"""End-to-end smoke test for the Doudizhu card arena stack."""

from __future__ import annotations

import random

from gage_eval.role.arena.games.doudizhu.env import GenericCardArena


def test_doudizhu_flow() -> None:
    """Run a random Doudizhu game to validate the arena wiring."""

    arena = GenericCardArena(game_type="doudizhu", rng_seed=7)
    arena.reset()
    rng = random.Random(13)
    max_steps = 1000
    result = None

    for _ in range(max_steps):
        if arena.is_terminal():
            break
        active_player = arena.get_active_player()
        observation = arena.observe(active_player)
        legal_moves = list(observation["legal_moves"])
        assert legal_moves, "Expected at least one legal move"
        action_text = rng.choice(legal_moves)
        result = arena.apply({"player_id": active_player, "action": action_text, "raw": action_text})
        if result is not None:
            break

    assert result is not None, "Expected the game to terminate with a result"
    print("game_result", result)
    print("move_log", result["move_log"])
