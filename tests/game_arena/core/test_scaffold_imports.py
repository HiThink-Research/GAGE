from __future__ import annotations

import importlib


def test_scaffold_imports() -> None:
    for module_name in (
        "gage_eval.role.arena",
        "gage_eval.role.arena.core",
        "gage_eval.role.arena.output",
        "gage_eval.game_kits",
    ):
        importlib.import_module(module_name)

    from gage_eval.role.arena.core.types import ArenaStopReason, ArenaSample

    assert ArenaStopReason("completed") is ArenaStopReason.COMPLETED
    sample = ArenaSample(game_kit="demo", env=None, players=({"name": "agent"},))
    assert sample.env is None
    assert sample.players == ({"name": "agent"},)
