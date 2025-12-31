from __future__ import annotations

from gage_eval.role.adapters.arena import ArenaRoleAdapter


def test_player_identity_maps_display_names_to_ids() -> None:
    adapter = ArenaRoleAdapter(
        adapter_id="arena",
        players=[
            {"name": "Black", "type": "backend", "ref": "model_a"},
            {"name": "White", "type": "backend", "ref": "model_b"},
        ],
    )
    sample = {
        "metadata": {
            "player_ids": ["p0", "p1"],
            "player_names": {"p0": "Black", "p1": "White"},
            "start_player_id": "p1",
        }
    }

    specs, player_ids, player_names, start_player_id = adapter._normalize_player_specs(sample)

    assert player_ids == ["p0", "p1"]
    assert [spec["player_id"] for spec in specs] == ["p0", "p1"]
    assert player_names["p0"] == "Black"
    assert start_player_id == "p1"


def test_player_identity_uses_spec_names_without_metadata() -> None:
    adapter = ArenaRoleAdapter(
        adapter_id="arena",
        players=[
            {"name": "Alpha", "type": "backend", "ref": "model_a"},
            {"name": "Beta", "type": "backend", "ref": "model_b"},
        ],
    )
    sample = {"metadata": {}}

    specs, player_ids, player_names, start_player_id = adapter._normalize_player_specs(sample)

    assert player_ids == ["Alpha", "Beta"]
    assert [spec["player_id"] for spec in specs] == ["Alpha", "Beta"]
    assert player_names["Alpha"] == "Alpha"
    assert start_player_id == "Alpha"
