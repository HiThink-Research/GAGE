import pytest

from gage_eval.role.arena.games.retro.action_codec import RetroActionCodec


def test_retro_action_codec_exposes_buttons_and_encodes_macro_moves():
    codec = RetroActionCodec(buttons=["LEFT", "RIGHT", "A", "B", "SELECT", "START"])

    assert codec.buttons() == ["LEFT", "RIGHT", "A", "B", "SELECT", "START"]
    assert "right_run_jump" in codec.legal_moves()

    encoded = codec.encode("right_run_jump")
    assert encoded.pressed == ["RIGHT", "A", "B"]
    assert len(encoded.buttons) == len(codec.buttons())


def test_retro_action_codec_rejects_unknown_moves():
    codec = RetroActionCodec(buttons=["LEFT", "RIGHT", "A", "B"])

    with pytest.raises(ValueError, match="unknown retro move"):
        codec.encode("does_not_exist")


def test_retro_action_codec_requires_non_empty_buttons():
    with pytest.raises(ValueError, match="buttons list is empty"):
        RetroActionCodec(buttons=[])


def test_retro_action_codec_filters_moves_by_available_buttons_and_legal_moves():
    codec = RetroActionCodec(
        buttons=["LEFT", "RIGHT", "A"],
        legal_moves=["noop", "left", "jump", "run", "right_run_jump"],
    )

    assert codec.legal_moves() == ["noop", "left", "jump"]
    assert codec.encode("jump").pressed == ["A"]


def test_retro_action_codec_reinserts_noop_when_overridden_with_invalid_combo():
    codec = RetroActionCodec(
        buttons=["LEFT"],
        macro_map={"noop": ["A"]},
        legal_moves=["left"],
    )

    assert codec.legal_moves()[:2] == ["noop", "left"]
    assert codec.encode("noop").buttons == [0]
