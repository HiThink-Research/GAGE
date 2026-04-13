from gage_eval.game_kits.real_time_game.vizdoom.environment import _encode_frame
from gage_eval.game_kits.real_time_game.vizdoom.parser import VizDoomParser


class _FakeFrame:
    shape = (2, 2, 3)
    dtype = "uint8"

    def tobytes(self) -> bytes:
        return bytes([0, 64, 128, 255]) * 3


def test_vizdoom_encode_frame_includes_inline_data_url(monkeypatch) -> None:
    monkeypatch.setattr(
        "gage_eval.game_kits.real_time_game.vizdoom.environment._build_image_data_url",
        lambda frame: "data:image/jpeg;base64,Zm9v",
    )

    encoded = _encode_frame(_FakeFrame())

    assert encoded is not None
    assert encoded["data_url"] == "data:image/jpeg;base64,Zm9v"
    assert encoded["shape"] == [2, 2, 3]


def test_vizdoom_parser_is_gamekit_owned_and_parses_reason_lines() -> None:
    parser = VizDoomParser(default_action=0)

    result = parser.parse(
        "Action: 2\nReason: Enemy appears on the left side, so move left to center.",
        legal_moves=["1", "2", "3"],
    )

    assert result.coord == "2"
    assert result.error is None
    assert result.reason == "Enemy appears on the left side, so move left to center."
