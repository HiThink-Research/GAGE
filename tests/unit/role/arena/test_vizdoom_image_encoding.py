from gage_eval.role.arena.games.vizdoom.env import _encode_frame


class _FakeFrame:
    shape = (2, 2, 3)
    dtype = "uint8"

    def tobytes(self) -> bytes:
        return bytes([0, 64, 128, 255]) * 3


def test_vizdoom_encode_frame_includes_inline_data_url(monkeypatch) -> None:
    monkeypatch.setattr(
        "gage_eval.role.arena.games.vizdoom.env._build_image_data_url",
        lambda frame: "data:image/jpeg;base64,Zm9v",
    )

    encoded = _encode_frame(_FakeFrame())

    assert encoded is not None
    assert encoded["data_url"] == "data:image/jpeg;base64,Zm9v"
    assert encoded["shape"] == [2, 2, 3]
