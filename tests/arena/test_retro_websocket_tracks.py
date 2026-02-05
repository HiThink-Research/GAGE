import pytest

from gage_eval.role.arena.games.retro import websocket_tracks


def test_parse_key_payload_accepts_bytes_and_rejects_invalid_json():
    payload = websocket_tracks.parse_key_payload(b'{"type":"keydown","key":"d"}')
    assert payload == {"type": "keydown", "key": "d"}

    assert websocket_tracks.parse_key_payload("not json") is None
    assert websocket_tracks.parse_key_payload('["list"]') is None


def test_encode_jpeg_returns_clear_errors_when_deps_missing(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(websocket_tracks, "Image", None)
    monkeypatch.setattr(websocket_tracks, "np", object())
    _, error = websocket_tracks.encode_jpeg(frame=object())
    assert error == "pillow_missing"

    monkeypatch.setattr(websocket_tracks, "Image", object())
    monkeypatch.setattr(websocket_tracks, "np", None)
    _, error = websocket_tracks.encode_jpeg(frame=object())
    assert error == "numpy_missing"

    monkeypatch.setattr(websocket_tracks, "Image", object())
    monkeypatch.setattr(websocket_tracks, "np", object())
    _, error = websocket_tracks.encode_jpeg(frame=object())
    assert error == "frame_missing_shape"


def test_encode_jpeg_encodes_small_rgb_frame():
    np = pytest.importorskip("numpy")

    frame = np.zeros((2, 3, 3), dtype=np.uint8)
    jpeg_bytes, error = websocket_tracks.encode_jpeg(frame=frame, quality=50)

    assert error is None
    assert isinstance(jpeg_bytes, (bytes, bytearray))
    assert jpeg_bytes
