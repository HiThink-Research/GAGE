from __future__ import annotations

from gage_eval.tools import gamekit_acceptance


class _FakeStreamingResponse:
    def __init__(self) -> None:
        self.headers = {"Content-Type": "multipart/x-mixed-replace; boundary=frame"}
        self._lines = iter(
            (
                b"--frame\r\n",
                b"Content-Type: image/png\r\n",
                b"Content-Length: 12\r\n",
                b"\r\n",
            )
        )

    def __enter__(self) -> _FakeStreamingResponse:
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        return False

    def readline(self) -> bytes:
        return next(self._lines, b"")

    def read(self, _size: int = -1) -> bytes:
        raise ConnectionResetError(54, "Connection reset by peer")


def test_read_stream_prefix_tolerates_early_stream_close(monkeypatch) -> None:
    monkeypatch.setattr(
        gamekit_acceptance,
        "urlopen",
        lambda *args, **kwargs: _FakeStreamingResponse(),
    )

    content_type, prefix = gamekit_acceptance._read_stream_prefix(
        "http://127.0.0.1:9999/stream"
    )

    assert content_type.startswith("multipart/x-mixed-replace")
    assert b"--frame" in prefix
    assert b"Content-Type: image/png" in prefix
