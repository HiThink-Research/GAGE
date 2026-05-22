from __future__ import annotations

from gage_eval.assets.datasets.utils.multimodal import embed_remote_image_as_data_url


class _Headers(dict):
    def get_content_type(self) -> str:
        return str(self.get("Content-Type") or "image/png")


class _Response:
    def __init__(self, content: bytes, headers: dict[str, str] | None = None) -> None:
        self._content = content
        self.headers = _Headers(headers or {"Content-Type": "image/png"})

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False

    def read(self, size: int = -1) -> bytes:
        return self._content if size < 0 else self._content[:size]


def test_embed_remote_image_as_data_url_retries_and_uses_cache(monkeypatch, tmp_path) -> None:
    calls = {"count": 0}

    def fake_urlopen(_request, timeout):
        calls["count"] += 1
        if calls["count"] == 1:
            raise OSError("temporary network failure")
        return _Response(b"abc", {"Content-Type": "image/png"})

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)

    first = embed_remote_image_as_data_url(
        "https://example.test/image.png",
        cache_dir=str(tmp_path),
        retries=2,
    )
    second = embed_remote_image_as_data_url(
        "https://example.test/image.png",
        cache_dir=str(tmp_path),
        retries=2,
    )

    assert first == "data:image/png;base64,YWJj"
    assert second == first
    assert calls["count"] == 2
    assert list(tmp_path.glob("remote-*.b64"))


def test_embed_remote_image_as_data_url_rejects_oversized_images(monkeypatch, tmp_path) -> None:
    def fake_urlopen(_request, timeout):
        return _Response(b"abcdef", {"Content-Type": "image/png", "Content-Length": "6"})

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)

    result = embed_remote_image_as_data_url(
        "https://example.test/large.png",
        cache_dir=str(tmp_path),
        max_bytes=3,
    )

    assert result is None
    assert not list(tmp_path.glob("remote-*.b64"))
