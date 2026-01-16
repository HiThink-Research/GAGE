from __future__ import annotations

from urllib.error import HTTPError, URLError

import pytest

from gage_eval.sandbox import docker_runtime


@pytest.mark.fast
def test_wait_for_http_accepts_client_error(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_urlopen(request, timeout=0):  # type: ignore[no-untyped-def]
        raise HTTPError(
            getattr(request, "full_url", "http://example"),
            404,
            "not found",
            hdrs=None,
            fp=None,
        )

    monkeypatch.setattr(docker_runtime, "urlopen", fake_urlopen)

    docker_runtime._wait_for_http("http://example", timeout_s=0.1, interval_s=0.01)


@pytest.mark.fast
def test_wait_for_http_times_out(monkeypatch: pytest.MonkeyPatch) -> None:
    clock = {"now": 0.0}

    def fake_urlopen(request, timeout=0):  # type: ignore[no-untyped-def]
        raise URLError("down")

    def fake_time() -> float:
        return clock["now"]

    def fake_sleep(delay: float) -> None:
        clock["now"] += delay

    monkeypatch.setattr(docker_runtime, "urlopen", fake_urlopen)
    monkeypatch.setattr(docker_runtime.time, "time", fake_time)
    monkeypatch.setattr(docker_runtime.time, "sleep", fake_sleep)

    with pytest.raises(RuntimeError, match="docker_wait_http_timeout"):
        docker_runtime._wait_for_http("http://example", timeout_s=0.3, interval_s=0.1)

    assert clock["now"] >= 0.3


@pytest.mark.fast
def test_wait_for_http_retries_on_connection_reset(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = {"count": 0}

    class FakeResponse:
        status = 200

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

    def fake_urlopen(request, timeout=0):  # type: ignore[no-untyped-def]
        calls["count"] += 1
        if calls["count"] == 1:
            raise ConnectionResetError("reset")
        return FakeResponse()

    monkeypatch.setattr(docker_runtime, "urlopen", fake_urlopen)

    docker_runtime._wait_for_http("http://example", timeout_s=0.2, interval_s=0.01)

    assert calls["count"] == 2
