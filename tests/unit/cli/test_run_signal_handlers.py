from __future__ import annotations

import signal
from types import SimpleNamespace

import pytest

import run as gage_run


class _HardExit(RuntimeError):
    def __init__(self, code: int) -> None:
        super().__init__(str(code))
        self.code = code


@pytest.mark.fast
def test_first_shutdown_signal_calls_runtime_shutdown_and_raises_system_exit(monkeypatch: pytest.MonkeyPatch) -> None:
    handlers = {}
    calls = []
    hard_exits = []

    monkeypatch.setattr(gage_run.signal, "signal", lambda sig, handler: handlers.setdefault(sig, handler))
    monkeypatch.setattr(gage_run.os, "_exit", lambda code: hard_exits.append(code))
    monkeypatch.setattr(gage_run, "_RUNTIME_REF", SimpleNamespace(shutdown=lambda: calls.append("shutdown")))

    gage_run._install_signal_handlers()

    with pytest.raises(SystemExit) as exc_info:
        handlers[signal.SIGINT](signal.SIGINT, None)

    assert exc_info.value.code == 128 + signal.SIGINT
    assert calls == ["shutdown"]
    assert hard_exits == []


@pytest.mark.fast
def test_second_shutdown_signal_uses_hard_exit_without_repeating_shutdown(monkeypatch: pytest.MonkeyPatch) -> None:
    handlers = {}
    calls = []

    monkeypatch.setattr(gage_run.signal, "signal", lambda sig, handler: handlers.setdefault(sig, handler))

    def fake_exit(code: int) -> None:
        raise _HardExit(code)

    monkeypatch.setattr(gage_run.os, "_exit", fake_exit)
    monkeypatch.setattr(gage_run, "_RUNTIME_REF", SimpleNamespace(shutdown=lambda: calls.append("shutdown")))

    gage_run._install_signal_handlers()
    with pytest.raises(SystemExit):
        handlers[signal.SIGTERM](signal.SIGTERM, None)

    with pytest.raises(_HardExit) as exc_info:
        handlers[signal.SIGTERM](signal.SIGTERM, None)

    assert exc_info.value.code == 128 + signal.SIGTERM
    assert calls == ["shutdown"]
