from __future__ import annotations

from typing import Any

import pytest

from gage_eval.game_kits.real_time_game.vizdoom import env_vizdoom_mp_proc as mp_proc_module


def _make_reset_message(health: float) -> dict[str, Any]:
    return {"done": False, "obs": {"HEALTH": health}}


class _FakeConn:
    def __init__(self) -> None:
        self.sent: list[dict[str, Any]] = []
        self.closed = False

    def send(self, payload: dict[str, Any]) -> None:
        self.sent.append(payload)

    def close(self) -> None:
        self.closed = True


class _GracefulProc:
    def __init__(self) -> None:
        self.pid = 1234
        self._alive = True
        self.join_calls: list[float | None] = []
        self.terminate_calls = 0
        self.kill_calls = 0

    def is_alive(self) -> bool:
        return self._alive

    def join(self, timeout: float | None = None) -> None:
        self.join_calls.append(timeout)
        self._alive = False

    def terminate(self) -> None:
        self.terminate_calls += 1
        self._alive = False

    def kill(self) -> None:
        self.kill_calls += 1
        self._alive = False


class _StubbornTerminateProc(_GracefulProc):
    def join(self, timeout: float | None = None) -> None:
        self.join_calls.append(timeout)

    def terminate(self) -> None:
        self.terminate_calls += 1
        self._alive = False


class _StubbornKillProc(_GracefulProc):
    def join(self, timeout: float | None = None) -> None:
        self.join_calls.append(timeout)

    def terminate(self) -> None:
        self.terminate_calls += 1

    def kill(self) -> None:
        self.kill_calls += 1
        self._alive = False


def test_reset_restarts_workers_when_stale_state_detected(monkeypatch: pytest.MonkeyPatch) -> None:
    cfg = mp_proc_module.ViZDoomMPProcEnvConfig(
        show_automap=False,
        show_pov=False,
        reset_retry_count=1,
    )
    env = mp_proc_module.ViZDoomMPProcEnv(cfg)

    start_calls: list[int] = []
    dispose_calls: list[int] = []
    reset_calls: list[tuple[Any, int]] = []
    reset_states = [
        (
            _make_reset_message(-1.0),
            _make_reset_message(100.0),
            True,
        ),
        (
            _make_reset_message(100.0),
            _make_reset_message(100.0),
            False,
        ),
    ]

    def _fake_start_processes() -> None:
        start_calls.append(1)
        env._host_proc = object()
        env._join_proc = object()
        env._host_conn = object()
        env._join_conn = object()

    def _fake_reset_with_existing_processes(*, seed: Any, attempts: int):
        reset_calls.append((seed, attempts))
        return reset_states.pop(0)

    def _fake_dispose_runtime_processes() -> None:
        dispose_calls.append(1)
        env._host_proc = None
        env._join_proc = None
        env._host_conn = None
        env._join_conn = None
        env.port = None

    monkeypatch.setattr(mp_proc_module.time, "sleep", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(env, "_start_processes", _fake_start_processes)
    monkeypatch.setattr(env, "_reset_with_existing_processes", _fake_reset_with_existing_processes)
    monkeypatch.setattr(env, "_dispose_runtime_processes", _fake_dispose_runtime_processes)

    obs_by_player = env.reset(seed=7)

    assert obs_by_player[0]["HEALTH"] == 100.0
    assert obs_by_player[1]["HEALTH"] == 100.0
    assert len(start_calls) == 2
    assert len(dispose_calls) == 1
    assert reset_calls == [(7, 1), (7, 1)]


def test_reset_raises_after_exhausting_stale_recovery(monkeypatch: pytest.MonkeyPatch) -> None:
    cfg = mp_proc_module.ViZDoomMPProcEnvConfig(
        show_automap=False,
        show_pov=False,
        reset_retry_count=1,
    )
    env = mp_proc_module.ViZDoomMPProcEnv(cfg)

    start_calls: list[int] = []
    dispose_calls: list[int] = []

    def _fake_start_processes() -> None:
        start_calls.append(1)
        env._host_proc = object()
        env._join_proc = object()
        env._host_conn = object()
        env._join_conn = object()

    def _fake_reset_with_existing_processes(*, seed: Any, attempts: int):
        _ = seed, attempts
        return _make_reset_message(-1.0), _make_reset_message(100.0), True

    def _fake_dispose_runtime_processes() -> None:
        dispose_calls.append(1)
        env._host_proc = None
        env._join_proc = None
        env._host_conn = None
        env._join_conn = None
        env.port = None

    monkeypatch.setattr(mp_proc_module.time, "sleep", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(env, "_start_processes", _fake_start_processes)
    monkeypatch.setattr(env, "_reset_with_existing_processes", _fake_reset_with_existing_processes)
    monkeypatch.setattr(env, "_dispose_runtime_processes", _fake_dispose_runtime_processes)

    with pytest.raises(RuntimeError, match="stale death/intermission state persisted"):
        env.reset(seed=None)

    expected_restarts = mp_proc_module._MAX_STALE_RESET_RECOVERY_RESTARTS + 1
    assert len(start_calls) == expected_restarts
    assert len(dispose_calls) == expected_restarts
    assert env._host_proc is None
    assert env._join_proc is None


def test_terminate_process_allows_worker_to_exit_gracefully() -> None:
    env = mp_proc_module.ViZDoomMPProcEnv(mp_proc_module.ViZDoomMPProcEnvConfig())
    conn = _FakeConn()
    proc = _GracefulProc()

    env._terminate_process(proc, conn)

    assert conn.sent == [{"type": "close"}]
    assert conn.closed is True
    assert proc.join_calls == [mp_proc_module._WORKER_CLOSE_GRACE_TIMEOUT_S]
    assert proc.terminate_calls == 0
    assert proc.kill_calls == 0


def test_terminate_process_falls_back_to_forceful_terminate() -> None:
    env = mp_proc_module.ViZDoomMPProcEnv(mp_proc_module.ViZDoomMPProcEnvConfig())
    conn = _FakeConn()
    proc = _StubbornTerminateProc()

    env._terminate_process(proc, conn)

    assert conn.sent == [{"type": "close"}]
    assert conn.closed is True
    assert proc.join_calls == [
        mp_proc_module._WORKER_CLOSE_GRACE_TIMEOUT_S,
        mp_proc_module._WORKER_FORCE_JOIN_TIMEOUT_S,
    ]
    assert proc.terminate_calls == 1
    assert proc.kill_calls == 0


def test_terminate_process_kills_worker_when_terminate_is_insufficient() -> None:
    env = mp_proc_module.ViZDoomMPProcEnv(mp_proc_module.ViZDoomMPProcEnvConfig())
    conn = _FakeConn()
    proc = _StubbornKillProc()

    env._terminate_process(proc, conn)

    assert conn.sent == [{"type": "close"}]
    assert conn.closed is True
    assert proc.join_calls == [
        mp_proc_module._WORKER_CLOSE_GRACE_TIMEOUT_S,
        mp_proc_module._WORKER_FORCE_JOIN_TIMEOUT_S,
        mp_proc_module._WORKER_FORCE_JOIN_TIMEOUT_S,
    ]
    assert proc.terminate_calls == 1
    assert proc.kill_calls == 1
