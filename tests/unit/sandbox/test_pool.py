from __future__ import annotations

from gage_eval.sandbox.pool import SandboxPool


class _FakeSandbox:
    def __init__(self, *, alive: bool = True) -> None:
        self._alive = alive
        self.teardown_calls = 0

    def is_alive(self, timeout_s: float | None = None) -> bool:  # noqa: ARG002
        return self._alive

    def teardown(self) -> None:
        self.teardown_calls += 1


def test_pool_reuses_healthy_sandbox() -> None:
    created: list[_FakeSandbox] = []

    def builder() -> _FakeSandbox:
        sandbox = _FakeSandbox(alive=True)
        created.append(sandbox)
        return sandbox

    pool = SandboxPool(builder=builder)
    first = pool.acquire()
    pool.release(first)
    second = pool.acquire()

    assert first is second
    assert len(created) == 1


def test_pool_discards_unhealthy_sandbox_and_rebuilds() -> None:
    created: list[_FakeSandbox] = []

    def builder() -> _FakeSandbox:
        sandbox = _FakeSandbox(alive=True)
        created.append(sandbox)
        return sandbox

    pool = SandboxPool(builder=builder)
    first = pool.acquire()
    pool.release(first)
    first._alive = False

    second = pool.acquire()

    assert second is not first
    assert len(created) == 2
    assert first.teardown_calls == 1
