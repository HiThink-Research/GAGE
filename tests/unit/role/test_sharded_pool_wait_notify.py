from __future__ import annotations

import threading
import time

import pytest

from gage_eval.role.role_instance import Role
from gage_eval.role.role_pool import RolePool
import gage_eval.role.runtime.sharded_pool as sharded_pool_module
from gage_eval.role.runtime.sharded_pool import PoolShard, ShardedRolePool


def _build_role(adapter_id: str) -> Role:
    return Role(adapter_id, object())


def _make_single_shard_pool(*, healthy: bool = True) -> ShardedRolePool:
    return ShardedRolePool(
        "demo",
        [
            PoolShard(
                shard_id="demo:0",
                pool=RolePool(
                    adapter_id="demo:0",
                    builder=lambda: _build_role("demo"),
                    max_size=1,
                ),
                healthy=healthy,
            )
        ],
    )


def _wait_for_waiter(pool: ShardedRolePool, *, timeout_s: float = 0.5) -> None:
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        if pool.snapshot()["extensions"]["waiting_threads"] >= 1:
            return
        time.sleep(0.005)
    raise AssertionError("waiter did not block on the sharded pool")


@pytest.mark.fast
def test_sharded_pool_release_wakes_waiter_without_polling(monkeypatch) -> None:
    pool = _make_single_shard_pool()
    sleep_calls: list[float] = []
    original_sleep = sharded_pool_module.time.sleep

    def fail_on_poll(seconds: float) -> None:
        if seconds == 0.01:
            sleep_calls.append(seconds)
            raise AssertionError("polling sleep should not be used by ShardedRolePool")
        original_sleep(seconds)

    monkeypatch.setattr(sharded_pool_module.time, "sleep", fail_on_poll)

    acquired = threading.Event()
    finished = threading.Event()
    errors: list[BaseException] = []

    def waiter() -> None:
        try:
            with pool.acquire(timeout=0.5):
                acquired.set()
        except BaseException as exc:  # pragma: no cover - asserted below
            errors.append(exc)
        finally:
            finished.set()

    with pool.acquire():
        thread = threading.Thread(target=waiter)
        thread.start()
        _wait_for_waiter(pool)

    assert finished.wait(0.5)
    thread.join(timeout=0.2)
    assert acquired.is_set()
    assert not errors
    assert sleep_calls == []


@pytest.mark.fast
def test_sharded_pool_health_recovery_notifies_waiter() -> None:
    pool = _make_single_shard_pool(healthy=False)
    acquired = threading.Event()
    finished = threading.Event()
    errors: list[BaseException] = []

    def waiter() -> None:
        try:
            with pool.acquire(timeout=0.5):
                acquired.set()
        except BaseException as exc:  # pragma: no cover - asserted below
            errors.append(exc)
        finally:
            finished.set()

    thread = threading.Thread(target=waiter)
    thread.start()
    _wait_for_waiter(pool)

    pool.set_shard_health("demo:0", True)

    assert finished.wait(0.5)
    thread.join(timeout=0.2)
    assert acquired.is_set()
    assert not errors
