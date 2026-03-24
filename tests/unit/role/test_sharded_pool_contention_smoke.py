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


def _make_contention_pool() -> ShardedRolePool:
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
            ),
            PoolShard(
                shard_id="demo:1",
                pool=RolePool(
                    adapter_id="demo:1",
                    builder=lambda: _build_role("demo"),
                    max_size=1,
                ),
            ),
        ],
    )


@pytest.mark.fast
def test_sharded_pool_contention_smoke_avoids_polling_signature(monkeypatch) -> None:
    pool = _make_contention_pool()
    original_sleep = sharded_pool_module.time.sleep
    poll_calls: list[float] = []

    def tracking_sleep(seconds: float) -> None:
        if seconds == 0.01:
            poll_calls.append(seconds)
        original_sleep(seconds)

    monkeypatch.setattr(sharded_pool_module.time, "sleep", tracking_sleep)

    start = threading.Event()
    errors: list[BaseException] = []

    def worker() -> None:
        start.wait()
        try:
            for _ in range(8):
                with pool.acquire(timeout=1.0):
                    time.sleep(0.001)
        except BaseException as exc:  # pragma: no cover - asserted below
            errors.append(exc)

    threads = [threading.Thread(target=worker) for _ in range(4)]
    for thread in threads:
        thread.start()

    start.set()
    for thread in threads:
        thread.join(timeout=2.0)

    assert not errors
    assert all(not thread.is_alive() for thread in threads)
    assert poll_calls == []

    snapshot = pool.snapshot()
    assert snapshot["extensions"]["notify_total"] > 0
    assert snapshot["extensions"]["waiting_threads"] == 0
