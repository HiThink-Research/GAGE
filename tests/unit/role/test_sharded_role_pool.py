from __future__ import annotations

import pytest

from gage_eval.role.role_instance import Role
from gage_eval.role.role_pool import RolePool
from gage_eval.role.runtime.shard_selection import LeastInUsePolicy, ShardSelectionContext, ShardSelectionDecision
from gage_eval.role.runtime.sharded_pool import PoolShard, ShardedRolePool


def _build_role(adapter_id: str) -> Role:
    return Role(adapter_id, object())


def _make_sharded_pool() -> ShardedRolePool:
    return ShardedRolePool(
        "demo",
        [
            PoolShard(
                shard_id="demo:0",
                pool=RolePool(
                    adapter_id="demo:0",
                    builder=lambda: _build_role("demo"),
                    max_size=2,
                ),
                metadata={"endpoint": "http://a"},
            ),
            PoolShard(
                shard_id="demo:1",
                pool=RolePool(
                    adapter_id="demo:1",
                    builder=lambda: _build_role("demo"),
                    max_size=2,
                ),
                metadata={"endpoint": "http://b"},
            ),
        ],
    )


@pytest.mark.fast
def test_sharded_role_pool_snapshot_aggregates_leaf_state() -> None:
    pool = _make_sharded_pool()

    initial = pool.snapshot()

    assert initial["pool_type"] == "sharded"
    assert initial["adapter_id"] == "demo"
    assert initial["capacity_total"] == 4
    assert initial["in_use_total"] == 0
    assert initial["available_total"] == 4
    assert initial["created_total"] == 0
    assert initial["healthy"] is True
    assert initial["shard_count"] == 2
    assert initial["extensions"]["selection_policy"] == "least_in_use"
    assert initial["extensions"]["fallback_policy"] == "least_in_use"
    assert initial["extensions"]["waiting_threads"] == 0
    assert initial["extensions"]["policy_fallback_total"] == 0
    assert initial["extensions"]["notify_total"] == 0
    assert len(initial["shards"]) == 2
    assert {shard["metadata"]["endpoint"] for shard in initial["shards"]} == {
        "http://a",
        "http://b",
    }

    with pool.acquire():
        active = pool.snapshot()
        assert active["capacity_total"] == 4
        assert active["in_use_total"] == 1
        assert active["available_total"] == 3
        assert active["created_total"] == 1
        assert sum(shard["in_use"] for shard in active["shards"]) == 1
        assert sum(shard["available"] for shard in active["shards"]) == 3

    released = pool.snapshot()

    assert released["in_use_total"] == 0
    assert released["available_total"] == 4
    assert released["created_total"] == 1
    assert released["extensions"]["notify_total"] >= 1
    assert sum(shard["in_use"] for shard in released["shards"]) == 0


@pytest.mark.fast
def test_sharded_role_pool_snapshot_marks_shutdown_as_unhealthy() -> None:
    pool = _make_sharded_pool()

    pool.shutdown()
    snapshot = pool.snapshot()

    assert snapshot["healthy"] is False
    assert snapshot["available_total"] == 0
    assert all(shard["healthy"] is False for shard in snapshot["shards"])
    assert all(shard["closed"] is True for shard in snapshot["shards"])


class _BrokenPolicy:
    name = "broken"

    def select(self, shards, context):  # noqa: ANN001, ANN201
        del shards, context
        raise RuntimeError("boom")


@pytest.mark.fast
def test_sharded_role_pool_falls_back_to_default_policy_on_runtime_error() -> None:
    pool = _make_sharded_pool()
    broken_pool = ShardedRolePool(
        pool.adapter_id,
        [
            PoolShard(
                shard_id=shard["shard_id"],
                pool=RolePool(
                    adapter_id=shard["shard_id"],
                    builder=lambda: _build_role("demo"),
                    max_size=2,
                ),
                metadata=shard["metadata"],
            )
            for shard in pool.snapshot()["shards"]
        ],
        selection_policy=_BrokenPolicy(),
        fallback_policy=LeastInUsePolicy(),
    )

    with broken_pool.acquire():
        snapshot = broken_pool.snapshot()

    assert snapshot["extensions"]["selection_policy"] == "broken"
    assert snapshot["extensions"]["fallback_policy"] == "least_in_use"
    assert snapshot["extensions"]["policy_fallback_total"] == 1
