from __future__ import annotations

import pytest

from gage_eval.role.role_instance import Role
from gage_eval.role.role_pool import RolePool


def _build_role() -> Role:
    return Role("demo", object())


@pytest.mark.fast
def test_role_pool_snapshot_tracks_capacity_and_usage() -> None:
    pool = RolePool(adapter_id="demo", builder=_build_role, max_size=2)

    initial = pool.snapshot()

    assert initial["pool_type"] == "role_pool"
    assert initial["adapter_id"] == "demo"
    assert initial["capacity"] == 2
    assert initial["in_use"] == 0
    assert initial["available"] == 2
    assert initial["created"] == 0
    assert initial["healthy"] is True
    assert initial["closed"] is False

    with pool.acquire():
        active = pool.snapshot()
        assert active["capacity"] == 2
        assert active["in_use"] == 1
        assert active["available"] == 1
        assert active["created"] == 1
        assert active["healthy"] is True
        assert active["closed"] is False

    released = pool.snapshot()

    assert released["capacity"] == 2
    assert released["in_use"] == 0
    assert released["available"] == 2
    assert released["created"] == 1
    assert released["healthy"] is True
    assert released["closed"] is False


@pytest.mark.fast
def test_role_pool_snapshot_marks_closed_pool_unhealthy() -> None:
    pool = RolePool(adapter_id="demo", builder=_build_role, max_size=1)

    with pool.acquire():
        pass

    pool.shutdown()
    snapshot = pool.snapshot()

    assert snapshot["capacity"] == 1
    assert snapshot["in_use"] == 0
    assert snapshot["available"] == 0
    assert snapshot["created"] == 1
    assert snapshot["healthy"] is False
    assert snapshot["closed"] is True
