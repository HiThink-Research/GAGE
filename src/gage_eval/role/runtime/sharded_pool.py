"""Composite pool that shards capacity across multiple RolePools."""

from __future__ import annotations

import threading
import time
from contextlib import AbstractContextManager
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from gage_eval.role.role_instance import Role
from gage_eval.role.role_pool import RolePool
from gage_eval.role.runtime.base_pool import BasePool
from gage_eval.role.runtime.rate_limiter import RateLimiter


@dataclass
class PoolShard:
    shard_id: str
    pool: RolePool
    rate_limiter: Optional[RateLimiter] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    healthy: bool = True
    in_use: int = 0


class ShardedRolePool(BasePool):
    """Distributes acquire requests across multiple RolePool shards."""

    def __init__(self, adapter_id: str, shards: List[PoolShard]) -> None:
        if not shards:
            raise ValueError("ShardedRolePool requires at least one shard")
        self.adapter_id = adapter_id
        self._shards = shards
        self._lock = threading.Lock()
        self._lease_map: Dict[int, PoolShard] = {}
        self._pointer = 0

    def acquire(self, timeout: Optional[float] = None):
        deadline = None if timeout is None else time.monotonic() + timeout
        last_error: Optional[Exception] = None
        seen: set[str] = set()

        while True:
            shard = self._select_shard(exclude=seen)
            if shard is None:
                if deadline is not None and time.monotonic() >= deadline:
                    if isinstance(last_error, Exception):
                        raise last_error
                    raise TimeoutError(f"No healthy shard available for adapter '{self.adapter_id}'")
                time.sleep(0.01)
                seen.clear()
                continue

            wait_remaining = None if deadline is None else max(0.0, deadline - time.monotonic())
            try:
                if shard.rate_limiter:
                    shard.rate_limiter.acquire(wait_remaining)
                lease = shard.pool.acquire(wait_remaining)
                return _ShardLease(self, shard, lease)
            except TimeoutError as exc:
                seen.add(shard.shard_id)
                last_error = exc
                continue

    def release(self, role: Role) -> None:
        shard = self._deregister_role(role)
        if shard is None:
            return
        shard.pool.release(role)
        self._after_release(shard)

    def shutdown(self) -> None:
        issues: list[str] = []
        for shard in self._shards:
            try:
                shard.pool.shutdown()
            except Exception as exc:
                issues.append(f"{shard.shard_id}: {type(exc).__name__}: {exc}")
        if issues:
            joined = "; ".join(issues)
            raise RuntimeError(
                f"ShardedRolePool '{self.adapter_id}' shutdown failed for {len(issues)} shard(s): {joined}"
            )

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            shard_views = [
                {
                    "shard_id": shard.shard_id,
                    "healthy": shard.healthy,
                    "in_use_hint": shard.in_use,
                    "metadata": dict(shard.metadata),
                    "rate_limit": _serialize_rate_limiter(shard.rate_limiter),
                    "pool": shard.pool,
                }
                for shard in self._shards
            ]

        shard_snapshots: list[dict[str, Any]] = []
        capacity_total = 0
        in_use_total = 0
        available_total = 0
        created_total = 0
        overall_healthy = True

        for shard_view in shard_views:
            pool_snapshot = shard_view["pool"].snapshot()
            capacity = _coerce_non_negative_int(pool_snapshot.get("capacity"))
            in_use = _coerce_non_negative_int(pool_snapshot.get("in_use")) or 0
            available = _coerce_non_negative_int(pool_snapshot.get("available")) or 0
            created = _coerce_non_negative_int(pool_snapshot.get("created")) or 0
            healthy = bool(shard_view["healthy"]) and bool(pool_snapshot.get("healthy", True))
            overall_healthy = overall_healthy and healthy
            capacity_total += capacity or 0
            in_use_total += in_use
            available_total += available
            created_total += created
            shard_snapshots.append(
                {
                    "shard_id": shard_view["shard_id"],
                    "capacity": capacity,
                    "in_use": in_use,
                    "available": available,
                    "created": created,
                    "healthy": healthy,
                    "closed": bool(pool_snapshot.get("closed", False)),
                    "rate_limit": shard_view["rate_limit"],
                    "metadata": shard_view["metadata"],
                    "extensions": {
                        "in_use_hint": shard_view["in_use_hint"],
                    },
                }
            )

        return {
            "pool_type": "sharded",
            "adapter_id": self.adapter_id,
            "capacity_total": capacity_total,
            "in_use_total": in_use_total,
            "available_total": available_total,
            "created_total": created_total,
            "healthy": overall_healthy,
            "shard_count": len(shard_snapshots),
            "shards": shard_snapshots,
            "extensions": {},
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _select_shard(self, exclude: set[str]) -> Optional[PoolShard]:
        with self._lock:
            eligible = [s for s in self._shards if s.healthy and s.shard_id not in exclude]
            if not eligible:
                return None
            eligible.sort(key=lambda s: s.in_use)
            return eligible[0]

    def _mark_in_use(self, shard: PoolShard) -> None:
        with self._lock:
            shard.in_use += 1

    def _after_release(self, shard: PoolShard) -> None:
        with self._lock:
            shard.in_use = max(0, shard.in_use - 1)

    def _register_role(self, shard: PoolShard, role: Role) -> None:
        with self._lock:
            self._lease_map[id(role)] = shard

    def _deregister_role(self, role: Role) -> Optional[PoolShard]:
        with self._lock:
            return self._lease_map.pop(id(role), None)


class _ShardLease(AbstractContextManager):
    def __init__(self, owner: ShardedRolePool, shard: PoolShard, inner):
        self._owner = owner
        self._shard = shard
        self._inner = inner
        self._released = False
        self._role: Optional[Role] = None

    def __enter__(self) -> Role:
        self._owner._mark_in_use(self._shard)
        role = self._inner.__enter__()
        self._role = role
        self._owner._register_role(self._shard, role)
        return role

    def __exit__(self, exc_type, exc, tb) -> bool:
        if not self._released:
            self._inner.__exit__(exc_type, exc, tb)
            if self._role is not None:
                self._owner._deregister_role(self._role)
            self._owner._after_release(self._shard)
            self._released = True
        return False


def _coerce_non_negative_int(value: Any) -> Optional[int]:
    if value is None:
        return None
    coerced = int(value)
    return max(0, coerced)


def _serialize_rate_limiter(rate_limiter: Optional[RateLimiter]) -> Optional[dict[str, float]]:
    if rate_limiter is None:
        return None
    return {
        "capacity": float(rate_limiter.capacity),
        "interval": float(rate_limiter.interval),
    }
