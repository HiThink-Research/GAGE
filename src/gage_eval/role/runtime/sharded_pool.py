"""Composite pool that shards capacity across multiple RolePools."""

from __future__ import annotations

import threading
import time
from contextlib import AbstractContextManager
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from loguru import logger
from gage_eval.role.role_instance import Role
from gage_eval.role.role_pool import RolePool
from gage_eval.role.runtime.base_pool import BasePool
from gage_eval.role.runtime.rate_limiter import RateLimiter
from gage_eval.role.runtime.shard_selection import (
    LeastInUsePolicy,
    ShardSchedulingConfig,
    ShardSelectionContext,
    ShardSelectionDecision,
    ShardSelectionPolicy,
    ShardSnapshot,
    build_shard_selection_policies,
)


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

    def __init__(
        self,
        adapter_id: str,
        shards: List[PoolShard],
        *,
        scheduling_config: ShardSchedulingConfig | None = None,
        selection_policy: ShardSelectionPolicy | None = None,
        fallback_policy: ShardSelectionPolicy | None = None,
    ) -> None:
        if not shards:
            raise ValueError("ShardedRolePool requires at least one shard")
        config = scheduling_config or ShardSchedulingConfig()
        default_selection, default_fallback = build_shard_selection_policies(config)
        self.adapter_id = adapter_id
        self._shards = shards
        self._lock = threading.Lock()
        self._condition = threading.Condition(self._lock)
        self._lease_map: Dict[int, PoolShard] = {}
        self._selection_policy = selection_policy or default_selection
        self._fallback_policy = fallback_policy or default_fallback
        self._scheduling_config = config
        self._waiting_threads = 0
        self._notify_total = 0
        self._policy_fallback_total = 0
        self._availability_version = 0
        self._closed = False

    def acquire(self, timeout: Optional[float] = None):
        deadline = None if timeout is None else time.monotonic() + timeout
        last_error: Optional[Exception] = None
        seen: set[str] = set()
        attempt = 0

        while True:
            # STEP 1: Snapshot shard state under a tight lock and run policy selection outside the lock.
            attempt += 1
            snapshots, shard_map, observed_version = self._snapshot_shards()
            decision, shard = self._select_shard(
                snapshots=snapshots,
                shard_map=shard_map,
                context=ShardSelectionContext(
                    adapter_id=self.adapter_id,
                    attempt=attempt,
                    excluded_shards=tuple(sorted(seen)),
                    timeout_remaining_ms=_remaining_timeout_ms(deadline),
                    route_tags=self._scheduling_config.route_tags,
                ),
            )

            # STEP 2: Wait on the pool-level condition when no shard is currently usable.
            if shard is None:
                if not self._wait_for_availability(deadline, observed_version):
                    if isinstance(last_error, Exception):
                        raise last_error
                    raise TimeoutError(f"No healthy shard available for adapter '{self.adapter_id}'")
                seen.clear()
                continue

            # STEP 3: Borrow from the leaf pool without holding the outer pool lock.
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
        with self._condition:
            self._closed = True
            for shard in self._shards:
                shard.healthy = False
            self._record_availability_change_unlocked(notify=True, force_broadcast=True)
        if issues:
            joined = "; ".join(issues)
            raise RuntimeError(
                f"ShardedRolePool '{self.adapter_id}' shutdown failed for {len(issues)} shard(s): {joined}"
            )

    def snapshot(self) -> dict[str, Any]:
        with self._condition:
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
            waiting_threads = self._waiting_threads
            notify_total = self._notify_total
            policy_fallback_total = self._policy_fallback_total
            selection_policy = self._selection_policy.name
            fallback_policy = self._fallback_policy.name
            notify_mode = self._scheduling_config.notify_mode
            route_tags = list(self._scheduling_config.route_tags)
            closed = self._closed

        shard_snapshots: list[dict[str, Any]] = []
        capacity_total = 0
        in_use_total = 0
        available_total = 0
        created_total = 0
        overall_healthy = not closed
        unhealthy_shards: list[str] = []

        for shard_view in shard_views:
            pool_snapshot = shard_view["pool"].snapshot()
            capacity = _coerce_non_negative_int(pool_snapshot.get("capacity"))
            in_use = _coerce_non_negative_int(pool_snapshot.get("in_use")) or 0
            available = _coerce_non_negative_int(pool_snapshot.get("available")) or 0
            created = _coerce_non_negative_int(pool_snapshot.get("created")) or 0
            healthy = (not closed) and bool(shard_view["healthy"]) and bool(pool_snapshot.get("healthy", True))
            overall_healthy = overall_healthy and healthy
            if not healthy:
                unhealthy_shards.append(shard_view["shard_id"])
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
            "extensions": {
                "selection_policy": selection_policy,
                "fallback_policy": fallback_policy,
                "waiting_threads": waiting_threads,
                "policy_fallback_total": policy_fallback_total,
                "notify_total": notify_total,
                "notify_mode": notify_mode,
                "route_tags": route_tags,
                "unhealthy_shards": unhealthy_shards,
            },
        }

    def set_shard_health(self, shard_id: str, healthy: bool) -> None:
        """Updates shard health and optionally wakes waiting borrowers."""

        with self._condition:
            shard = self._lookup_shard_unlocked(shard_id)
            changed = shard.healthy != healthy
            shard.healthy = healthy
            if not changed:
                return
            self._record_availability_change_unlocked(
                notify=healthy,
                force_broadcast=healthy,
            )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _snapshot_shards(self) -> tuple[list[ShardSnapshot], dict[str, PoolShard], int]:
        with self._condition:
            shard_views = [
                {
                    "shard_id": shard.shard_id,
                    "healthy": shard.healthy,
                    "in_use": shard.in_use,
                    "metadata": dict(shard.metadata),
                    "pool": shard.pool,
                }
                for shard in self._shards
            ]
            shard_map = {shard.shard_id: shard for shard in self._shards}
            observed_version = self._availability_version

        snapshots: list[ShardSnapshot] = []
        for shard_view in shard_views:
            pool_snapshot = shard_view["pool"].snapshot()
            available = _coerce_non_negative_int(pool_snapshot.get("available"))
            pool_healthy = bool(pool_snapshot.get("healthy", True))
            snapshots.append(
                ShardSnapshot(
                    shard_id=shard_view["shard_id"],
                    healthy=bool(shard_view["healthy"]) and pool_healthy,
                    in_use=shard_view["in_use"],
                    available=available,
                    metadata=shard_view["metadata"],
                )
            )
        return snapshots, shard_map, observed_version

    def _select_shard(
        self,
        *,
        snapshots: list[ShardSnapshot],
        shard_map: dict[str, PoolShard],
        context: ShardSelectionContext,
    ) -> tuple[ShardSelectionDecision, Optional[PoolShard]]:
        decision = self._run_policy_with_fallback(
            snapshots=snapshots,
            shard_map=shard_map,
            context=context,
        )
        if decision.shard_id is None:
            return decision, None
        shard = shard_map[decision.shard_id]
        return decision, shard

    def _wait_for_availability(self, deadline: Optional[float], observed_version: int) -> bool:
        with self._condition:
            if self._closed:
                raise RuntimeError(f"ShardedRolePool '{self.adapter_id}' is shut down")
            if self._availability_version != observed_version:
                return True
            remaining = None if deadline is None else deadline - time.monotonic()
            if remaining is not None and remaining <= 0:
                return False
            self._waiting_threads += 1
            try:
                self._condition.wait(timeout=remaining)
            finally:
                self._waiting_threads = max(0, self._waiting_threads - 1)
            return True

    def _run_policy_with_fallback(
        self,
        *,
        snapshots: list[ShardSnapshot],
        shard_map: dict[str, PoolShard],
        context: ShardSelectionContext,
    ) -> ShardSelectionDecision:
        try:
            decision = self._selection_policy.select(snapshots, context)
            self._validate_decision(decision, shard_map)
            return decision
        except Exception as primary_error:
            if self._selection_policy is self._fallback_policy:
                raise
            with self._condition:
                self._policy_fallback_total += 1
            logger.warning(
                "ShardedRolePool policy fallback adapter_id={} policy={} fallback_policy={} error_type={} error={}",
                self.adapter_id,
                self._selection_policy.name,
                self._fallback_policy.name,
                type(primary_error).__name__,
                primary_error,
            )
            try:
                fallback_decision = self._fallback_policy.select(snapshots, context)
                self._validate_decision(fallback_decision, shard_map)
            except Exception as fallback_error:
                raise RuntimeError(
                    f"Shard policy fallback failed for adapter '{self.adapter_id}'"
                ) from fallback_error
            return ShardSelectionDecision(
                shard_id=fallback_decision.shard_id,
                policy_name=self._fallback_policy.name,
                fallback_used=True,
                reason=fallback_decision.reason,
                score=fallback_decision.score,
                trace_tags=fallback_decision.trace_tags,
            )

    def _mark_in_use(self, shard: PoolShard) -> None:
        with self._condition:
            shard.in_use += 1

    def _after_release(self, shard: PoolShard) -> None:
        with self._condition:
            shard.in_use = max(0, shard.in_use - 1)
            self._record_availability_change_unlocked(notify=True)

    def _register_role(self, shard: PoolShard, role: Role) -> None:
        with self._condition:
            self._lease_map[id(role)] = shard

    def _deregister_role(self, role: Role) -> Optional[PoolShard]:
        with self._condition:
            return self._lease_map.pop(id(role), None)

    def _record_availability_change_unlocked(
        self,
        *,
        notify: bool,
        force_broadcast: bool = False,
    ) -> None:
        self._availability_version += 1
        if not notify:
            return
        self._notify_total += 1
        if force_broadcast or self._scheduling_config.notify_mode == "broadcast":
            self._condition.notify_all()
            return
        self._condition.notify(1)

    def _lookup_shard_unlocked(self, shard_id: str) -> PoolShard:
        for shard in self._shards:
            if shard.shard_id == shard_id:
                return shard
        raise KeyError(f"Unknown shard id: {shard_id}")

    def _validate_decision(
        self,
        decision: ShardSelectionDecision,
        shard_map: dict[str, PoolShard],
    ) -> None:
        if decision.shard_id is None:
            return
        if decision.shard_id not in shard_map:
            raise KeyError(f"Policy selected unknown shard: {decision.shard_id}")


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


def _remaining_timeout_ms(deadline: Optional[float]) -> Optional[int]:
    if deadline is None:
        return None
    return max(0, int((deadline - time.monotonic()) * 1000.0))
