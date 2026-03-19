"""Runtime helper exports.

The legacy scheduling core (InferenceRuntime / BatchingScheduler) has been
removed. This package only keeps the RolePool-related runtime infrastructure
such as sharded pools and rate limiters.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from gage_eval.role.runtime.base_pool import BasePool
    from gage_eval.role.runtime.rate_limiter import RateLimiter
    from gage_eval.role.runtime.sharded_pool import PoolShard, ShardedRolePool

__all__ = ["BasePool", "PoolShard", "ShardedRolePool", "RateLimiter"]


def __getattr__(name: str) -> Any:
    if name == "BasePool":
        from gage_eval.role.runtime.base_pool import BasePool

        return BasePool
    if name == "PoolShard":
        from gage_eval.role.runtime.sharded_pool import PoolShard

        return PoolShard
    if name == "ShardedRolePool":
        from gage_eval.role.runtime.sharded_pool import ShardedRolePool

        return ShardedRolePool
    if name == "RateLimiter":
        from gage_eval.role.runtime.rate_limiter import RateLimiter

        return RateLimiter
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
