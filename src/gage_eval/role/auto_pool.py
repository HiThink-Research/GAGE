"""Auto pool planner translating resource profiles into shard plans."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

from gage_eval.role.resource_profile import ResourceProfile


@dataclass
class PoolShardPlan:
    shard_id: str
    size: int
    rate_limit: Optional[Dict[str, Any]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class AutoPoolPlanner:
    """Auto scaling brain derived from llm-eval's worker dispatcher."""

    def plan_instances(self, profile: ResourceProfile, adapter) -> List[PoolShardPlan]:
        requirement = adapter.__dict__.get("resource_requirement", {}) or {}
        base_capacity = self._estimate_capacity(profile, requirement)
        endpoints = _normalize_endpoint_list(requirement.get("endpoint_list") or requirement.get("endpoints"))
        shard_plans: List[PoolShardPlan] = []

        if endpoints:
            per_endpoint = max(1, base_capacity // len(endpoints))
            remainder = base_capacity % len(endpoints)
            for idx, endpoint in enumerate(endpoints):
                size = per_endpoint + (1 if idx < remainder else 0)
                shard_plans.append(
                    PoolShardPlan(
                        shard_id=f"{adapter.adapter_id}:{idx}",
                        size=size,
                        rate_limit=_extract_rate_limit(requirement, endpoint),
                        metadata={
                            "endpoint": endpoint,
                        },
                    )
                )
        else:
            shard_plans.append(
                PoolShardPlan(
                    shard_id=f"{adapter.adapter_id}:default",
                    size=base_capacity,
                    rate_limit=_extract_rate_limit(requirement, None),
                    metadata={},
                )
            )

        return shard_plans

    def _estimate_capacity(self, profile: ResourceProfile, requirement: Dict[str, Any]) -> int:
        explicit_pool = requirement.get("pool_size")
        if isinstance(explicit_pool, int) and explicit_pool > 0:
            return explicit_pool

        per_instance_gpus = requirement.get("gpus", 1) or 1
        total = 0
        for node in profile.nodes:
            available = max(node.gpus, 1)
            total += max(1, available // per_instance_gpus)
        return max(1, total)


def _normalize_endpoint_list(value: Any) -> Sequence[Any]:
    if value is None:
        return ()
    if isinstance(value, (list, tuple)):
        return value
    return (value,)


def _extract_rate_limit(requirement: Dict[str, Any], endpoint: Any) -> Optional[Dict[str, Any]]:
    rate_limit = requirement.get("rate_limit")
    if isinstance(rate_limit, dict):
        return rate_limit
    if isinstance(rate_limit, (list, tuple)) and endpoint is not None:
        idx = _normalize_endpoint_list(requirement.get("endpoint_list") or requirement.get("endpoints")).index(endpoint)
        if idx < len(rate_limit):
            rl = rate_limit[idx]
            if isinstance(rl, dict):
                return rl
    return None
