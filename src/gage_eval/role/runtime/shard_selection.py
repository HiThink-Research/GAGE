"""Shard selection policies for sharded role pools."""

from __future__ import annotations

from dataclasses import dataclass, field
import threading
from typing import Any, Mapping, Protocol, Sequence


_VALID_POLICIES = frozenset({"least_in_use", "round_robin", "weighted"})
_VALID_NOTIFY_MODES = frozenset({"single", "broadcast"})


@dataclass(frozen=True)
class ShardSchedulingConfig:
    """Normalized scheduling configuration for one sharded pool."""

    policy: str = "least_in_use"
    fallback_policy: str = "least_in_use"
    notify_mode: str = "single"
    weights: dict[str, float] = field(default_factory=dict)
    metadata_weight_key: str | None = None
    route_tags: tuple[str, ...] = ()
    extensions: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "policy", self.policy.strip() or "least_in_use")
        object.__setattr__(self, "fallback_policy", self.fallback_policy.strip() or "least_in_use")
        object.__setattr__(self, "notify_mode", self.notify_mode.strip() or "single")
        object.__setattr__(self, "weights", dict(self.weights))
        object.__setattr__(self, "route_tags", tuple(str(tag) for tag in self.route_tags))
        object.__setattr__(self, "extensions", dict(self.extensions))


@dataclass(frozen=True)
class ShardSnapshot:
    """Immutable shard state exposed to selection policies."""

    shard_id: str
    healthy: bool
    in_use: int
    available: int | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "metadata", dict(self.metadata))


@dataclass(frozen=True)
class ShardSelectionContext:
    """Read-only context provided to each policy invocation."""

    adapter_id: str
    attempt: int
    excluded_shards: tuple[str, ...] = ()
    timeout_remaining_ms: int | None = None
    route_tags: tuple[str, ...] = ()
    sample_metadata: dict[str, Any] = field(default_factory=dict)
    extensions: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "excluded_shards", tuple(self.excluded_shards))
        object.__setattr__(self, "route_tags", tuple(str(tag) for tag in self.route_tags))
        object.__setattr__(self, "sample_metadata", dict(self.sample_metadata))
        object.__setattr__(self, "extensions", dict(self.extensions))


@dataclass(frozen=True)
class ShardSelectionDecision:
    """Structured policy output used by the sharded pool."""

    shard_id: str | None
    policy_name: str
    fallback_used: bool = False
    reason: str | None = None
    score: float | None = None
    trace_tags: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "trace_tags", dict(self.trace_tags))


class ShardSelectionPolicy(Protocol):
    """Protocol implemented by shard selection policies."""

    name: str

    def select(
        self,
        shards: Sequence[ShardSnapshot],
        context: ShardSelectionContext,
    ) -> ShardSelectionDecision:
        """Selects one shard from the current candidate set."""


class LeastInUsePolicy:
    """Select the healthy shard with the lowest in-use count."""

    name = "least_in_use"

    def select(
        self,
        shards: Sequence[ShardSnapshot],
        context: ShardSelectionContext,
    ) -> ShardSelectionDecision:
        eligible = _eligible_shards(shards, context)
        if not eligible:
            return ShardSelectionDecision(None, self.name, reason="no_eligible_shard")
        best = min(eligible, key=lambda shard: (shard.in_use, shard.shard_id))
        return ShardSelectionDecision(best.shard_id, self.name, reason="least_in_use")


class RoundRobinPolicy:
    """Rotate selections across healthy shards."""

    name = "round_robin"

    def __init__(self) -> None:
        self._pointer = 0
        self._lock = threading.Lock()

    def select(
        self,
        shards: Sequence[ShardSnapshot],
        context: ShardSelectionContext,
    ) -> ShardSelectionDecision:
        eligible = sorted(_eligible_shards(shards, context), key=lambda shard: shard.shard_id)
        if not eligible:
            return ShardSelectionDecision(None, self.name, reason="no_eligible_shard")
        with self._lock:
            index = self._pointer % len(eligible)
            self._pointer += 1
        return ShardSelectionDecision(eligible[index].shard_id, self.name, reason="round_robin")


class WeightedPolicy:
    """Prefer shards with lower weighted load."""

    name = "weighted"

    def __init__(self, weights: Mapping[str, float], metadata_weight_key: str | None = None) -> None:
        self._weights = dict(weights)
        self._metadata_weight_key = metadata_weight_key

    def select(
        self,
        shards: Sequence[ShardSnapshot],
        context: ShardSelectionContext,
    ) -> ShardSelectionDecision:
        candidates: list[tuple[tuple[float, float, str], ShardSnapshot, float]] = []
        for shard in _eligible_shards(shards, context):
            weight = self._weight_for(shard)
            if weight <= 0:
                continue
            score = shard.in_use / weight
            candidates.append(((score, -weight, shard.shard_id), shard, weight))
        if not candidates:
            return ShardSelectionDecision(None, self.name, reason="no_weighted_candidate")
        _, selected, weight = min(candidates, key=lambda item: item[0])
        return ShardSelectionDecision(
            selected.shard_id,
            self.name,
            reason="lowest_weighted_load",
            score=selected.in_use / weight,
            trace_tags={"selected_weight": weight},
        )

    def _weight_for(self, shard: ShardSnapshot) -> float:
        if shard.shard_id in self._weights:
            return self._weights[shard.shard_id]
        if self._metadata_weight_key:
            raw = shard.metadata.get(self._metadata_weight_key)
            try:
                return float(raw)
            except (TypeError, ValueError):
                return 0.0
        return 1.0


def normalize_shard_scheduling_config(requirement: Mapping[str, Any] | None) -> ShardSchedulingConfig:
    """Builds a validated scheduling config from adapter requirements."""

    requirement_map = dict(requirement or {})
    nested = requirement_map.get("shard_selection") or {}
    if nested and not isinstance(nested, Mapping):
        raise ValueError("resource_requirement.shard_selection must be a mapping when provided")
    nested_map = dict(nested) if isinstance(nested, Mapping) else {}

    policy = str(nested_map.get("policy") or requirement_map.get("shard_selection_policy") or "least_in_use")
    fallback_policy = str(
        nested_map.get("fallback_policy")
        or requirement_map.get("shard_selection_fallback_policy")
        or "least_in_use"
    )
    notify_mode = str(nested_map.get("notify_mode") or requirement_map.get("shard_notify_mode") or "single")
    metadata_weight_key = nested_map.get("metadata_weight_key") or requirement_map.get(
        "shard_selection_metadata_weight_key"
    )
    raw_weights = nested_map.get("weights") or requirement_map.get("shard_selection_weights") or {}
    weights = _normalize_weights(raw_weights)
    route_tags = _normalize_route_tags(
        nested_map.get("route_tags") or requirement_map.get("shard_selection_route_tags") or ()
    )
    extensions = nested_map.get("extensions") if isinstance(nested_map.get("extensions"), Mapping) else {}

    if policy not in _VALID_POLICIES:
        raise ValueError(f"Unsupported shard selection policy: {policy}")
    if fallback_policy not in _VALID_POLICIES:
        raise ValueError(f"Unsupported shard selection fallback policy: {fallback_policy}")
    if notify_mode not in _VALID_NOTIFY_MODES:
        raise ValueError(f"Unsupported shard notify mode: {notify_mode}")
    if policy == "weighted" and not weights and not metadata_weight_key:
        raise ValueError("weighted policy requires at least one positive weight source")

    return ShardSchedulingConfig(
        policy=policy,
        fallback_policy=fallback_policy,
        notify_mode=notify_mode,
        weights=weights,
        metadata_weight_key=str(metadata_weight_key) if metadata_weight_key else None,
        route_tags=route_tags,
        extensions=dict(extensions),
    )


def build_shard_selection_policies(
    config: ShardSchedulingConfig,
) -> tuple[ShardSelectionPolicy, ShardSelectionPolicy]:
    """Creates the main policy and fallback policy for one pool."""

    return _build_policy(config.policy, config), _build_policy(config.fallback_policy, config)


def _build_policy(policy_name: str, config: ShardSchedulingConfig) -> ShardSelectionPolicy:
    if policy_name == "least_in_use":
        return LeastInUsePolicy()
    if policy_name == "round_robin":
        return RoundRobinPolicy()
    if policy_name == "weighted":
        return WeightedPolicy(config.weights, config.metadata_weight_key)
    raise ValueError(f"Unsupported shard selection policy: {policy_name}")


def _normalize_weights(raw_weights: Any) -> dict[str, float]:
    if raw_weights in (None, {}):
        return {}
    if not isinstance(raw_weights, Mapping):
        raise ValueError("shard selection weights must be a mapping")
    normalized: dict[str, float] = {}
    for shard_id, raw_weight in raw_weights.items():
        try:
            weight = float(raw_weight)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Invalid shard weight for {shard_id!r}: {raw_weight!r}") from exc
        if weight <= 0:
            continue
        normalized[str(shard_id)] = weight
    return normalized


def _normalize_route_tags(raw_tags: Any) -> tuple[str, ...]:
    if raw_tags is None:
        return ()
    if isinstance(raw_tags, str):
        return (raw_tags,)
    if isinstance(raw_tags, Sequence):
        return tuple(str(tag) for tag in raw_tags)
    raise ValueError("shard selection route tags must be a string or sequence of strings")


def _eligible_shards(
    shards: Sequence[ShardSnapshot],
    context: ShardSelectionContext,
) -> list[ShardSnapshot]:
    excluded = set(context.excluded_shards)
    return [
        shard
        for shard in shards
        if shard.healthy
        and shard.shard_id not in excluded
        and (shard.available is None or shard.available > 0)
    ]


__all__ = [
    "LeastInUsePolicy",
    "RoundRobinPolicy",
    "ShardSchedulingConfig",
    "ShardSelectionContext",
    "ShardSelectionDecision",
    "ShardSelectionPolicy",
    "ShardSnapshot",
    "WeightedPolicy",
    "build_shard_selection_policies",
    "normalize_shard_scheduling_config",
]
