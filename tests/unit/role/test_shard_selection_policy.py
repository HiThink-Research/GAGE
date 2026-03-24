from __future__ import annotations

import pytest

from gage_eval.role.runtime.shard_selection import (
    LeastInUsePolicy,
    RoundRobinPolicy,
    ShardSelectionContext,
    ShardSnapshot,
    WeightedPolicy,
    normalize_shard_scheduling_config,
)


def _snapshot(shard_id: str, *, in_use: int, healthy: bool = True, **metadata) -> ShardSnapshot:
    return ShardSnapshot(shard_id=shard_id, in_use=in_use, healthy=healthy, metadata=metadata)


@pytest.mark.fast
def test_least_in_use_skips_unhealthy_and_excluded_shards() -> None:
    policy = LeastInUsePolicy()
    decision = policy.select(
        [
            _snapshot("demo:0", in_use=2),
            _snapshot("demo:1", in_use=0, healthy=False),
            _snapshot("demo:2", in_use=1),
        ],
        ShardSelectionContext(adapter_id="demo", attempt=1, excluded_shards=("demo:2",)),
    )

    assert decision.shard_id == "demo:0"
    assert decision.policy_name == "least_in_use"


@pytest.mark.fast
def test_round_robin_rotates_across_eligible_shards() -> None:
    policy = RoundRobinPolicy()
    shards = [
        _snapshot("demo:0", in_use=0),
        _snapshot("demo:1", in_use=0),
        _snapshot("demo:2", in_use=0),
    ]
    picks = [
        policy.select(shards, ShardSelectionContext(adapter_id="demo", attempt=index + 1)).shard_id
        for index in range(5)
    ]

    assert picks == ["demo:0", "demo:1", "demo:2", "demo:0", "demo:1"]


@pytest.mark.fast
def test_weighted_prefers_higher_weight_when_load_matches() -> None:
    policy = WeightedPolicy({"demo:0": 1.0, "demo:1": 3.0})
    decision = policy.select(
        [
            _snapshot("demo:0", in_use=0),
            _snapshot("demo:1", in_use=0),
        ],
        ShardSelectionContext(adapter_id="demo", attempt=1),
    )

    assert decision.shard_id == "demo:1"
    assert decision.policy_name == "weighted"
    assert decision.trace_tags["selected_weight"] == pytest.approx(3.0)


@pytest.mark.fast
def test_normalize_shard_scheduling_config_supports_flat_policy_field() -> None:
    config = normalize_shard_scheduling_config({"shard_selection_policy": "round_robin"})

    assert config.policy == "round_robin"
    assert config.fallback_policy == "least_in_use"
    assert config.notify_mode == "single"


@pytest.mark.fast
def test_normalize_shard_scheduling_config_rejects_weighted_without_weight_source() -> None:
    with pytest.raises(ValueError, match="weighted policy requires at least one positive weight source"):
        normalize_shard_scheduling_config({"shard_selection_policy": "weighted"})
