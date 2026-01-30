"""Tau2 metrics (reward, pass rate, costs)."""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

from gage_eval.metrics.base import MetricContext, SimpleMetric
from gage_eval.registry import registry


def _coerce_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _is_success(reward: float) -> bool:
    return (1.0 - 1e-6) <= reward <= (1.0 + 1e-6)


def _extract_tau2_metadata(context: MetricContext) -> Dict[str, Any]:
    tau2_meta = context.get("sample.metadata.tau2")
    if not isinstance(tau2_meta, dict):
        tau2_meta = {}
    return {
        "task_id": tau2_meta.get("task_id"),
        "domain": tau2_meta.get("domain"),
        "trial": tau2_meta.get("trial"),
        "seed": tau2_meta.get("seed"),
    }


@registry.asset(
    "metrics",
    "tau2_reward",
    desc="Tau2 reward metric",
    tags=("tau2",),
    default_aggregation="mean",
)
class Tau2RewardMetric(SimpleMetric):
    value_key = "reward"

    def compute_value(self, context: MetricContext) -> Tuple[float, Dict[str, Any]]:
        reward = _coerce_float(context.get("judge_output.tau2.reward"))
        if reward is None:
            return 0.0, {"missing_reward": True}
        return reward, {"missing_reward": False}


@registry.asset(
    "metrics",
    "tau2_pass",
    desc="Tau2 pass indicator (reward == 1)",
    tags=("tau2",),
    default_aggregation="mean",
)
class Tau2PassMetric(SimpleMetric):
    value_key = "pass"

    def compute_value(self, context: MetricContext) -> Tuple[float, Dict[str, Any]]:
        reward = _coerce_float(context.get("judge_output.tau2.reward"))
        if reward is None:
            return 0.0, {"missing_reward": True}
        passed = 1.0 if _is_success(reward) else 0.0
        return passed, {"missing_reward": False, "reward": reward}


@registry.asset(
    "metrics",
    "tau2_pass_hat_k",
    desc="Tau2 pass-hat@k (pass^k) metric",
    tags=("tau2",),
    default_aggregation="tau2_pass_hat",
)
class Tau2PassHatMetric(SimpleMetric):
    value_key = "pass"

    def compute_value(self, context: MetricContext) -> Tuple[float, Dict[str, Any]]:
        reward = _coerce_float(context.get("judge_output.tau2.reward"))
        metadata = _extract_tau2_metadata(context)
        if reward is None:
            metadata["missing_reward"] = True
            return 0.0, metadata
        passed = 1.0 if _is_success(reward) else 0.0
        metadata.update({"missing_reward": False, "reward": reward})
        return passed, metadata


@registry.asset(
    "metrics",
    "tau2_agent_cost",
    desc="Tau2 agent cost",
    tags=("tau2",),
    default_aggregation="mean",
)
class Tau2AgentCostMetric(SimpleMetric):
    value_key = "agent_cost"

    def compute_value(self, context: MetricContext) -> Tuple[float, Dict[str, Any]]:
        cost = _coerce_float(context.get("judge_output.tau2.agent_cost"))
        if cost is None:
            return 0.0, {"missing_agent_cost": True}
        return cost, {"missing_agent_cost": False}


@registry.asset(
    "metrics",
    "tau2_user_cost",
    desc="Tau2 user cost",
    tags=("tau2",),
    default_aggregation="mean",
)
class Tau2UserCostMetric(SimpleMetric):
    value_key = "user_cost"

    def compute_value(self, context: MetricContext) -> Tuple[float, Dict[str, Any]]:
        cost = _coerce_float(context.get("judge_output.tau2.user_cost"))
        if cost is None:
            return 0.0, {"missing_user_cost": True}
        return cost, {"missing_user_cost": False}


__all__ = [
    "Tau2RewardMetric",
    "Tau2PassMetric",
    "Tau2PassHatMetric",
    "Tau2AgentCostMetric",
    "Tau2UserCostMetric",
]
