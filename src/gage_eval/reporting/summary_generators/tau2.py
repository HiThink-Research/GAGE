"""Tau2 summary generator (reward, pass^k, costs)."""

from __future__ import annotations

import math
from typing import Any, Dict, Optional

from gage_eval.evaluation.cache import EvalCache
from gage_eval.registry import registry
from gage_eval.reporting.summary_generators import SummaryGenerator


@registry.asset(
    "summary_generators",
    "tau2_summary",
    desc="Tau2 summary generator",
    tags=("tau2", "report"),
    default_enabled=True,
)
class Tau2SummaryGenerator(SummaryGenerator):
    def generate(self, cache: EvalCache) -> Optional[Dict[str, Any]]:
        summary = _build_tau2_summary(cache)
        if not summary:
            return None
        return {"tau2_summary": summary}


def _build_tau2_summary(cache: EvalCache) -> Optional[Dict[str, Any]]:
    reward_sum = 0.0
    reward_count = 0
    agent_cost_sum = 0.0
    agent_cost_count = 0
    user_cost_sum = 0.0
    user_cost_count = 0

    task_stats: Dict[str, Dict[str, int]] = {}
    by_domain: Dict[str, Dict[str, float]] = {}

    for record in cache.iter_samples():
        sample = record.get("sample") if isinstance(record, dict) else None
        if not isinstance(sample, dict):
            continue
        meta = sample.get("metadata") if isinstance(sample.get("metadata"), dict) else {}
        tau2_meta = meta.get("tau2") if isinstance(meta.get("tau2"), dict) else None
        if not tau2_meta:
            continue
        judge_output = record.get("judge_output") if isinstance(record.get("judge_output"), dict) else {}
        tau2_output = judge_output.get("tau2") if isinstance(judge_output.get("tau2"), dict) else {}
        reward = _coerce_float(tau2_output.get("reward"))
        if reward is None:
            continue
        reward_sum += reward
        reward_count += 1

        task_id = str(tau2_meta.get("task_id") or sample.get("id"))
        stats = task_stats.setdefault(task_id, {"total": 0, "success": 0})
        stats["total"] += 1
        if _is_success(reward):
            stats["success"] += 1

        domain = str(tau2_meta.get("domain") or "unknown")
        dom_stats = by_domain.setdefault(domain, {"total": 0.0, "reward_sum": 0.0})
        dom_stats["total"] += 1.0
        dom_stats["reward_sum"] += reward

        agent_cost = _coerce_float(tau2_output.get("agent_cost"))
        if agent_cost is not None:
            agent_cost_sum += agent_cost
            agent_cost_count += 1
        user_cost = _coerce_float(tau2_output.get("user_cost"))
        if user_cost is not None:
            user_cost_sum += user_cost
            user_cost_count += 1

    if reward_count == 0:
        return None

    max_k = _resolve_max_k(task_stats)
    pass_hat = {k: _pass_hat_k(task_stats, k) for k in range(1, max_k + 1)}

    return {
        "overall": {
            "total": reward_count,
            "avg_reward": reward_sum / reward_count if reward_count else 0.0,
            "avg_agent_cost": (agent_cost_sum / agent_cost_count) if agent_cost_count else None,
            "avg_user_cost": (user_cost_sum / user_cost_count) if user_cost_count else None,
        },
        "pass_hat_k": pass_hat,
        "by_domain": {
            domain: {
                "total": int(stats["total"]),
                "avg_reward": (stats["reward_sum"] / stats["total"]) if stats["total"] else 0.0,
            }
            for domain, stats in by_domain.items()
        },
    }


def _resolve_max_k(task_stats: Dict[str, Dict[str, int]]) -> int:
    if not task_stats:
        return 0
    return min(stats["total"] for stats in task_stats.values())


def _pass_hat_k(task_stats: Dict[str, Dict[str, int]], k: int) -> float:
    if k <= 0:
        return 0.0
    values = []
    for stats in task_stats.values():
        total = stats["total"]
        success = stats["success"]
        if total < k:
            continue
        if success < k:
            values.append(0.0)
        else:
            values.append(math.comb(success, k) / math.comb(total, k))
    if not values:
        return 0.0
    return sum(values) / len(values)


def _is_success(reward: float) -> bool:
    return (1.0 - 1e-6) <= reward <= (1.0 + 1e-6)


def _coerce_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None
