"""Ranking / retrieval style metrics."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

from gage_eval.metrics.base import MetricContext, SimpleMetric
from gage_eval.metrics.utils import ensure_list_of_strings
from gage_eval.registry import registry


def _extract_candidates(candidates_raw: Any, candidate_field: str | None) -> list[str]:
    if candidates_raw is None:
        return []
    if candidate_field:
        items = candidates_raw if isinstance(candidates_raw, (list, tuple)) else [candidates_raw]
        extracted: list[str] = []
        for item in items:
            if isinstance(item, Mapping):
                val = item.get(candidate_field)
                if val is not None:
                    extracted.append(str(val))
            else:
                extracted.append(str(item))
        return extracted
    return ensure_list_of_strings(candidates_raw)


@registry.asset(
    "metrics",
    "ranking",
    desc="MRR / Hit@K 排序命中指标",
    tags=("ranking", "retrieval"),
    default_aggregation="mean",
)
class RankingMetric(SimpleMetric):
    """计算排名命中率，支持 MRR 与 Hit@K。"""

    value_key = "hit"

    def compute_value(self, context: MetricContext) -> tuple[float, dict]:
        candidate_field = self.args.get("candidate_field")
        candidates_raw = context.get("model_output.candidates", [])
        candidates = _extract_candidates(candidates_raw, candidate_field)

        targets = set(ensure_list_of_strings(context.get("sample.targets", [])))
        metric_type = str(self.args.get("metric_type", "mrr")).lower()
        k = int(self.args.get("k", 3))

        hit_rank = next((i for i, c in enumerate(candidates, 1) if c in targets), None)
        mrr = 0.0
        hit_at_k = 0.0
        if hit_rank is not None:
            mrr = 1.0 / hit_rank
            hit_at_k = 1.0 if hit_rank <= k else 0.0

        metadata = {
            "mrr": mrr,
            "hit_at_k": hit_at_k,
            "hit_rank": hit_rank,
            "k": k,
            "metric_type": metric_type,
            "candidate_field": candidate_field,
        }

        if metric_type == "hit@k":
            return hit_at_k, metadata
        return mrr, metadata

