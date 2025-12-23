from gage_eval.metrics.base import MetricContext
from gage_eval.metrics.builtin.ranking import RankingMetric
from gage_eval.config.pipeline_config import MetricSpec
from gage_eval.observability.trace import ObservabilityTrace


def _ctx(sample=None, model_output=None, params=None):
    spec = MetricSpec(metric_id="rank", implementation="ranking", params=params or {})
    metric = RankingMetric(spec)
    ctx = MetricContext(
        sample_id="s1",
        sample=sample or {},
        model_output=model_output or {},
        judge_output={},
        args=spec.params,
        trace=ObservabilityTrace(),
    )
    return metric, ctx


def test_ranking_mrr_default():
    metric, ctx = _ctx(
        sample={"targets": ["docB"]},
        model_output={"candidates": ["docA", "docB", "docC"]},
    )
    result = metric.compute(ctx)
    assert result.values["hit"] == 0.5  # rank 2 -> 1/2
    assert result.metadata["hit_at_k"] == 1.0
    assert result.metadata["hit_rank"] == 2


def test_ranking_hit_at_k_mode():
    metric, ctx = _ctx(
        sample={"targets": ["docC"]},
        model_output={"candidates": ["docA", "docB", "docC", "docD"]},
        params={"metric_type": "hit@k", "k": 2},
    )
    result = metric.compute(ctx)
    assert result.values["hit"] == 0.0  # rank 3 > k=2
    assert result.metadata["hit_rank"] == 3
    assert result.metadata["metric_type"] == "hit@k"


def test_ranking_candidate_field_extracts_from_objects():
    metric, ctx = _ctx(
        sample={"targets": ["tool_b"]},
        model_output={
            "candidates": [
                {"name": "tool_a", "desc": "a"},
                {"name": "tool_b", "desc": "b"},
            ]
        },
        params={"candidate_field": "name", "metric_type": "mrr"},
    )
    result = metric.compute(ctx)
    assert result.values["hit"] == 0.5  # rank 2 -> 1/2
    assert result.metadata["candidate_field"] == "name"
