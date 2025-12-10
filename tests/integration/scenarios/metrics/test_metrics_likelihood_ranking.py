import math

from gage_eval.config.pipeline_config import PipelineConfig
from gage_eval.metrics import MetricRegistry, MetricContext
from gage_eval.observability.trace import ObservabilityTrace


def _build_pipeline_config(metrics_config):
    # 补足 PipelineConfig 所需的最小字段
    base = {
        "datasets": [{"dataset_id": "d1", "loader": "dummy"}],
        "role_adapters": [{"adapter_id": "r1", "role_type": "dut_model"}],
        "custom": {"steps": [{"step": "auto_eval"}]},
    }
    return PipelineConfig.from_dict({**base, "metrics": metrics_config})


def test_likelihood_pipeline_with_ppl():
    config = _build_pipeline_config(
        [
            {"likelihood": {"metric_type": "ppl"}},
        ]
    )
    registry = MetricRegistry()
    instances = [registry.build_metric(spec) for spec in config.metrics]

    samples = [
        {
            "id": "s1",
            "model_output": {"token_logprobs": [-1.0, -2.0]},
        },
        {
            "id": "s2",
            "model_output": {"loss": 1.0},
        },
    ]

    trace = ObservabilityTrace()
    for sample in samples:
        for instance in instances:
            ctx = MetricContext(
                sample_id=str(sample.get("id")),
                sample=sample,
                model_output=sample.get("model_output", {}),
                judge_output={},
                args=instance.spec.params,
                trace=trace,
            )
            instance.evaluate(ctx)

    aggregated = [inst.finalize() for inst in instances]
    ppl_metric = aggregated[0]
    expected = (math.exp(1.5) + math.exp(1.0)) / 2  # avg loss of samples -> ppl
    assert abs(ppl_metric["values"]["nll"] - expected) < 1e-6
    assert ppl_metric["count"] == 2


def test_ranking_pipeline_with_objects_and_hit_at_k():
    config = _build_pipeline_config(
        [
            {"ranking": {"metric_type": "hit@k", "k": 2, "candidate_field": "name"}},
        ]
    )
    registry = MetricRegistry()
    instances = [registry.build_metric(spec) for spec in config.metrics]

    samples = [
        {
            "id": "s1",
            "targets": ["tool_b"],
            "model_output": {
                "candidates": [
                    {"name": "tool_a", "desc": "a"},
                    {"name": "tool_b", "desc": "b"},
                ]
            },
        },
        {
            "id": "s2",
            "targets": ["tool_x"],
            "model_output": {
                "candidates": [
                    {"name": "tool_c", "desc": "c"},
                    {"name": "tool_d", "desc": "d"},
                ]
            },
        },
    ]

    trace = ObservabilityTrace()
    for sample in samples:
        for instance in instances:
            ctx = MetricContext(
                sample_id=str(sample.get("id")),
                sample=sample,
                model_output=sample.get("model_output", {}),
                judge_output={},
                args=instance.spec.params,
                trace=trace,
            )
            instance.evaluate(ctx)

    aggregated = [inst.finalize() for inst in instances]
    rank_metric = aggregated[0]
    # 第一条命中 rank=2 -> hit@k=1，第二条未命中 -> 0，均值 0.5
    assert abs(rank_metric["values"]["hit"] - 0.5) < 1e-6
    assert rank_metric["count"] == 2
