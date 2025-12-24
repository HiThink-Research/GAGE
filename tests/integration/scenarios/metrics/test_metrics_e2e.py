from gage_eval.config.pipeline_config import PipelineConfig
from gage_eval.metrics import MetricRegistry, MetricContext
from gage_eval.observability.trace import ObservabilityTrace


def test_e2e_metrics_pipeline():
    # STEP 1: Metric config sugar (string / function-style / KV shorthand).
    config_dict = {
        "metrics": [
            "exact_match",  # plain string
            "regex_match(pattern='\\d+',aggregation='mean')",  # function-style shorthand
            {"numeric_match": {"tolerance": 0.1}},  # KV shorthand
        ]
    }
    pipeline_config = PipelineConfig.from_dict(config_dict | {"datasets": [{"dataset_id": "d1", "loader": "dummy"}], "role_adapters": [{"adapter_id": "r1", "role_type": "dut_model"}], "custom": {"steps": [{"step": "auto_eval"}]}})
    assert len(pipeline_config.metrics) == 3
    registry = MetricRegistry()
    instances = [registry.build_metric(spec) for spec in pipeline_config.metrics]

    # STEP 2: Samples and model outputs (including missing-field cases).
    samples = [
        {
            "id": "s1",
            "label": "Paris",
            "model_output": {"answer": "Paris", "latency_ms": 12},
        },
        {
            "id": "s2",
            # Missing label: numeric_match should not match.
            "model_output": {"answer": "London", "latency_ms": "bad"},
        },
    ]

    trace = ObservabilityTrace()
    results = []
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
            res = instance.evaluate(ctx)
            results.append(res)

    # STEP 3: Aggregation checks.
    aggregated = [inst.finalize() for inst in instances]
    # exact_match: 1/2 hits -> mean 0.5
    exact = next(a for a in aggregated if a["metric_id"] == "exact_match")
    assert abs(exact["values"]["score"] - 0.5) < 1e-6

    # regex_match: no digits -> mean 0
    regex = next(a for a in aggregated if a["metric_id"] == "regex_match")
    assert regex["values"].get("score", 0) == 0

    # numeric_match: missing label -> 0, parse failure -> 0, mean 0
    num = next(a for a in aggregated if a["metric_id"] == "numeric_match")
    assert num["values"].get("score", 0) == 0

    # STEP 4: Metadata should exist and stay dict-like.
    for res in results:
        if res.metadata:
            assert isinstance(res.metadata, dict)
