from gage_eval.config.pipeline_config import PipelineConfig
from gage_eval.metrics import MetricRegistry, MetricContext
from gage_eval.observability.trace import ObservabilityTrace


def test_build_metric_from_sugar_strings():
    cfg = PipelineConfig.from_dict(
        {
            "datasets": [{"dataset_id": "d1", "loader": "jsonl"}],
            "role_adapters": [{"adapter_id": "dut", "role_type": "dut_model"}],
            "custom": {"steps": [{"step": "auto_eval"}]},
            "metrics": ["exact_match", "regex_match(pattern='\\d+')"],
        }
    )
    registry = MetricRegistry()
    specs = cfg.metrics
    metrics = [registry.build_metric(spec) for spec in specs]

    sample = {"id": "s1", "label": "Paris"}
    model_output = {"answer": "Paris", "text": "123"}
    trace = ObservabilityTrace()
    for metric in metrics:
        ctx = MetricContext(
            sample_id="s1",
            sample=sample,
            model_output=model_output,
            judge_output={},
            args=metric.spec.params,
            trace=trace,
        )
        res = metric.evaluate(ctx)
        assert res.values

