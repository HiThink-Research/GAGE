from gage_eval.config.pipeline_config import PipelineConfig
from gage_eval.metrics import MetricRegistry, MetricContext
from gage_eval.observability.trace import ObservabilityTrace


def test_e2e_metrics_pipeline():
    # 1) 配置简写（字符串、函数式、KV）
    config_dict = {
        "metrics": [
            "exact_match",  # 纯字符串
            "regex_match(pattern='\\d+',aggregation='mean')",  # 函数式简写
            {"numeric_match": {"tolerance": 0.1}},  # KV 简写
        ]
    }
    pipeline_config = PipelineConfig.from_dict(config_dict | {"datasets": [{"dataset_id": "d1", "loader": "dummy"}], "role_adapters": [{"adapter_id": "r1", "role_type": "dut_model"}], "custom": {"steps": [{"step": "auto_eval"}]}})
    assert len(pipeline_config.metrics) == 3
    registry = MetricRegistry()
    instances = [registry.build_metric(spec) for spec in pipeline_config.metrics]

    # 2) 样本与模型输出（包含缺失字段场景，触发 warn/默认）
    samples = [
        {
            "id": "s1",
            "label": "Paris",
            "model_output": {"answer": "Paris", "latency_ms": 12},
        },
        {
            "id": "s2",
            # 缺失 label，numeric 预测不匹配
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

    # 3) 聚合结果检查
    aggregated = [inst.finalize() for inst in instances]
    # exact_match: 1/2 命中 -> 均值 0.5
    exact = next(a for a in aggregated if a["metric_id"] == "exact_match")
    assert abs(exact["values"]["score"] - 0.5) < 1e-6

    # regex_match: 匹配不到数字 -> 均值 0
    regex = next(a for a in aggregated if a["metric_id"] == "regex_match")
    assert regex["values"].get("score", 0) == 0

    # numeric_match: 第一条缺 label -> 0，第二条解析失败 -> 0，均值 0
    num = next(a for a in aggregated if a["metric_id"] == "numeric_match")
    assert num["values"].get("score", 0) == 0

    # 4) 元数据存在 prediction/reference 等关键字段
    for res in results:
        if res.metadata:
            assert isinstance(res.metadata, dict)
