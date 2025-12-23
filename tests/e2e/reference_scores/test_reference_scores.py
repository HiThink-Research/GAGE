import json
from pathlib import Path

import pytest

from gage_eval.config.pipeline_config import MetricSpec
from gage_eval.metrics import MetricRegistry, MetricContext
from gage_eval.observability.trace import ObservabilityTrace


def _load_baseline(name: str) -> dict:
    path = Path(__file__).resolve().parents[2] / "data" / "baselines" / f"{name}.json"
    return json.loads(path.read_text(encoding="utf-8"))


def test_reference_scores_det_semantic():
    baseline = _load_baseline("baseline_text")
    registry = MetricRegistry()
    specs = [
        MetricSpec(metric_id="exact_match", implementation="exact_match", aggregation="mean"),
        MetricSpec(metric_id="latency", implementation="latency", aggregation="mean"),
    ]
    instances = [registry.build_metric(spec) for spec in specs]

    samples = [
        {"id": "s1", "label": "Paris", "model_output": {"answer": "Paris", "latency_ms": 50}},
        {"id": "s2", "label": "Paris", "model_output": {"answer": "London", "latency_ms": 60}},
    ]
    trace = ObservabilityTrace()
    for sample in samples:
        ctx = MetricContext(
            sample_id=sample["id"],
            sample=sample,
            model_output=sample["model_output"],
            judge_output={},
            args={},
            trace=trace,
        )
        for inst in instances:
            inst.evaluate(ctx)

    aggregated = {inst.spec.metric_id: inst.finalize() for inst in instances}

    for entry in baseline["deterministic"]:
        metric_id = entry["metric_id"]
        expected = entry["expected"]
        delta = entry.get("delta", 0.0)
        value = aggregated[metric_id]["values"]["score"]
        assert pytest.approx(expected, abs=delta) == value

    for entry in baseline["semantic"]:
        metric_id = entry["metric_id"]
        expected = entry["expected"]
        delta = entry.get("delta", 0.05)
        value = aggregated[metric_id]["values"]["latency_ms"]
        assert abs(value - expected) <= delta
