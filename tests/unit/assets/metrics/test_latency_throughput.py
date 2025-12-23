import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2] / "src"
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from gage_eval.config.pipeline_config import MetricSpec
from gage_eval.metrics import MetricRegistry, MetricContext
from gage_eval.observability.trace import ObservabilityTrace
from gage_eval.evaluation.runtime_builder import TaskOrchestratorRuntime
from gage_eval.pipeline.steps.report import ReportStep
from gage_eval.evaluation.cache import EvalCache


def test_latency_metric_handles_missing():
    spec = MetricSpec(metric_id="latency", implementation="latency", aggregation="mean", params={"default_latency": 5})
    registry = MetricRegistry()
    inst = registry.build_metric(spec)
    samples = [
        {"id": "s1", "model_output": {"latency_ms": 10}},
        {"id": "s2", "model_output": {}},
    ]
    trace = ObservabilityTrace()
    for sample in samples:
        ctx = MetricContext(
            sample_id=sample["id"],
            sample=sample,
            model_output=sample.get("model_output", {}),
            judge_output={},
            args=spec.params,
            trace=trace,
        )
        inst.evaluate(ctx)
    aggregated = inst.finalize()
    assert aggregated["values"]["latency_ms"] == pytest.approx(7.5)


def test_record_throughput_metrics():
    cache = EvalCache(base_dir="runs", run_id="throughput-test")
    step = ReportStep(auto_eval_step=None, cache_store=cache)
    runtime = TaskOrchestratorRuntime([], role_manager=None, trace=ObservabilityTrace(), report_step=step)

    runtime._record_throughput_metrics(sample_count=4, wall_runtime_s=2.0, inference_s=1.0, evaluation_s=0.5)

    timings = cache._timings
    assert timings["throughput_total_samples_per_s"] == pytest.approx(2.0)
    assert timings["latency_total_ms_per_sample"] == pytest.approx(500.0)
    assert timings["throughput_inference_samples_per_s"] == pytest.approx(4.0)
    assert timings["throughput_auto_eval_samples_per_s"] == pytest.approx(8.0)
