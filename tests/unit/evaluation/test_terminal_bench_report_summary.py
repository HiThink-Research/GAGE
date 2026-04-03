from __future__ import annotations

import json

from gage_eval.config.pipeline_config import MetricSpec
from gage_eval.evaluation.cache import EvalCache
from gage_eval.observability.trace import ObservabilityTrace
from gage_eval.pipeline.steps.auto_eval import AutoEvalStep
from gage_eval.pipeline.steps.report import ReportStep
from gage_eval.reporting.recorders import InMemoryRecorder


def test_terminal_bench_report_summary_uses_standard_metrics_format(tmp_path) -> None:
    trace = ObservabilityTrace(
        recorder=InMemoryRecorder(run_id="terminal-bench-summary"),
        run_id="terminal-bench-summary",
    )
    cache = EvalCache(base_dir=tmp_path, run_id=trace.run_id)
    auto_eval = AutoEvalStep(
        metric_specs=(
            MetricSpec(
                metric_id="terminal_bench_resolve_rate",
                implementation="terminal_bench_resolve_rate",
            ),
            MetricSpec(
                metric_id="terminal_bench_failure_reason",
                implementation="terminal_bench_failure_reason",
                aggregation="categorical_count",
            ),
        ),
        cache_store=cache,
    )
    sample = {
        "id": "tb2__1",
        "_dataset_id": "tb2_smoke",
        "eval_result": {
            "status": "failed",
            "score": 0.0,
            "summary": "Missing required surfaces: terminal",
            "resolved": False,
            "failure_reason": "missing_required_surfaces",
            "raw_output": {
                "resolved": False,
                "failure_reason": "missing_required_surfaces",
            },
        },
    }

    auto_eval.execute(
        sample_id="tb2__1",
        sample=sample,
        model_output={},
        judge_output={},
        trace=trace,
        task_id="tb2_smoke_agent_eval",
    )
    report = ReportStep(auto_eval_step=auto_eval, cache_store=cache)
    payload = report.finalize(trace)
    summary = json.loads((cache.run_dir / "summary.json").read_text(encoding="utf-8"))

    assert payload["metrics"][0]["metric_id"] == "terminal_bench_resolve_rate"
    assert payload["metrics"][0]["values"]["resolve_rate"] == "0.00000"
    assert payload["metrics"][0]["raw_values"]["resolve_rate"] == 0.0
    assert payload["metrics"][1]["metric_id"] == "terminal_bench_failure_reason"
    assert payload["metrics"][1]["values"]["missing_required_surfaces"] == "1.00000"
    assert summary["metrics"][0]["metric_id"] == "terminal_bench_resolve_rate"
    assert summary["metrics"][1]["metric_id"] == "terminal_bench_failure_reason"
