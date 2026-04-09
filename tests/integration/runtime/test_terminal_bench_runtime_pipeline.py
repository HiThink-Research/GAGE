from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from gage_eval.config import build_default_registry
from gage_eval.config.pipeline_config import PipelineConfig
from gage_eval.evaluation.runtime_builder import build_runtime
from gage_eval.observability.trace import ObservabilityTrace
from gage_eval.role.resource_profile import NodeResource, ResourceProfile


@pytest.mark.fast
def test_terminal_runtime_config_binds_runtime_judge_into_summary_and_artifacts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("GAGE_EVAL_SAVE_DIR", str(tmp_path))
    config_path = (
        Path(__file__).resolve().parents[3]
        / "config"
        / "custom"
        / "terminal_bench"
        / "terminal_bench_smoke_runtime.yaml"
    )
    payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    payload["tasks"][0]["max_samples"] = 1

    config = PipelineConfig.from_dict(payload)
    registry = build_default_registry()
    resource_profile = ResourceProfile(nodes=[NodeResource(node_id="local", gpus=0, cpus=1)])
    trace = ObservabilityTrace(run_id="terminal-runtime-config-test")

    runtime = build_runtime(config, registry, resource_profile, trace=trace)
    captured: list[dict] = []
    for entry in runtime._tasks:
        entry.sample_loop.register_hook(lambda sample, store=captured: store.append(sample))

    runtime.run()

    assert captured
    sample = captured[0]
    assert sample["eval_result"]["resolved"] is True
    runtime_session = sample["predict_result"][0]["runtime_session"]
    sample_root = Path(runtime_session["sample_root"])
    assert (sample_root / "artifacts" / "tool_trace.json").exists()
    assert (sample_root / "artifacts" / "stdout.log").exists()
    assert (sample_root / "artifacts" / "stderr.log").exists()
    assert (sample_root / "artifacts" / "workspace_diff.json").exists()

    summary = json.loads((tmp_path / trace.run_id / "summary.json").read_text(encoding="utf-8"))
    metric_ids = {entry["metric_id"] for entry in summary["metrics"]}
    assert "terminal_bench_resolve_rate" in metric_ids
    assert "terminal_bench_failure_reason" in metric_ids

    sample_dir = tmp_path / trace.run_id / "samples" / "task_terminal_bench_smoke_runtime"
    record_path = next(sample_dir.glob("*.json"))
    record = json.loads(record_path.read_text(encoding="utf-8"))
    assert record["judge_output"]["resolved"] is True
    assert record["sample"]["eval_result"]["resolved"] is True
    assert record["model_output"]["runtime_judge_outcome"]["judge_output"]["artifact_paths"]["tool_trace"] == (
        "artifacts/tool_trace.json"
    )
