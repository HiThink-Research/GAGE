from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from gage_eval.evaluation.cache import EvalCache
from gage_eval.evaluation.execution_controller import TaskExecutionController
from gage_eval.observability.trace import ObservabilityTrace
from gage_eval.pipeline.steps.auto_eval import AutoEvalStep
from gage_eval.reporting.recorders import InMemoryRecorder


class _FakeMetricResult:
    def __init__(self, metric_id: str) -> None:
        self._metric_id = metric_id

    def to_dict(self):
        return {"score": self._metric_id}


class _FakeMetricInstance:
    def __init__(self, metric_id: str) -> None:
        self.spec = SimpleNamespace(metric_id=metric_id, params={})

    def evaluate(self, context):
        return _FakeMetricResult(self.spec.metric_id)

    def finalize(self):
        return {"metric_id": self.spec.metric_id, "values": {"score": 1.0}}


@pytest.mark.fast
def test_auto_eval_uses_shared_metric_lane() -> None:
    step = AutoEvalStep(metric_specs=())
    step._instances = [_FakeMetricInstance("m1"), _FakeMetricInstance("m2")]
    controller = TaskExecutionController(sample_workers=1, metric_workers=2)
    step.attach_execution_controller(controller)
    trace = ObservabilityTrace(recorder=InMemoryRecorder(run_id="auto-eval-shared"))
    sample = {"id": "s1", "messages": [{"role": "user", "content": [{"type": "text", "text": "hi"}]}]}

    step.execute(
        sample_id="s1",
        sample=sample,
        model_output={},
        judge_output={},
        trace=trace,
        task_id="task-1",
    )

    auto_eval_events = [event for event in trace.events if event["event"] == "auto_eval_sample"]
    controller.shutdown()

    assert auto_eval_events
    assert auto_eval_events[-1]["payload"]["worker_count"] == 2
    assert auto_eval_events[-1]["payload"]["metrics"]["m1"]["score"] == "m1"
    assert auto_eval_events[-1]["payload"]["metrics"]["m2"]["score"] == "m2"


@pytest.mark.fast
def test_auto_eval_falls_back_inline_when_metric_lane_disabled() -> None:
    step = AutoEvalStep(metric_specs=())
    step._instances = [_FakeMetricInstance("m1"), _FakeMetricInstance("m2")]
    controller = TaskExecutionController(sample_workers=1, metric_workers=1)
    step.attach_execution_controller(controller)
    trace = ObservabilityTrace(recorder=InMemoryRecorder(run_id="auto-eval-inline"))
    sample = {"id": "s1", "messages": [{"role": "user", "content": [{"type": "text", "text": "hi"}]}]}

    step.execute(
        sample_id="s1",
        sample=sample,
        model_output={},
        judge_output={},
        trace=trace,
        task_id="task-1",
    )

    controller.shutdown()

    assert any(event["event"] == "metric_lane_fallback_inline" for event in trace.events)


@pytest.mark.fast
def test_auto_eval_persists_samples_jsonl_projection_scalars(tmp_path) -> None:
    cache = EvalCache(base_dir=tmp_path, run_id="projection-scalars")
    step = AutoEvalStep(metric_specs=(), cache_store=cache)
    trace = ObservabilityTrace(recorder=InMemoryRecorder(run_id="projection-scalars"))
    model_output = {
        "answer": "done",
        "runtime_session": {"session_id": "session-1"},
        "agent_eval": {
            "trial_aggregate": {
                "samples_jsonl_projection": {
                    "primary_trial_id": "trial_0001",
                    "status": {"value": "completed", "source_trial_id": "trial_0001"},
                    "resolved": {"value": True, "source_trial_id": "trial_0001"},
                    "score": {"value": 1.0, "source_trial_id": "trial_0001"},
                    "reward": {"value": 1.0, "source_trial_id": "trial_0001"},
                }
            }
        },
    }

    step.persist_sample_artifact(
        sample_id="sample-1",
        sample={"id": "sample-1"},
        model_output=model_output,
        judge_output={"status": "completed", "score": 1.0, "judge_source": "runtime_verifier"},
        trace=trace,
        task_id="task-1",
    )

    [line] = cache.samples_jsonl.read_text(encoding="utf-8").splitlines()
    record = json.loads(line)
    assert record["status"] == "completed"
    assert record["status_source_trial_id"] == "trial_0001"
    assert record["resolved"] is True
    assert record["score"] == 1.0
    assert record["reward"] == 1.0
    assert record["primary_trial_id"] == "trial_0001"
    assert record["predict_result"]["answer"] == "done"
    assert record["auto_eval_result"]["status"] == "completed"
    assert record["auto_eval_result"]["score"] == 1.0
