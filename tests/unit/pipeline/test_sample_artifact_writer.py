from __future__ import annotations

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from gage_eval.agent_runtime.artifacts import RuntimeArtifactSink
from gage_eval.agent_runtime.trace_schema import TrialResult
from gage_eval.agent_runtime.trials import aggregate_trial_results
from gage_eval.assets.datasets.sample import SCHEMA_VERSION
from gage_eval.evaluation.cache import EvalCache
from gage_eval.pipeline import sample_artifact_writer as writer_module
from gage_eval.pipeline.sample_artifact_writer import SampleArtifactWriter


@pytest.mark.fast
def test_writer_outputs_samples_jsonl_sample_json_and_sample_record_tree(tmp_path: Path) -> None:
    run_id = "run-writer"
    task_id = "task-1"
    sample_id = "sample-1"
    sink = RuntimeArtifactSink(base_dir=str(tmp_path))
    cache = EvalCache(base_dir=tmp_path, run_id=run_id)
    trial = _trial_result(sink=sink, run_id=run_id, task_id=task_id, sample_id=sample_id)
    aggregate = aggregate_trial_results([trial], aggregation="single")
    writer = SampleArtifactWriter(cache_store=cache, artifact_sink=sink, dut_id="harbor")

    written = writer.write_sample_record(
        {
            "task_id": task_id,
            "sample_id": sample_id,
            "sample": _sample(sample_id),
            "trial_results": [trial],
            "aggregate": aggregate,
            "infra_artifacts": [
                {"name": "provider_invocation.json", "content": {"job_name": "gage_job"}},
                {"name": "provider_job_result.json", "content": {"id": "job-id"}},
            ],
            "environment_descriptor": {"external_harness": "harbor"},
        }
    )

    run_dir = tmp_path / run_id
    assert (run_dir / "samples.jsonl").is_file()
    assert Path(written.sample_cache_path).is_file()
    assert (run_dir / "samples/task_task-1/sample-1.json").is_file()
    assert (run_dir / "artifacts/task-1/sample-1/infra/sample_record.json").is_file()
    assert (run_dir / "artifacts/task-1/sample-1/infra/trial_aggregate.json").is_file()
    assert (run_dir / "artifacts/task-1/sample-1/infra/provider_invocation.json").is_file()
    assert (run_dir / "artifacts/task-1/sample-1/infra/provider_job_result.json").is_file()
    assert (run_dir / "artifacts/task-1/sample-1/trials/trial_0001/infra/trial_result.json").is_file()
    assert (run_dir / "artifacts/task-1/sample-1/trials/trial_0001/infra/harbor_raw_result.json").is_file()
    assert (run_dir / "artifacts/task-1/sample-1/trials/trial_0001/verifier/reward.json").is_file()
    sample_record = json.loads(
        (run_dir / "artifacts/task-1/sample-1/infra/sample_record.json").read_text(encoding="utf-8")
    )
    assert sample_record["dut_id"] == "harbor"
    assert sample_record["trial_results"][0]["trial_id"] == "trial_0001"
    assert not list(run_dir.rglob("harbor_artifacts"))


@pytest.mark.fast
def test_sample_record_model_validate_is_called_before_write(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    run_id = "run-validate"
    task_id = "task-1"
    sample_id = "sample-1"
    sink = RuntimeArtifactSink(base_dir=str(tmp_path))
    cache = EvalCache(base_dir=tmp_path, run_id=run_id)
    trial = _trial_result(sink=sink, run_id=run_id, task_id=task_id, sample_id=sample_id)
    aggregate = aggregate_trial_results([trial], aggregation="single")
    calls = []
    original = writer_module.SampleRecord.model_validate

    def spy(payload, *args, **kwargs):
        calls.append(payload)
        return original(payload, *args, **kwargs)

    monkeypatch.setattr(writer_module.SampleRecord, "model_validate", spy)
    writer = SampleArtifactWriter(cache_store=cache, artifact_sink=sink)

    writer.write_sample_record(
        {
            "task_id": task_id,
            "sample_id": sample_id,
            "sample": _sample(sample_id),
            "trial_results": [trial],
            "aggregate": aggregate,
        }
    )

    assert calls
    assert calls[0]["sample_id"] == sample_id


@pytest.mark.fast
def test_invalid_sample_record_fails_validation_before_sample_record_write(tmp_path: Path) -> None:
    cache = EvalCache(base_dir=tmp_path, run_id="run-invalid")
    writer = SampleArtifactWriter(cache_store=cache)

    with pytest.raises(ValidationError):
        writer.write_sample_record(
            {
                "task_id": "task-1",
                "sample_id": "sample-1",
                "sample": _sample("sample-1"),
                "trial_results": [{"trial_id": "trial_0001"}],
                "aggregate": {"trial_count": 1},
            }
        )

    assert not (tmp_path / "run-invalid/artifacts/task-1/sample-1/infra/sample_record.json").exists()


def _trial_result(*, sink: RuntimeArtifactSink, run_id: str, task_id: str, sample_id: str) -> TrialResult:
    raw_ref = sink.write_artifact(
        run_id=run_id,
        task_id=task_id,
        sample_id=sample_id,
        trial_id="trial_0001",
        owner="infra",
        name="harbor_raw_result.json",
        content={"trial_name": "sample-1__0001"},
        mime_type="application/json",
    )
    reward_ref = sink.write_artifact(
        run_id=run_id,
        task_id=task_id,
        sample_id=sample_id,
        trial_id="trial_0001",
        owner="verifier",
        name="reward.json",
        content={"rewards": {"reward": 1.0}},
        mime_type="application/json",
    )
    trace_ref = sink.append_trace_event(
        run_id=run_id,
        task_id=task_id,
        sample_id=sample_id,
        trial_id="trial_0001",
        actor="verifier",
        event_type="verifier.result",
        payload={"metric": {"score": 1.0, "passed": True}},
        artifact_refs=[reward_ref],
    )
    trial = TrialResult.model_validate(
        {
            "trial_id": "trial_0001",
            "status": "completed",
            "scheduler_result": {"status": "completed"},
            "verifier_result": {"score": 1.0, "reward": 1.0, "passed": True, "resolved": True},
            "environment_descriptor": {"external_harness": "harbor"},
            "artifact_refs": [raw_ref.model_dump(mode="python"), reward_ref.model_dump(mode="python")],
            "trace_ref": trace_ref.model_dump(mode="python"),
            "failure": None,
        }
    )
    record_ref = sink.write_trial_record(
        run_id=run_id,
        task_id=task_id,
        sample_id=sample_id,
        trial_id="trial_0001",
        trial_result=trial,
    )
    trial.artifact_refs.append(record_ref)
    return trial


def _sample(sample_id: str) -> dict:
    return {
        "schema_version": SCHEMA_VERSION,
        "id": sample_id,
        "messages": [{"role": "user", "content": [{"type": "text", "text": "do it"}]}],
        "predict_result": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": [{"type": "text", "text": "done"}]},
            }
        ],
        "eval_result": {"score": 1.0},
    }
