from __future__ import annotations

import json
from pathlib import Path

import pytest

from gage_eval.evaluation.cache import EvalCache
from gage_eval.external_harness_kits.errors import ExternalHarnessParseError
from gage_eval.external_harness_kits.harbor.trace_translation import HarborATIFTranslator
from gage_eval.external_harness_kits.harbor.results import (
    EXTERNAL_HARNESS_CANCELLED,
    HARBOR_JOB_RESULT_MISSING,
    HARBOR_LAUNCHER_FAILED,
    HARBOR_TRIAL_EXCEPTION,
    HARBOR_VERIFIER_RESULT_MISSING,
    MALFORMED_RESULT_WARNING,
    MISSING_RESULT_WARNING,
    TRIAL_COUNT_MISMATCH_WARNING,
    HarborResultBundle,
    parse_harbor_results,
)
from gage_eval.pipeline.steps.harbor import HarborJobHandle
from gage_eval.external_harness_kits.base import TaskBatchHarnessResult


SPIKE_ROOT = Path("/Users/panke/AI-learning/eval-framework/new/agent-eval/0508/spike/_tmp")
_UNSET = object()


class _Context:
    def __init__(self, tmp_path: Path, *, run_id: str = "harbor-parse") -> None:
        self.task_id = "tb2_one_case"
        self.dataset_id = "terminal_bench_2_0"
        self.cache_store = EvalCache(base_dir=tmp_path, run_id=run_id)


@pytest.mark.fast
def test_real_harbor_fixture_parses_one_sample(tmp_path: Path) -> None:
    workdir = SPIKE_ROOT / "tb2_1case_lmstudio_hostapi_nosource"
    job_name = "gage_tb2_1case_lmstudio_20260510_161501"
    handle = _handle(workdir=workdir, job_name=job_name)

    bundle = parse_harbor_results(
        TaskBatchHarnessResult(adapter_id="harbor", payload={"handle": handle.to_dict()}),
        context=_Context(tmp_path),
        handle=handle,
    )

    assert isinstance(bundle, HarborResultBundle)
    assert len(bundle.samples) == 1
    sample = bundle.samples[0]
    assert sample.sample_id == "gpt2-codegolf"
    assert sample.sample["dataset_source"]["benchmark"] == "harbor"
    assert sample.trial_results[0].status == "completed"
    assert sample.sample["evaluation"]["score"] == 0.0
    assert sample.sample["eval_result"]["external_trial_pass_values"] == [False]


@pytest.mark.fast
def test_trajectory_tool_calls_are_preserved(tmp_path: Path) -> None:
    handle = _synthetic_tree(
        tmp_path,
        trials=[
            _trial_payload(
                task_name="tool-task",
                reward=1.0,
                trajectory=[
                    {"role": "assistant", "content": "checking"},
                    {"type": "tool_call", "tool_name": "bash", "arguments": {"cmd": "pytest"}, "output": "ok"},
                ],
            )
        ],
    )

    bundle = parse_harbor_results(
        _result(handle),
        context=_Context(tmp_path),
        handle=handle,
        trace_translator=HarborATIFTranslator(),
    )

    trajectory = bundle.samples[0].sample["prediction"]["trajectory"]
    assert any(item.get("type") == "tool_call" and item.get("tool_name") == "bash" for item in trajectory)
    predict_result = bundle.samples[0].sample["predict_result"][0]
    assert predict_result["agent_trace"]
    assert any(step.get("trace_role") == "tool" and step.get("name") == "bash" for step in predict_result["agent_trace"])


@pytest.mark.fast
def test_reward_key_override_maps_verifier_reward_to_score(tmp_path: Path) -> None:
    handle = _synthetic_tree(
        tmp_path,
        trials=[
            _trial_payload(
                task_name="score-task",
                reward=0.0,
                rewards={"custom_score": 0.75, "reward": 0.0, "resolved": True},
            )
        ],
    )

    bundle = parse_harbor_results(
        _result(handle, reward_key="custom_score"),
        context=_Context(tmp_path),
        handle=handle,
    )

    trial = bundle.samples[0].trial_results[0]
    assert trial.verifier_result["reward_key"] == "custom_score"
    assert trial.verifier_result["score"] == 0.75
    assert trial.verifier_result["rewards"]["reward"] == 0.0


@pytest.mark.fast
def test_exception_trial_generates_harbor_trial_exception_record(tmp_path: Path) -> None:
    handle = _synthetic_tree(
        tmp_path,
        trials=[
            _trial_payload(
                task_name="exception-task",
                verifier_result=None,
                exception_info={"exception_type": "RuntimeError", "exception_message": "boom"},
            )
        ],
    )

    bundle = parse_harbor_results(_result(handle), context=_Context(tmp_path), handle=handle)

    trial = bundle.samples[0].trial_results[0]
    assert trial.status == "failed"
    assert trial.failure["failure_code"] == HARBOR_TRIAL_EXCEPTION
    assert bundle.samples[0].sample["trials"][0]["passed"] is None
    assert (tmp_path / "harbor-parse/artifacts/tb2_one_case/exception-task/trials/trial_0001/infra/trial_result.json").is_file()


@pytest.mark.fast
def test_verifier_result_missing_records_harbor_verifier_result_missing(tmp_path: Path) -> None:
    handle = _synthetic_tree(
        tmp_path,
        trials=[_trial_payload(task_name="missing-verifier", verifier_result=None)],
    )

    bundle = parse_harbor_results(_result(handle), context=_Context(tmp_path), handle=handle)

    trial = bundle.samples[0].trial_results[0]
    assert trial.status == "failed"
    assert trial.failure["failure_code"] == HARBOR_VERIFIER_RESULT_MISSING


@pytest.mark.fast
def test_parser_calls_aggregate_trial_results(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    from gage_eval.external_harness_kits.harbor import results as harbor_results

    calls = []

    class _Aggregate:
        aggregation = "single"
        samples_jsonl_projection = {"primary_trial_id": "trial_0001"}
        pass_rate = 1.0
        score_mean = 1.0
        metric_projection = {}

        def to_dict(self):
            return {"trial_count": 1}

    def fake_aggregate(trials, *, aggregation=None):
        calls.append((list(trials), aggregation))
        return _Aggregate()

    monkeypatch.setattr(harbor_results.trial_aggregation, "aggregate_trial_results", fake_aggregate)
    handle = _synthetic_tree(tmp_path, trials=[_trial_payload(task_name="aggregate-task", reward=1.0)])

    parse_harbor_results(_result(handle), context=_Context(tmp_path), handle=handle)

    assert len(calls) == 1
    assert calls[0][1] == "single"


@pytest.mark.fast
def test_launcher_failed_raises_harbor_launcher_failed(tmp_path: Path) -> None:
    handle = _synthetic_tree(tmp_path, trials=[_trial_payload()])

    with pytest.raises(ExternalHarnessParseError) as exc_info:
        parse_harbor_results(
            TaskBatchHarnessResult(
                adapter_id="harbor",
                payload={"handle": handle.to_dict(), "launcher_result": {"exit_code": 2, "launcher_error": "boom"}},
            ),
            context=_Context(tmp_path),
            handle=handle,
        )

    assert exc_info.value.code == HARBOR_LAUNCHER_FAILED


@pytest.mark.fast
def test_job_result_missing_without_trials_raises_harbor_job_result_missing(tmp_path: Path) -> None:
    handle = _synthetic_tree(tmp_path, trials=[], write_job_result=False)

    with pytest.raises(ExternalHarnessParseError) as exc_info:
        parse_harbor_results(_result(handle), context=_Context(tmp_path), handle=handle)

    assert exc_info.value.code == HARBOR_JOB_RESULT_MISSING


@pytest.mark.fast
def test_cancelled_marker_without_trials_imports_aborted_sample(tmp_path: Path) -> None:
    handle = _synthetic_tree(tmp_path, trials=[], write_job_result=False)
    task_dir = tmp_path / "sample-task"
    handle = HarborJobHandle.from_dict(
        {
            **handle.to_dict(),
            "invocation_metadata": {
                **handle.invocation_metadata,
                "job_config": {"tasks": [{"path": str(task_dir), "source": "terminal-bench"}]},
            },
        }
    )
    (handle.workdir / "cancelled.json").write_text(
        json.dumps(
            {
                "status": "cancelled",
                "reason": "adapter_shutdown",
                "job_name": handle.job_name,
            }
        ),
        encoding="utf-8",
    )

    bundle = parse_harbor_results(_result(handle), context=_Context(tmp_path), handle=handle)

    sample = bundle.samples[0]
    trial = sample.trial_results[0]
    assert sample.sample_id == "sample-task"
    assert trial.status == "aborted"
    assert trial.failure["failure_code"] == EXTERNAL_HARNESS_CANCELLED
    assert sample.sample["eval_result"]["status"]["value"] == "aborted"
    assert sample.sample["eval_result"]["external_harness_cancelled"]["reason"] == "adapter_shutdown"
    assert bundle.task_metrics["aborted_count"] == 1


@pytest.mark.fast
def test_cancelled_marker_does_not_override_completed_trial_results(tmp_path: Path) -> None:
    handle = _synthetic_tree(tmp_path, trials=[_trial_payload(task_name="completed-after-marker", reward=1.0)])
    (handle.workdir / "cancelled.json").write_text(
        json.dumps(
            {
                "status": "cancelled",
                "reason": "stale_marker",
                "job_name": handle.job_name,
            }
        ),
        encoding="utf-8",
    )

    bundle = parse_harbor_results(_result(handle), context=_Context(tmp_path), handle=handle)

    sample = bundle.samples[0]
    trial = sample.trial_results[0]
    assert trial.status == "completed"
    assert trial.failure is None
    assert sample.sample["eval_result"]["harbor_resolve_rate"] == 1.0
    assert "external_harness_cancelled" not in sample.sample["eval_result"]


@pytest.mark.fast
def test_job_result_missing_with_trial_continues_partial_parse(tmp_path: Path) -> None:
    handle = _synthetic_tree(tmp_path, trials=[_trial_payload(task_name="partial")], write_job_result=False)

    bundle = parse_harbor_results(_result(handle), context=_Context(tmp_path), handle=handle)

    assert len(bundle.samples) == 1
    assert any(warning.code == "external_harness.parse.job_result_missing_partial" for warning in bundle.warnings)


@pytest.mark.fast
def test_trial_result_missing_warns_external_harness_parse_missing_result(tmp_path: Path) -> None:
    handle = _synthetic_tree(tmp_path, trials=[_trial_payload(task_name="valid")])
    (handle.job_dir / "missing-result__abc").mkdir()

    bundle = parse_harbor_results(_result(handle), context=_Context(tmp_path), handle=handle)

    assert len(bundle.samples) == 1
    assert any(warning.code == MISSING_RESULT_WARNING for warning in bundle.warnings)


@pytest.mark.fast
def test_malformed_result_constructs_failed_trial_when_task_locatable(tmp_path: Path) -> None:
    handle = _synthetic_tree(tmp_path, trials=[])
    trial_dir = handle.job_dir / "bad-task__abc"
    trial_dir.mkdir()
    (trial_dir / "result.json").write_text("[not valid json", encoding="utf-8")

    bundle = parse_harbor_results(_result(handle), context=_Context(tmp_path), handle=handle)

    trial = bundle.samples[0].trial_results[0]
    assert bundle.samples[0].sample_id == "bad-task"
    assert trial.status == "failed"
    assert trial.failure["failure_code"] == MALFORMED_RESULT_WARNING
    assert any(warning.code == MALFORMED_RESULT_WARNING for warning in bundle.warnings)


@pytest.mark.fast
def test_all_trials_parse_failed_raises_external_harness_parse_malformed_output(tmp_path: Path) -> None:
    handle = _synthetic_tree(tmp_path, trials=[])
    (handle.job_dir / "missing-only__abc").mkdir()

    with pytest.raises(ExternalHarnessParseError) as exc_info:
        parse_harbor_results(_result(handle), context=_Context(tmp_path), handle=handle)

    assert exc_info.value.code == MALFORMED_RESULT_WARNING


@pytest.mark.fast
def test_trial_count_mismatch_warns_but_aggregates_parsed_trials(tmp_path: Path) -> None:
    handle = _synthetic_tree(tmp_path, trials=[_trial_payload(task_name="mismatch", reward=1.0)])

    bundle = parse_harbor_results(
        _result(handle, trial_policy={"trials": 2, "aggregation": "single"}),
        context=_Context(tmp_path),
        handle=handle,
    )

    assert bundle.samples[0].aggregate.trial_count == 1
    assert any(warning.code == TRIAL_COUNT_MISMATCH_WARNING for warning in bundle.warnings)


@pytest.mark.fast
def test_job_result_without_trial_results_uses_trial_directories(tmp_path: Path) -> None:
    handle = _synthetic_tree(
        tmp_path,
        trials=[_trial_payload(task_name="no-job-trials", reward=1.0)],
        job_result={"id": "job-id", "n_total_trials": 1, "stats": {}},
    )

    bundle = parse_harbor_results(_result(handle), context=_Context(tmp_path), handle=handle)

    assert bundle.samples[0].sample_id == "no-job-trials"
    assert bundle.samples[0].trial_results[0].verifier_result["score"] == 1.0


@pytest.mark.fast
def test_single_sample_parse_baseline_under_100ms(tmp_path: Path) -> None:
    handle = _synthetic_tree(tmp_path, trials=[_trial_payload(task_name="perf", reward=1.0)])

    bundle = parse_harbor_results(_result(handle), context=_Context(tmp_path), handle=handle)

    assert bundle.elapsed_s < 0.1


def _handle(*, workdir: Path, job_name: str) -> HarborJobHandle:
    return HarborJobHandle(
        job_name=job_name,
        jobs_dir=workdir / "jobs",
        job_dir=workdir / "jobs" / job_name,
        job_config_path=workdir / "harbor-job.json",
        launcher_result_path=workdir / "launcher_result.json",
        workdir=workdir,
        environment={"type": "docker"},
        invocation_metadata={"launcher_mode": "python_subprocess", "expected_total_trials": 1},
    )


def _synthetic_tree(
    tmp_path: Path,
    *,
    trials: list[dict],
    write_job_result: bool = True,
    job_result: dict | None = None,
) -> HarborJobHandle:
    workdir = tmp_path / "harbor-work"
    job_name = "gage_synthetic"
    job_dir = workdir / "jobs" / job_name
    job_dir.mkdir(parents=True)
    if write_job_result:
        (job_dir / "result.json").write_text(
            json.dumps(
                job_result
                or {
                    "id": "job-id",
                    "n_total_trials": len(trials),
                    "stats": {"n_completed_trials": len(trials), "cost_usd": None},
                }
            ),
            encoding="utf-8",
        )
    handle = _handle(workdir=workdir, job_name=job_name)
    for index, payload in enumerate(trials, start=1):
        trial_name = payload.get("trial_name") or f"{payload.get('task_name', 'sample')}__{index:04d}"
        trial_dir = job_dir / trial_name
        trial_dir.mkdir()
        trajectory = payload.pop("_trajectory", None)
        payload = {**payload, "trial_name": trial_name, "trial_uri": trial_dir.as_uri()}
        (trial_dir / "result.json").write_text(json.dumps(payload), encoding="utf-8")
        if trajectory is not None:
            agent_dir = trial_dir / "agent"
            agent_dir.mkdir()
            (agent_dir / "trajectory.json").write_text(json.dumps(trajectory), encoding="utf-8")
    (workdir / "launcher_result.json").write_text(json.dumps({"exit_code": 0}), encoding="utf-8")
    return handle


def _trial_payload(
    *,
    task_name: str = "sample-task",
    reward: float | None = 1.0,
    rewards: dict | None = None,
    verifier_result: dict | None | object = _UNSET,
    exception_info: dict | None = None,
    trajectory: list[dict] | None = None,
) -> dict:
    task_dir = Path("/Users/panke/.cache/harbor/tasks/CGqXnFmcaVcQCTbCZMpg2T/gpt2-codegolf")
    if verifier_result is _UNSET:
        verifier_result = {"rewards": rewards or {"reward": reward, "resolved": bool(reward and reward > 0)}}
    return {
        "id": f"{task_name}-trial",
        "task_name": task_name,
        "task_id": {"path": str(task_dir)},
        "source": "terminal-bench",
        "task_checksum": "abc123",
        "config": {
            "task": {"path": str(task_dir), "source": "terminal-bench"},
            "agent": {"model_name": "lm_studio/qwen/qwen3.5-9b"},
            "environment": {"type": "docker"},
        },
        "agent_info": {"name": "terminus-2", "model_info": {"name": "qwen/qwen3.5-9b", "provider": "lm_studio"}},
        "agent_result": {
            "n_input_tokens": 1,
            "n_cache_tokens": 0,
            "n_output_tokens": 2,
            "cost_usd": None,
            "metadata": {"n_episodes": 1},
            "final_answer": "done",
        },
        "verifier_result": verifier_result,
        "exception_info": exception_info,
        "_trajectory": trajectory,
    }


def _result(
    handle: HarborJobHandle,
    *,
    reward_key: str | None = None,
    trial_policy: dict | None = None,
) -> TaskBatchHarnessResult:
    payload = {"handle": handle.to_dict()}
    if reward_key:
        payload["reward_key"] = reward_key
    if trial_policy:
        payload["trial_policy"] = trial_policy
    return TaskBatchHarnessResult(adapter_id="harbor", payload=payload)
