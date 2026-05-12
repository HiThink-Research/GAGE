from __future__ import annotations

import json
from pathlib import Path
import time
from typing import Any

import pytest

from gage_eval.evaluation.cache import EvalCache
from gage_eval.external_harness_kits.harbor.observability import (
    DEFAULT_PROGRESS_POLL_INTERVAL_S,
    resolve_progress_poll_interval_s,
)
from gage_eval.external_harness_kits.harbor.results import parse_harbor_results
from gage_eval.pipeline.steps.harbor import HarborJobHandle
from gage_eval.external_harness_kits.base import TaskBatchHarnessResult


FIXTURE_DIR = Path("tests/_support/external_harness_kits/harbor_tb2_1case")


class _Context:
    def __init__(self, tmp_path: Path, *, run_id: str = "harbor-perf") -> None:
        self.task_id = "tb2_one_case"
        self.dataset_id = "terminal_bench_2_0"
        self.cache_store = EvalCache(base_dir=tmp_path, run_id=run_id)


@pytest.mark.performance
def test_synthetic_50_by_3_fixture_parse_baseline_under_5s(tmp_path: Path) -> None:
    handle = _synthetic_tree(tmp_path, sample_count=50, trials_per_sample=3)

    started = time.perf_counter()
    bundle = parse_harbor_results(
        _result(handle, trial_policy={"trials": 3, "aggregation": "all"}),
        context=_Context(tmp_path),
        handle=handle,
        expected_trials=3,
    )
    elapsed_s = time.perf_counter() - started

    assert len(bundle.samples) == 50
    assert sum(len(sample.trial_results) for sample in bundle.samples) == 150
    assert elapsed_s <= 5.0


@pytest.mark.performance
def test_single_tb2_fixture_parse_baseline_under_100ms(tmp_path: Path) -> None:
    handle = _tb2_one_case_tree(tmp_path)

    started = time.perf_counter()
    bundle = parse_harbor_results(_result(handle), context=_Context(tmp_path), handle=handle)
    elapsed_s = time.perf_counter() - started

    assert len(bundle.samples) == 1
    assert elapsed_s < 0.1


@pytest.mark.performance
def test_progress_poll_interval_never_allows_tight_loop() -> None:
    assert DEFAULT_PROGRESS_POLL_INTERVAL_S >= 1.0
    assert resolve_progress_poll_interval_s(None) == DEFAULT_PROGRESS_POLL_INTERVAL_S
    assert resolve_progress_poll_interval_s(0) == 1.0
    assert resolve_progress_poll_interval_s(0.25) == 1.0
    assert resolve_progress_poll_interval_s(2.5) == 2.5


def _tb2_one_case_tree(tmp_path: Path) -> HarborJobHandle:
    workdir = tmp_path / "tb2-fixture"
    job_name = "gage_tb2_one_case_fixture"
    job_dir = workdir / "jobs" / job_name
    trial_payload = json.loads((FIXTURE_DIR / "trial_result.json").read_text(encoding="utf-8"))
    trial_name = str(trial_payload["trial_name"])
    trial_dir = job_dir / trial_name
    trial_dir.mkdir(parents=True)
    trial_payload["trial_uri"] = trial_dir.as_uri()
    (trial_dir / "result.json").write_text(json.dumps(trial_payload), encoding="utf-8")
    (job_dir / "result.json").write_text(
        (FIXTURE_DIR / "job_result.json").read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    return _handle(workdir=workdir, job_name=job_name)


def _synthetic_tree(tmp_path: Path, *, sample_count: int, trials_per_sample: int) -> HarborJobHandle:
    workdir = tmp_path / "synthetic"
    job_name = "gage_perf_synthetic"
    job_dir = workdir / "jobs" / job_name
    job_dir.mkdir(parents=True)
    task_dir = tmp_path / "task"
    task_dir.mkdir()
    (task_dir / "instruction.md").write_text("Return success.\n", encoding="utf-8")
    (task_dir / "task.toml").write_text("name = 'perf'\n", encoding="utf-8")
    for sample_index in range(sample_count):
        task_name = f"perf-task-{sample_index:03d}"
        for trial_index in range(trials_per_sample):
            trial_name = f"{task_name}__{trial_index:04d}"
            trial_dir = job_dir / trial_name
            trial_dir.mkdir()
            (trial_dir / "result.json").write_text(
                json.dumps(_trial_payload(task_dir=task_dir, task_name=task_name, trial_name=trial_name)),
                encoding="utf-8",
            )
    total_trials = sample_count * trials_per_sample
    (job_dir / "result.json").write_text(
        json.dumps(
            {
                "id": "perf-job",
                "n_total_trials": total_trials,
                "stats": {
                    "n_completed_trials": total_trials,
                    "n_errored_trials": 0,
                    "n_cancelled_trials": 0,
                    "n_running_trials": 0,
                    "cost_usd": None,
                },
            }
        ),
        encoding="utf-8",
    )
    return _handle(workdir=workdir, job_name=job_name)


def _trial_payload(*, task_dir: Path, task_name: str, trial_name: str) -> dict[str, Any]:
    return {
        "id": f"{trial_name}-id",
        "task_name": task_name,
        "trial_name": trial_name,
        "trial_uri": (task_dir.parent / trial_name).as_uri(),
        "task_id": {"path": str(task_dir)},
        "source": "terminal-bench",
        "task_checksum": "perf-checksum",
        "config": {
            "task": {"path": str(task_dir), "source": "terminal-bench"},
            "agent": {"model_name": "lm_studio/qwen/qwen3.5-9b"},
            "environment": {"type": "docker"},
        },
        "agent_info": {"name": "terminus-2", "model_info": {"name": "qwen/qwen3.5-9b"}},
        "agent_result": {
            "n_input_tokens": 1,
            "n_cache_tokens": 0,
            "n_output_tokens": 1,
            "cost_usd": None,
            "metadata": {"n_episodes": 1},
            "final_answer": "done",
        },
        "verifier_result": {"rewards": {"reward": 1.0, "resolved": True}},
        "exception_info": None,
    }


def _handle(*, workdir: Path, job_name: str) -> HarborJobHandle:
    (workdir / "launcher_result.json").write_text(json.dumps({"exit_code": 0}), encoding="utf-8")
    return HarborJobHandle(
        job_name=job_name,
        jobs_dir=workdir / "jobs",
        job_dir=workdir / "jobs" / job_name,
        job_config_path=workdir / "harbor-job.json",
        launcher_result_path=workdir / "launcher_result.json",
        workdir=workdir,
        environment={"type": "docker"},
        invocation_metadata={"launcher_mode": "python_subprocess"},
    )


def _result(handle: HarborJobHandle, *, trial_policy: dict[str, Any] | None = None) -> TaskBatchHarnessResult:
    payload: dict[str, Any] = {"handle": handle.to_dict()}
    if trial_policy:
        payload["trial_policy"] = trial_policy
    return TaskBatchHarnessResult(adapter_id="harbor", payload=payload)
