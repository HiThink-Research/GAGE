from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from gage_eval.external_harness_kits.harbor.observability import (
    ExternalHarnessProgress,
    derive_harbor_progress,
    emit_external_harness_job_completed,
    emit_external_harness_job_submitted,
    emit_external_harness_progress,
)
from gage_eval.observability.progress_sink import ProgressSink


FIXTURE_DIR = (
    Path(__file__).resolve().parents[2]
    / "_support"
    / "external_harness_kits"
    / "harbor_tb2_1case"
)


class _Trace:
    def __init__(self) -> None:
        self.events: list[tuple[str, dict[str, Any], str | None]] = []

    def emit(self, event: str, payload: dict[str, Any], sample_id: str | None = None) -> None:
        self.events.append((event, payload, sample_id))


def _fixture(name: str) -> dict[str, Any]:
    return json.loads((FIXTURE_DIR / name).read_text(encoding="utf-8"))


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


@pytest.mark.fast
def test_harbor_observability_imports_progress_sink_from_observability_namespace() -> None:
    from gage_eval.external_harness_kits.harbor import observability as harbor_observability

    assert harbor_observability.ProgressSink is ProgressSink


@pytest.mark.fast
def test_empty_jobs_dir_derives_pending_progress_without_reading_state_json(tmp_path: Path) -> None:
    jobs_dir = tmp_path / "jobs"

    progress = derive_harbor_progress(
        jobs_dir=jobs_dir,
        job_name="gage_tb2_one_case",
        task_id="tb2_one_case",
        total_trials=1,
    )

    assert progress == ExternalHarnessProgress(
        task_id="tb2_one_case",
        total_trials=1,
        completed_trials=0,
        failed_trials=0,
        running_trials=0,
        current_sample_ids=[],
    )
    assert not (jobs_dir / "gage_tb2_one_case" / "state.json").exists()


@pytest.mark.fast
def test_trial_result_derives_completed_progress_from_result_files(tmp_path: Path) -> None:
    jobs_dir = tmp_path / "jobs"
    job_name = "gage_tb2_one_case"
    _write_json(jobs_dir / job_name / "result.json", _fixture("job_result.json"))
    _write_json(
        jobs_dir / job_name / "gpt2-codegolf__sLkuvPz" / "result.json",
        _fixture("trial_result.json"),
    )

    progress = derive_harbor_progress(
        jobs_dir=jobs_dir,
        job_name=job_name,
        task_id="tb2_one_case",
        total_trials=1,
    )

    assert progress == ExternalHarnessProgress(
        task_id="tb2_one_case",
        total_trials=1,
        completed_trials=1,
        failed_trials=0,
        running_trials=0,
        current_sample_ids=["gpt2-codegolf"],
        phase="completed",
    )


@pytest.mark.fast
def test_launcher_error_derives_failed_progress(tmp_path: Path) -> None:
    launcher_result = {
        "exit_code": 1,
        "job_name": "gage_tb2_one_case",
        "jobs_dir": str(tmp_path / "jobs"),
        "launcher_error": {
            "type": "RuntimeError",
            "message": "GAGE could not complete the external harness job",
            "traceback_ref": "launcher.traceback.log",
        },
    }
    launcher_result_path = tmp_path / "launcher_result.json"
    _write_json(launcher_result_path, launcher_result)

    progress = derive_harbor_progress(
        jobs_dir=tmp_path / "jobs",
        job_name="gage_tb2_one_case",
        task_id="tb2_one_case",
        total_trials=1,
        launcher_result_path=launcher_result_path,
    )

    assert progress.failed_trials == 1
    assert progress.completed_trials == 0
    assert progress.current_sample_ids == []


@pytest.mark.fast
def test_failed_trial_result_counts_as_failed_progress(tmp_path: Path) -> None:
    jobs_dir = tmp_path / "jobs"
    job_name = "gage_tb2_one_case"
    _write_json(
        jobs_dir / job_name / "gpt2-codegolf__synthetic_failure" / "result.json",
        _fixture("trial_with_exception_info.json"),
    )

    progress = derive_harbor_progress(
        jobs_dir=jobs_dir,
        job_name=job_name,
        task_id="tb2_one_case",
        total_trials=1,
    )

    assert progress.failed_trials == 1
    assert progress.completed_trials == 0
    assert progress.current_sample_ids == ["gpt2-codegolf"]


@pytest.mark.fast
def test_trace_helpers_emit_exact_external_harness_payloads() -> None:
    trace = _Trace()
    progress = ExternalHarnessProgress(
        task_id="tb2_one_case",
        total_trials=1,
        completed_trials=1,
        failed_trials=0,
        running_trials=0,
        current_sample_ids=["gpt2-codegolf"],
    )

    emit_external_harness_job_submitted(
        trace,
        kit_id="harbor_tb2",
        job_name="gage_tb2_one_case",
        total_trials=1,
        dataset_ref="terminal-bench://tb2_one_case",
    )
    emit_external_harness_progress(
        trace,
        job_name="gage_tb2_one_case",
        progress=progress,
        phase="completed",
        elapsed_s=12.5,
    )
    emit_external_harness_job_completed(
        trace,
        job_name="gage_tb2_one_case",
        harbor_job_uuid="ed934e6f-2aec-46f3-8fed-0ec0974b93f2",
        exit_code=0,
        progress=progress,
    )

    assert trace.events == [
        (
            "external_harness_job_submitted",
            {
                "kit_id": "harbor_tb2",
                "job_name": "gage_tb2_one_case",
                "total_trials": 1,
                "dataset_ref": "terminal-bench://tb2_one_case",
            },
            None,
        ),
        (
            "external_harness_progress",
            {
                "job_name": "gage_tb2_one_case",
                "completed": 1,
                "total": 1,
                "phase": "completed",
                "elapsed_s": 12.5,
            },
            None,
        ),
        (
            "external_harness_job_completed",
            {
                "job_name": "gage_tb2_one_case",
                "harbor_job_uuid": "ed934e6f-2aec-46f3-8fed-0ec0974b93f2",
                "exit_code": 0,
                "total_trials": 1,
                "completed_trials": 1,
                "failed_trials": 0,
            },
            None,
        ),
    ]
