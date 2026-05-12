"""Fake Harbor launcher used by mocked external harness integration tests."""

from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any, Mapping

from gage_eval.external_harness_kits.harbor.launcher import (
    LauncherSubprocessResult,
    RESULT_SCHEMA_VERSION,
    STDERR_REF,
    STDOUT_REF,
)


def run_fake_launcher(
    *,
    config_path: Path | str,
    result_file: Path | str,
    timeout_s: float | None,
    environ: Mapping[str, str] | None = None,
    python: str | None = None,
    workdir: Path | str | None = None,
    live_log: bool = True,
    job_log_path: Path | str | None = None,
) -> LauncherSubprocessResult:
    del timeout_s, environ, live_log, job_log_path
    config_path = Path(config_path)
    result_file = Path(result_file)
    workdir_path = Path(workdir) if workdir is not None else result_file.parent
    launcher_input = json.loads(config_path.read_text(encoding="utf-8"))
    job_config = dict(launcher_input.get("job_config") or launcher_input)
    job_name = str(job_config.get("job_name") or launcher_input.get("job_name") or "gage_fake_harbor")
    jobs_dir = Path(str(job_config.get("jobs_dir") or launcher_input.get("jobs_dir") or workdir_path / "jobs"))
    job_dir = jobs_dir / job_name
    trial_dir = job_dir / "gpt2-codegolf__mocked"
    task_path = _task_path(job_config)
    started_at = _now()

    trial_dir.mkdir(parents=True, exist_ok=True)
    (workdir_path / STDOUT_REF).parent.mkdir(parents=True, exist_ok=True)
    (workdir_path / STDOUT_REF).write_text("fake harbor launcher stdout\n", encoding="utf-8")
    (workdir_path / STDERR_REF).write_text("", encoding="utf-8")
    (job_dir / "result.json").write_text(
        json.dumps(
            {
                "id": "fake-harbor-job",
                "n_total_trials": 1,
                "started_at": started_at,
                "updated_at": _now(),
                "finished_at": _now(),
                "stats": {
                    "n_completed_trials": 1,
                    "n_errored_trials": 0,
                    "n_running_trials": 0,
                    "n_pending_trials": 0,
                    "n_cancelled_trials": 0,
                    "cost_usd": None,
                },
            }
        ),
        encoding="utf-8",
    )
    (trial_dir / "agent").mkdir(exist_ok=True)
    (trial_dir / "agent" / "trajectory.json").write_text(
        json.dumps(
            [
                {"role": "assistant", "content": "I will inspect the task."},
                {
                    "type": "tool_call",
                    "tool_name": "bash",
                    "arguments": {"cmd": "python - <<'PY'\nprint('ok')\nPY"},
                    "output": "ok\n",
                },
            ]
        ),
        encoding="utf-8",
    )
    (trial_dir / "verifier").mkdir(exist_ok=True)
    (trial_dir / "verifier" / "reward.json").write_text(
        json.dumps({"rewards": {"reward": 1.0, "resolved": True}}),
        encoding="utf-8",
    )
    (trial_dir / "trial.log").write_text("fake trial log\n", encoding="utf-8")
    trial_result = {
        "id": "fake-trial",
        "task_name": "gpt2-codegolf",
        "trial_name": "gpt2-codegolf__mocked",
        "trial_uri": trial_dir.as_uri(),
        "task_id": {"path": str(task_path)},
        "source": "terminal-bench",
        "task_checksum": "fake-checksum",
        "config": {
            "task": {"path": str(task_path), "source": "terminal-bench"},
            "trial_name": "gpt2-codegolf__mocked",
            "agent": _first(job_config.get("agents")) or {},
            "environment": job_config.get("environment") or {},
            "job_id": "fake-harbor-job",
        },
        "agent_info": {
            "name": "terminus-2",
            "version": "fake",
            "model_info": {"name": "qwen/qwen3.5-9b", "provider": "lm_studio"},
        },
        "agent_result": {
            "n_input_tokens": 10,
            "n_cache_tokens": 0,
            "n_output_tokens": 5,
            "cost_usd": None,
            "metadata": {"n_episodes": 1},
            "final_answer": "mocked answer",
        },
        "verifier_result": {"rewards": {"reward": 1.0, "resolved": True}},
        "exception_info": None,
        "started_at": started_at,
        "finished_at": _now(),
    }
    (trial_dir / "result.json").write_text(json.dumps(trial_result), encoding="utf-8")
    result_file.parent.mkdir(parents=True, exist_ok=True)
    result_file.write_text(
        json.dumps(
            {
                "schema_version": RESULT_SCHEMA_VERSION,
                "exit_code": 0,
                "started_at": started_at,
                "finished_at": _now(),
                "pid": 0,
                "python": python or "fake-python",
                "config_path": str(config_path),
                "job_name": job_name,
                "jobs_dir": str(jobs_dir),
                "job_dir": str(job_dir),
                "harbor_job_result": {"id": "fake-harbor-job"},
                "launcher_error": None,
                "stdout_ref": STDOUT_REF,
                "stderr_ref": STDERR_REF,
            }
        ),
        encoding="utf-8",
    )
    return LauncherSubprocessResult(
        argv=[python or "fake-python", "-m", "fake_harbor_launcher"],
        exit_code=0,
        timed_out=False,
        result_file=result_file,
        stdout_path=workdir_path / STDOUT_REF,
        stderr_path=workdir_path / STDERR_REF,
    )


def _task_path(job_config: Mapping[str, Any]) -> Path:
    tasks = job_config.get("tasks")
    if isinstance(tasks, list) and tasks:
        path = (tasks[0] or {}).get("path") if isinstance(tasks[0], Mapping) else None
        if path:
            return Path(str(path))
    return Path("tests/data/external_harness_kits/terminal_bench/gpt2-codegolf")


def _first(value: Any) -> Any:
    return value[0] if isinstance(value, list) and value else None


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


run_launcher_subprocess = run_fake_launcher
