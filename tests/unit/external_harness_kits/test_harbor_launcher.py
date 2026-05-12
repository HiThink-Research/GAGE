from __future__ import annotations

import builtins
import json
import os
from pathlib import Path
import sys
from typing import Any

import pytest

from gage_eval.external_harness_kits.harbor import launcher as harbor_launcher
from gage_eval.external_harness_kits.harbor.launcher import (
    RESULT_SCHEMA_VERSION,
    build_launcher_argv,
    run_launcher,
    run_launcher_subprocess,
)


def _launcher_input(tmp_path: Path) -> dict[str, Any]:
    return {
        "job_config": {
            "job_name": "gage_terminal_bench_lmstudio_20260509",
            "jobs_dir": str(tmp_path / "jobs"),
            "environment": {"type": "docker"},
            "agents": [{"name": "nop"}],
            "datasets": [],
            "tasks": [{"path": "/tmp/gage-harbor-task"}],
        },
        "jobs_dir": str(tmp_path / "jobs"),
        "job_name": "gage_terminal_bench_lmstudio_20260509",
        "metadata": {
            "gage_task_id": "terminal_bench_20_lmstudio_smoke",
            "adapter_id": "harbor_tb2",
        },
    }


def _write_launcher_input(tmp_path: Path, payload: dict[str, Any] | None = None) -> Path:
    config_path = tmp_path / "harbor_launcher_input.json"
    config_path.write_text(json.dumps(payload or _launcher_input(tmp_path)), encoding="utf-8")
    return config_path


@pytest.mark.fast
def test_launcher_writes_result_json_on_success(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from harbor.job import Job

    class FakeJobResult:
        def model_dump(self, *, mode: str) -> dict[str, Any]:
            assert mode == "json"
            return {"id": "result-1", "trial_results": []}

    class FakeJob:
        async def run(self) -> FakeJobResult:
            return FakeJobResult()

    async def fake_create(cls, config):
        assert config.job_name == "gage_terminal_bench_lmstudio_20260509"
        return FakeJob()

    monkeypatch.setattr(Job, "create", classmethod(fake_create))
    config_path = _write_launcher_input(tmp_path)
    result_path = tmp_path / "launcher_result.json"

    exit_code = run_launcher(config_path=config_path, result_file=result_path)

    result = json.loads(result_path.read_text(encoding="utf-8"))
    assert exit_code == 0
    assert result["schema_version"] == RESULT_SCHEMA_VERSION
    assert result["exit_code"] == 0
    assert result["pid"] == os.getpid()
    assert result["python"] == sys.executable
    assert result["config_path"] == str(config_path)
    assert result["job_name"] == "gage_terminal_bench_lmstudio_20260509"
    assert result["jobs_dir"] == str((tmp_path / "jobs").resolve())
    assert result["job_dir"] == str((tmp_path / "jobs" / "gage_terminal_bench_lmstudio_20260509").resolve())
    assert result["harbor_job_result"] == {"id": "result-1", "trial_results": []}
    assert result["launcher_error"] is None
    assert result["stdout_ref"] == "launcher.stdout.log"
    assert result["stderr_ref"] == "launcher.stderr.log"
    assert (tmp_path / "launcher.stdout.log").exists()
    assert (tmp_path / "launcher.stderr.log").exists()
    assert result["started_at"] <= result["finished_at"]


@pytest.mark.fast
def test_launcher_installs_swe_agent_call_limit_compat_patch() -> None:
    from harbor.agents.installed.swe_agent import SweAgent

    original_flags = list(SweAgent.CLI_FLAGS)
    try:
        SweAgent.CLI_FLAGS = [
            flag for flag in SweAgent.CLI_FLAGS if flag.kwarg != "per_instance_call_limit"
        ]

        harbor_launcher._install_harbor_compat_patches()
        harbor_launcher._install_harbor_compat_patches()

        matching = [
            flag for flag in SweAgent.CLI_FLAGS if flag.kwarg == "per_instance_call_limit"
        ]
        assert len(matching) == 1
        assert matching[0].cli == "--agent.model.per_instance_call_limit"
        assert matching[0].type == "int"
    finally:
        SweAgent.CLI_FLAGS = original_flags


@pytest.mark.fast
def test_install_harbor_compat_patches_noop_when_swe_agent_module_missing(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    real_import = builtins.__import__

    def fake_import(name: str, *args: Any, **kwargs: Any) -> Any:
        if name == "harbor.agents.installed.swe_agent":
            raise ModuleNotFoundError("No module named 'harbor.agents.installed.swe_agent'")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)

    harbor_launcher._install_harbor_compat_patches()

    captured = capsys.readouterr()
    assert captured.out == ""
    assert "GAGE Harbor compatibility patch skipped" in captured.err
    assert "harbor.agents.installed.swe_agent" in captured.err


@pytest.mark.fast
def test_launcher_writes_launcher_error_on_exception(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from harbor.job import Job
    secret = "sk-test-launcher-error"
    monkeypatch.setenv("OPENAI_API_KEY", secret)

    async def fake_create(cls, config):
        raise RuntimeError(f"boom for {config.job_name} with {secret}")

    monkeypatch.setattr(Job, "create", classmethod(fake_create))
    config_path = _write_launcher_input(tmp_path)
    result_path = tmp_path / "launcher_result.json"

    exit_code = run_launcher(config_path=config_path, result_file=result_path)

    result = json.loads(result_path.read_text(encoding="utf-8"))
    assert exit_code == 1
    assert result["exit_code"] == 1
    assert result["harbor_job_result"] is None
    assert result["launcher_error"]["type"] == "RuntimeError"
    assert result["launcher_error"]["traceback_ref"] == "launcher.traceback.log"
    assert "boom for gage_terminal_bench_lmstudio_20260509" in result["launcher_error"]["message"]
    assert secret not in result["launcher_error"]["message"]
    traceback_text = (tmp_path / "launcher.traceback.log").read_text(encoding="utf-8")
    assert "RuntimeError" in traceback_text
    assert secret not in traceback_text
    assert "<redacted>" in traceback_text


@pytest.mark.fast
def test_launcher_writes_launcher_error_when_input_json_is_invalid(tmp_path: Path) -> None:
    config_path = tmp_path / "harbor_launcher_input.json"
    config_path.write_text("{not-json", encoding="utf-8")
    result_path = tmp_path / "launcher_result.json"

    exit_code = run_launcher(config_path=config_path, result_file=result_path)

    result = json.loads(result_path.read_text(encoding="utf-8"))
    assert exit_code == 1
    assert result["schema_version"] == RESULT_SCHEMA_VERSION
    assert result["job_name"] == "harbor_job"
    assert result["harbor_job_result"] is None
    assert result["launcher_error"]["type"] == "JSONDecodeError"
    assert result["launcher_error"]["traceback_ref"] == "launcher.traceback.log"


@pytest.mark.fast
def test_timeout_path_terminates_subprocess_and_writes_structured_error(tmp_path: Path) -> None:
    fake_package = tmp_path / "fake_harbor"
    _write_fake_slow_harbor(fake_package)
    config_path = _write_launcher_input(tmp_path)
    result_path = tmp_path / "launcher_result.json"
    repo_src = Path(__file__).resolve().parents[3] / "src"
    env = {
        "PYTHONPATH": f"{fake_package}{os.pathsep}{repo_src}{os.pathsep}{os.environ.get('PYTHONPATH', '')}",
    }

    result = run_launcher_subprocess(
        config_path=config_path,
        result_file=result_path,
        timeout_s=0.2,
        environ=env,
        python=sys.executable,
        workdir=tmp_path,
    )

    payload = json.loads(result_path.read_text(encoding="utf-8"))
    assert result.timed_out is True
    assert result.exit_code != 0
    assert payload["schema_version"] == RESULT_SCHEMA_VERSION
    assert payload["exit_code"] == result.exit_code
    assert payload["launcher_error"]["type"] == "TimeoutExpired"
    assert payload["launcher_error"]["traceback_ref"] == "launcher.traceback.log"
    assert payload["stdout_ref"] == "launcher.stdout.log"
    assert payload["stderr_ref"] == "launcher.stderr.log"
    assert "timed out" in (tmp_path / "launcher.traceback.log").read_text(encoding="utf-8")


@pytest.mark.fast
def test_subprocess_is_terminated_when_parent_wait_is_interrupted(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_package = tmp_path / "fake_harbor"
    _write_fake_slow_harbor(fake_package)
    config_path = _write_launcher_input(tmp_path)
    result_path = tmp_path / "launcher_result.json"
    repo_src = Path(__file__).resolve().parents[3] / "src"
    env = {
        "PYTHONPATH": f"{fake_package}{os.pathsep}{repo_src}{os.pathsep}{os.environ.get('PYTHONPATH', '')}",
    }
    captured: dict[str, Any] = {}

    def interrupt_wait(process, **_kwargs):
        captured["process"] = process
        raise KeyboardInterrupt

    monkeypatch.setattr(harbor_launcher, "_wait_for_subprocess_with_heartbeats", interrupt_wait)

    with pytest.raises(KeyboardInterrupt):
        run_launcher_subprocess(
            config_path=config_path,
            result_file=result_path,
            timeout_s=5,
            environ=env,
            python=sys.executable,
            workdir=tmp_path,
        )

    process = captured["process"]
    assert process.poll() is not None
    stderr_text = (tmp_path / "launcher.stderr.log").read_text(encoding="utf-8")
    assert "Launcher interrupted while subprocess pgid=" in stderr_text
    assert "sending SIGTERM" in stderr_text


@pytest.mark.fast
def test_subprocess_overwrites_stale_result_when_child_exits_without_result(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workdir = tmp_path / "launcher-workdir"
    parent_cwd = tmp_path / "parent-cwd"
    workdir.mkdir()
    parent_cwd.mkdir()
    monkeypatch.chdir(parent_cwd)
    stale_result = workdir / "launcher_result.json"
    stale_result.write_text(
        json.dumps(
            {
                "schema_version": RESULT_SCHEMA_VERSION,
                "exit_code": 0,
                "pid": 123,
                "launcher_error": None,
                "harbor_job_result": {"id": "stale"},
            }
        ),
        encoding="utf-8",
    )

    result = run_launcher_subprocess(
        config_path=Path("missing_input.json"),
        result_file=Path("launcher_result.json"),
        timeout_s=5,
        environ={"PYTHONPATH": ""},
        python=sys.executable,
        workdir=workdir,
    )

    payload = json.loads(stale_result.read_text(encoding="utf-8"))
    assert result.exit_code != 0
    assert payload["launcher_error"]["type"] == "MissingLauncherResult"
    assert payload["harbor_job_result"] is None
    assert payload["pid"] != 123


@pytest.mark.fast
def test_subprocess_rejects_result_json_with_mismatched_exit_code(tmp_path: Path) -> None:
    fake_package = tmp_path / "fake_harbor"
    _write_fake_mismatched_exit_harbor(fake_package)
    config_path = _write_launcher_input(tmp_path)
    result_path = tmp_path / "launcher_result.json"
    repo_src = Path(__file__).resolve().parents[3] / "src"
    env = {
        "PYTHONPATH": f"{fake_package}{os.pathsep}{repo_src}{os.pathsep}{os.environ.get('PYTHONPATH', '')}",
    }

    result = run_launcher_subprocess(
        config_path=config_path,
        result_file=result_path,
        timeout_s=5,
        environ=env,
        python=sys.executable,
        workdir=tmp_path,
    )

    payload = json.loads(result_path.read_text(encoding="utf-8"))
    assert result.exit_code == 7
    assert payload["exit_code"] == 7
    assert payload["launcher_error"]["type"] == "MissingLauncherResult"
    assert payload["harbor_job_result"] is None


@pytest.mark.fast
def test_subprocess_accepts_relative_config_and_result_paths_from_workdir(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    parent_cwd = tmp_path / "parent-cwd"
    workdir = tmp_path / "launcher-workdir"
    parent_cwd.mkdir()
    workdir.mkdir()
    monkeypatch.chdir(parent_cwd)
    fake_package = tmp_path / "fake_harbor"
    _write_fake_success_harbor(fake_package)
    config_path = _write_launcher_input(workdir)
    repo_src = Path(__file__).resolve().parents[3] / "src"
    env = {
        "PYTHONPATH": f"{fake_package}{os.pathsep}{repo_src}{os.pathsep}{os.environ.get('PYTHONPATH', '')}",
    }

    result = run_launcher_subprocess(
        config_path=Path(config_path.name),
        result_file=Path("launcher_result.json"),
        timeout_s=5,
        environ=env,
        python=sys.executable,
        workdir=workdir,
    )

    payload = json.loads((workdir / "launcher_result.json").read_text(encoding="utf-8"))
    assert result.exit_code == 0
    assert result.timed_out is False
    assert result.result_file == workdir / "launcher_result.json"
    assert payload["exit_code"] == 0
    assert payload["launcher_error"] is None
    assert payload["harbor_job_result"] == {"id": "fake-job", "trial_results": []}
    assert not (parent_cwd / "launcher_result.json").exists()


@pytest.mark.fast
def test_subprocess_tees_child_stdout_and_stderr_to_parent_streams(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    fake_package = tmp_path / "fake_harbor"
    _write_fake_logging_harbor(fake_package)
    config_path = _write_launcher_input(tmp_path)
    result_path = tmp_path / "launcher_result.json"
    repo_src = Path(__file__).resolve().parents[3] / "src"
    env = {
        "PYTHONPATH": f"{fake_package}{os.pathsep}{repo_src}{os.pathsep}{os.environ.get('PYTHONPATH', '')}",
    }

    result = run_launcher_subprocess(
        config_path=config_path,
        result_file=result_path,
        timeout_s=5,
        environ=env,
        python=sys.executable,
        workdir=tmp_path,
    )

    captured = capsys.readouterr()
    assert result.exit_code == 0
    assert "harbor stdout visible" in captured.out
    assert "harbor stdout partial" in captured.out
    assert "harbor stderr visible" in captured.err
    assert "harbor stdout visible" in (tmp_path / "launcher.stdout.log").read_text(encoding="utf-8")
    assert "harbor stdout partial" in (tmp_path / "launcher.stdout.log").read_text(encoding="utf-8")
    assert "harbor stderr visible" in (tmp_path / "launcher.stderr.log").read_text(encoding="utf-8")


@pytest.mark.fast
def test_subprocess_tails_harbor_job_log_into_parent_stderr(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Harbor 0.6.6 routes loguru to <job_dir>/job.log instead of stdout/stderr.

    The launcher must therefore tail that file from the parent process so
    operators see real-time Harbor progress (e.g. "Sending keys: ls -la").
    """

    fake_package = tmp_path / "fake_harbor"
    job_name = "gage_terminal_bench_lmstudio_20260509"
    jobs_dir = tmp_path / "jobs"
    job_dir = jobs_dir / job_name
    job_log_path = job_dir / "job.log"
    _write_fake_job_log_harbor(fake_package, job_log_path=job_log_path)

    config_path = _write_launcher_input(tmp_path)
    result_path = tmp_path / "launcher_result.json"
    repo_src = Path(__file__).resolve().parents[3] / "src"
    env = {
        "PYTHONPATH": f"{fake_package}{os.pathsep}{repo_src}{os.pathsep}{os.environ.get('PYTHONPATH', '')}",
    }

    result = run_launcher_subprocess(
        config_path=config_path,
        result_file=result_path,
        timeout_s=10,
        environ=env,
        python=sys.executable,
        workdir=tmp_path,
        live_log=True,
        job_log_path=job_log_path,
    )

    captured = capsys.readouterr()
    assert result.exit_code == 0
    assert "[harbor-job]" in captured.err
    assert "Installing tmux, asciinema" in captured.err
    assert "Sending keys: ls -la" in captured.err
    # Disabling live_log must suppress terminal mirroring while preserving the
    # subprocess stdout/stderr artifact files (asserted in the tee test above).
    capsys.readouterr()
    result_disabled = run_launcher_subprocess(
        config_path=config_path,
        result_file=result_path,
        timeout_s=10,
        environ=env,
        python=sys.executable,
        workdir=tmp_path,
        live_log=False,
        job_log_path=job_log_path,
    )
    captured_disabled = capsys.readouterr()
    assert result_disabled.exit_code == 0
    assert "[harbor-job]" not in captured_disabled.err
    assert "Installing tmux, asciinema" not in captured_disabled.err


@pytest.mark.fast
def test_launcher_command_args_do_not_contain_env_secret_strings(tmp_path: Path) -> None:
    secret = "sk-test-secret-value"
    fake_package = tmp_path / "fake_harbor"
    _write_fake_success_harbor(fake_package)
    config_path = _write_launcher_input(tmp_path)
    result_path = tmp_path / "launcher_result.json"
    repo_src = Path(__file__).resolve().parents[3] / "src"

    result = run_launcher_subprocess(
        config_path=config_path,
        result_file=result_path,
        python=sys.executable,
        timeout_s=5,
        environ={
            "OPENAI_API_KEY": secret,
            "PYTHONPATH": f"{fake_package}{os.pathsep}{repo_src}{os.pathsep}{os.environ.get('PYTHONPATH', '')}",
        },
        workdir=tmp_path,
    )

    assert secret not in " ".join(result.argv)
    assert str(config_path) in result.argv
    assert str(result_path) in result.argv


@pytest.mark.fast
def test_launcher_command_args_reject_secret_like_path_values(tmp_path: Path) -> None:
    secret = "sk-test-secret-value"

    with pytest.raises(ValueError, match="secret-like"):
        build_launcher_argv(
            config_path=tmp_path / f"{secret}.json",
            result_file=tmp_path / "launcher_result.json",
            python=sys.executable,
        )

    with pytest.raises(ValueError, match="secret-like"):
        build_launcher_argv(
            config_path=tmp_path / "harbor_launcher_input.json",
            result_file=tmp_path / "launcher_result.json",
            python=f"/tmp/{secret}/python",
        )


@pytest.mark.fast
def test_launcher_redacts_harbor_job_result_values(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from harbor.job import Job
    secret = "sk-test-job-result"
    monkeypatch.setenv("OPENAI_API_KEY", secret)

    class FakeJobResult:
        def model_dump(self, *, mode: str) -> dict[str, Any]:
            assert mode == "json"
            return {
                "id": "result-1",
                "metadata": {
                    "access_token": "plain-access-token-value",
                    "session_token": "plain-session-token-value",
                    "token": secret,
                    "ordinary": "visible",
                },
            }

    class FakeJob:
        async def run(self) -> FakeJobResult:
            return FakeJobResult()

    async def fake_create(cls, config):
        return FakeJob()

    monkeypatch.setattr(Job, "create", classmethod(fake_create))
    config_path = _write_launcher_input(tmp_path)
    result_path = tmp_path / "launcher_result.json"

    run_launcher(config_path=config_path, result_file=result_path)

    result = json.loads(result_path.read_text(encoding="utf-8"))
    assert secret not in json.dumps(result)
    assert "plain-access-token-value" not in json.dumps(result)
    assert "plain-session-token-value" not in json.dumps(result)
    assert result["harbor_job_result"]["metadata"]["access_token"] == "<redacted>"
    assert result["harbor_job_result"]["metadata"]["session_token"] == "<redacted>"
    assert result["harbor_job_result"]["metadata"]["token"] == "<redacted>"
    assert result["harbor_job_result"]["metadata"]["ordinary"] == "visible"


def _write_fake_success_harbor(root: Path) -> None:
    _write_fake_harbor_package(
        root,
        run_body=[
            "    async def run(self):",
            "        return FakeResult()",
            "class FakeResult:",
            "    def model_dump(self, *, mode):",
            "        return {'id': 'fake-job', 'trial_results': []}",
        ],
    )


def _write_fake_logging_harbor(root: Path) -> None:
    _write_fake_harbor_package(
        root,
        run_body=[
            "    async def run(self):",
            "        import sys",
            "        print('harbor stdout visible', flush=True)",
            "        sys.stdout.write('harbor stdout partial')",
            "        sys.stdout.flush()",
            "        print('harbor stderr visible', file=sys.stderr, flush=True)",
            "        return FakeResult()",
            "class FakeResult:",
            "    def model_dump(self, *, mode):",
            "        return {'id': 'fake-job', 'trial_results': []}",
        ],
    )


def _write_fake_job_log_harbor(root: Path, *, job_log_path: Path) -> None:
    """Fake Harbor that writes lines to ``<job_dir>/job.log`` over ~150ms.

    Mirrors Harbor 0.6.6's loguru-to-file behaviour: nothing on stdout/stderr,
    everything in a file at ``<jobs_dir>/<job_name>/job.log``.
    """

    job_log_literal = repr(str(job_log_path))
    _write_fake_harbor_package(
        root,
        run_body=[
            "    async def run(self):",
            "        import asyncio, pathlib",
            f"        log_path = pathlib.Path({job_log_literal})",
            "        log_path.parent.mkdir(parents=True, exist_ok=True)",
            "        log_path.write_text('', encoding='utf-8')",
            "        lines = [",
            "            'Installing tmux, asciinema',",
            "            'Sending keys: ls -la',",
            "            'Trajectory dumped',",
            "        ]",
            "        with log_path.open('a', encoding='utf-8') as fh:",
            "            for line in lines:",
            "                fh.write(line + '\\n')",
            "                fh.flush()",
            "                await asyncio.sleep(0.05)",
            "        return FakeResult()",
            "class FakeResult:",
            "    def model_dump(self, *, mode):",
            "        return {'id': 'fake-job', 'trial_results': []}",
        ],
    )


def _write_fake_slow_harbor(root: Path) -> None:
    _write_fake_harbor_package(
        root,
        run_body=[
            "    async def run(self):",
            "        await asyncio.sleep(30)",
        ],
    )


def _write_fake_mismatched_exit_harbor(root: Path) -> None:
    _write_fake_harbor_package(
        root,
        run_body=[
            "    async def run(self):",
            "        import json, os, sys",
            "        result_path = sys.argv[sys.argv.index('--result-file') + 1]",
            "        with open(result_path, 'w', encoding='utf-8') as fh:",
            "            json.dump({",
            "                'schema_version': 'gage.harbor_launcher_result.v1',",
            "                'exit_code': 0,",
            "                'started_at': '2026-05-09T10:00:00Z',",
            "                'finished_at': '2026-05-09T10:00:01Z',",
            "                'pid': os.getpid(),",
            "                'python': sys.executable,",
            "                'config_path': sys.argv[sys.argv.index('--config') + 1],",
            "                'job_name': 'fake',",
            "                'jobs_dir': 'jobs',",
            "                'job_dir': 'jobs/fake',",
            "                'harbor_job_result': {'id': 'contradictory'},",
            "                'launcher_error': None,",
            "                'stdout_ref': 'launcher.stdout.log',",
            "                'stderr_ref': 'launcher.stderr.log',",
            "            }, fh)",
            "        os._exit(7)",
        ],
    )


def _write_fake_harbor_package(root: Path, *, run_body: list[str]) -> None:
    (root / "harbor" / "models" / "job").mkdir(parents=True)
    (root / "harbor" / "__init__.py").write_text("", encoding="utf-8")
    (root / "harbor" / "models" / "__init__.py").write_text("", encoding="utf-8")
    (root / "harbor" / "models" / "job" / "__init__.py").write_text("", encoding="utf-8")
    (root / "harbor" / "models" / "job" / "config.py").write_text(
        "\n".join(
            [
                "from types import SimpleNamespace",
                "class JobConfig(SimpleNamespace):",
                "    @classmethod",
                "    def model_validate(cls, payload):",
                "        return cls(**payload)",
            ]
        ),
        encoding="utf-8",
    )
    (root / "harbor" / "job.py").write_text(
        "\n".join(
            [
                "import asyncio",
                "class Job:",
                "    @classmethod",
                "    async def create(cls, config):",
                "        return cls()",
                *run_body,
            ]
        ),
        encoding="utf-8",
    )
