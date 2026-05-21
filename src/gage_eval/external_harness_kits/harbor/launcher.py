"""Python subprocess launcher for Harbor jobs."""

from __future__ import annotations

import argparse
import asyncio
from dataclasses import dataclass
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import signal
import shutil
import subprocess
import sys
import threading
import time
import traceback
from typing import Any, BinaryIO, Mapping, Sequence, TextIO

from loguru import logger

from gage_eval.external_harness_kits.secret_redaction import (
    contains_secret_like_text,
    redact_for_artifact,
    redact_text,
)

RESULT_SCHEMA_VERSION = "gage.harbor_launcher_result.v1"
STDOUT_REF = "launcher.stdout.log"
STDERR_REF = "launcher.stderr.log"
TRACEBACK_REF = "launcher.traceback.log"
JOB_LOG_FILENAME = "job.log"
DEFAULT_JOB_LOG_POLL_INTERVAL_S = 0.5
LIVE_LOG_STDOUT_PREFIX = "[harbor-stdout] "
LIVE_LOG_STDERR_PREFIX = "[harbor-stderr] "
LIVE_LOG_JOB_PREFIX = "[harbor-job] "
SUBPROCESS_HEARTBEAT_INTERVAL_S = 15.0
GRACEFUL_TERMINATE_TIMEOUT_S = 30.0
FORCE_KILL_TIMEOUT_S = 5.0
DOCKER_CLEANUP_TIMEOUT_S = 10.0
DOCKER_COMPOSE_PROJECT_LABEL = "com.docker.compose.project"


@dataclass(frozen=True)
class LauncherSubprocessResult:
    argv: list[str]
    exit_code: int
    timed_out: bool
    result_file: Path
    stdout_path: Path
    stderr_path: Path


def build_launcher_argv(
    *,
    config_path: Path | str,
    result_file: Path | str,
    python: str | None = None,
    environ: Mapping[str, str] | None = None,
) -> list[str]:
    """Build the secret-free argv used for the Harbor launcher subprocess."""

    _reject_secret_like_argv_path(config_path, environ=environ)
    _reject_secret_like_argv_path(result_file, environ=environ)
    if python is not None:
        _reject_secret_like_argv_path(python, environ=environ)
    return [
        python or sys.executable,
        "-m",
        "gage_eval.external_harness_kits.harbor.launcher",
        "--config",
        str(config_path),
        "--result-file",
        str(result_file),
    ]


def run_launcher(
    *,
    config_path: Path | str,
    result_file: Path | str,
) -> int:
    """Run a Harbor job from a launcher input JSON and write the result schema."""

    config_path = Path(config_path)
    result_file = Path(result_file)
    workdir = result_file.parent
    stdout_path = workdir / STDOUT_REF
    stderr_path = workdir / STDERR_REF
    started_at = _utc_now()
    result_file.parent.mkdir(parents=True, exist_ok=True)
    stdout_path.touch(exist_ok=True)
    stderr_path.touch(exist_ok=True)
    job_config_payload: dict[str, Any] = {}
    job_name = "harbor_job"
    jobs_dir = (workdir / "jobs").resolve()
    job_dir = jobs_dir / job_name
    exit_code = 0
    harbor_job_result: dict[str, Any] | None = None
    launcher_error: dict[str, Any] | None = None
    try:
        launcher_input = _read_launcher_input(config_path)
        job_config_payload = _job_config_payload(launcher_input)
        job_name = str(job_config_payload.get("job_name") or launcher_input.get("job_name") or job_name)
        jobs_dir = Path(str(job_config_payload.get("jobs_dir") or launcher_input.get("jobs_dir") or jobs_dir)).resolve()
        job_config_payload["job_name"] = job_name
        job_config_payload["jobs_dir"] = str(jobs_dir)
        job_dir = jobs_dir / job_name
        harbor_job_result = asyncio.run(_run_harbor_job(job_config_payload))
    except Exception as exc:
        exit_code = 1
        _write_text(workdir / TRACEBACK_REF, "".join(traceback.format_exception(exc)))
        launcher_error = _launcher_error(
            error_type=type(exc).__name__,
            message=str(exc),
            traceback_ref=TRACEBACK_REF,
        )

    result = _result_payload(
        exit_code=exit_code,
        started_at=started_at,
        finished_at=_utc_now(),
        pid=os.getpid(),
        python=sys.executable,
        config_path=config_path,
        job_name=job_name,
        jobs_dir=jobs_dir,
        job_dir=job_dir,
        harbor_job_result=harbor_job_result,
        launcher_error=launcher_error,
    )
    _write_json(result_file, result)
    return exit_code


def run_launcher_subprocess(
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
    """Launch Harbor in a child Python process and enforce timeout via process group kill.

    ``live_log`` controls whether subprocess stdout/stderr and the Harbor
    ``job.log`` file are mirrored to the operator terminal while the job runs. Output
    is always captured to disk under ``stdout_path`` and ``stderr_path``; the
    live mirror is purely an operator convenience for long-running Harbor jobs
    that otherwise appear silent on the terminal.
    """

    config_arg = Path(config_path)
    result_arg = Path(result_file)
    workdir_path = Path(workdir) if workdir is not None else result_arg.parent
    workdir_path.mkdir(parents=True, exist_ok=True)
    config_path = _resolve_from_workdir(config_arg, workdir_path)
    result_file = _resolve_from_workdir(result_arg, workdir_path)
    stdout_path = workdir_path / STDOUT_REF
    stderr_path = workdir_path / STDERR_REF
    env = os.environ.copy()
    if environ:
        env.update({str(key): str(value) for key, value in environ.items()})
    argv = build_launcher_argv(config_path=config_arg, result_file=result_arg, python=python, environ=env)
    _safe_unlink(result_file)
    _safe_unlink(result_file.parent / TRACEBACK_REF)
    started_at = _utc_now()
    job_log_target = Path(job_log_path) if job_log_path is not None else None
    job_log_stop_event = threading.Event()
    logger.info(
        "Starting Harbor launcher subprocess (config={}, result_file={}, workdir={})",
        redact_text(str(config_path)),
        redact_text(str(result_file)),
        redact_text(str(workdir_path)),
    )
    with stdout_path.open("wb") as stdout_fh, stderr_path.open("wb") as stderr_fh:
        process = subprocess.Popen(
            argv,
            cwd=str(workdir_path),
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            start_new_session=True,
        )
        tee_threads = _start_subprocess_log_tee(
            process,
            stdout_fh=stdout_fh,
            stderr_fh=stderr_fh,
            live_log=live_log,
        )
        job_log_thread = _start_job_log_tail(
            job_log_path=job_log_target,
            stop_event=job_log_stop_event,
            live_log=live_log,
        )
        timed_out = False
        elapsed_s = 0.0
        cleanup_reason: str | None = None
        try:
            exit_code_or_none, elapsed_s = _wait_for_subprocess_with_heartbeats(
                process,
                timeout_s=timeout_s,
                heartbeat_interval_s=SUBPROCESS_HEARTBEAT_INTERVAL_S,
            )
            if exit_code_or_none is None:
                timed_out = True
                cleanup_reason = "timeout"
                _kill_process_group(process.pid)
                exit_code = process.wait()
                _write_timeout_result(
                    config_path=config_path,
                    result_file=result_file,
                    started_at=started_at,
                    finished_at=_utc_now(),
                    pid=process.pid,
                    python=argv[0],
                    timeout_s=timeout_s,
                )
            else:
                exit_code = exit_code_or_none
        finally:
            if process.poll() is None:
                cleanup_reason = cleanup_reason or "interrupted"
                _terminate_process_group_gracefully(
                    process,
                    stderr_fh=stderr_fh,
                    graceful_timeout_s=GRACEFUL_TERMINATE_TIMEOUT_S,
                    force_timeout_s=FORCE_KILL_TIMEOUT_S,
                )
            if cleanup_reason is not None:
                _cleanup_harbor_docker_resources_for_config(
                    config_path,
                    stderr_fh=stderr_fh,
                    reason=cleanup_reason,
                )
            _join_threads(tee_threads)
            job_log_stop_event.set()
            if job_log_thread is not None:
                job_log_thread.join(timeout=DEFAULT_JOB_LOG_POLL_INTERVAL_S * 4)
    logger.info(
        "Harbor launcher subprocess finished (pid={}, exit_code={}, timed_out={}, elapsed_s={:.1f})",
        process.pid,
        exit_code,
        timed_out,
        elapsed_s,
    )
    if not timed_out and not _launcher_result_file_valid(
        result_file,
        expected_pid=process.pid,
        expected_exit_code=exit_code,
    ):
        exit_code = exit_code if exit_code != 0 else 1
        _write_missing_result_error(
            config_path=config_path,
            result_file=result_file,
            started_at=started_at,
            finished_at=_utc_now(),
            pid=process.pid,
            python=argv[0],
            exit_code=exit_code,
        )
        _cleanup_harbor_docker_resources_for_config(
            config_path,
            stderr_fh=None,
            reason="missing_result",
        )
    return LauncherSubprocessResult(
        argv=argv,
        exit_code=exit_code,
        timed_out=timed_out,
        result_file=result_file,
        stdout_path=stdout_path,
        stderr_path=stderr_path,
    )


def _wait_for_subprocess_with_heartbeats(
    process: subprocess.Popen[bytes],
    *,
    timeout_s: float | None,
    heartbeat_interval_s: float,
) -> tuple[int | None, float]:
    started = time.monotonic()
    next_heartbeat = started + max(1.0, float(heartbeat_interval_s))
    while True:
        exit_code = process.poll()
        now = time.monotonic()
        elapsed_s = now - started
        if exit_code is not None:
            return exit_code, elapsed_s
        if timeout_s is not None and elapsed_s >= timeout_s:
            return None, elapsed_s
        if now >= next_heartbeat:
            logger.info(
                "Harbor launcher subprocess still running (pid={}, elapsed_s={:.1f})",
                process.pid,
                elapsed_s,
            )
            next_heartbeat = now + max(1.0, float(heartbeat_interval_s))
        time.sleep(0.2)


def _start_subprocess_log_tee(
    process: subprocess.Popen[bytes],
    *,
    stdout_fh: BinaryIO,
    stderr_fh: BinaryIO,
    live_log: bool,
) -> list[threading.Thread]:
    threads: list[threading.Thread] = []
    if process.stdout is not None:
        threads.append(
            threading.Thread(
                target=_tee_subprocess_stream,
                kwargs={
                    "source": process.stdout,
                    "artifact_fh": stdout_fh,
                    "terminal_stream": sys.stdout if live_log else None,
                    "prefix": LIVE_LOG_STDOUT_PREFIX,
                },
                name="gage-harbor-stdout-tee",
                daemon=True,
            )
        )
    if process.stderr is not None:
        threads.append(
            threading.Thread(
                target=_tee_subprocess_stream,
                kwargs={
                    "source": process.stderr,
                    "artifact_fh": stderr_fh,
                    "terminal_stream": sys.stderr if live_log else None,
                    "prefix": LIVE_LOG_STDERR_PREFIX,
                },
                name="gage-harbor-stderr-tee",
                daemon=True,
            )
        )
    for thread in threads:
        thread.start()
    return threads


def _tee_subprocess_stream(
    *,
    source: BinaryIO,
    artifact_fh: BinaryIO,
    terminal_stream: TextIO | None,
    prefix: str,
) -> None:
    """Pump bytes from ``source`` into ``artifact_fh`` and optionally to ``terminal_stream``.

    The artifact file always receives raw bytes (unchanged contract with the
    existing log file consumers). The terminal mirror is chunked instead of
    line-buffered so no-newline progress output remains visible while Harbor
    is still running.
    """

    line_open = False
    try:
        while True:
            chunk = os.read(source.fileno(), 4096)
            if not chunk:
                break
            artifact_fh.write(chunk)
            artifact_fh.flush()
            if terminal_stream is None:
                continue
            line_open = _emit_live_chunk(terminal_stream, prefix, chunk, line_open=line_open)
    finally:
        try:
            source.close()
        except Exception:
            pass
        if terminal_stream is not None and line_open:
            terminal_stream.write("\n")
            terminal_stream.flush()


def _emit_live_line(stream: TextIO, prefix: str, line_bytes: bytes) -> None:
    text = line_bytes.decode("utf-8", errors="replace")
    if not text.endswith("\n"):
        text = text + "\n"
    try:
        stream.write(f"{prefix}{redact_text(text)}")
        stream.flush()
    except Exception:
        # Never let a terminal write failure crash the launcher.
        return


def _emit_live_chunk(stream: TextIO, prefix: str, chunk: bytes, *, line_open: bool) -> bool:
    text = redact_text(chunk.decode("utf-8", errors="replace"))
    if not text:
        return line_open
    try:
        for part in text.splitlines(keepends=True) or [text]:
            if not line_open:
                stream.write(prefix)
            stream.write(part)
            line_open = not part.endswith("\n")
        stream.flush()
    except Exception:
        return line_open
    return line_open


def _start_job_log_tail(
    *,
    job_log_path: Path | None,
    stop_event: threading.Event,
    live_log: bool,
) -> threading.Thread | None:
    if not live_log or job_log_path is None:
        return None
    thread = threading.Thread(
        target=_tail_job_log,
        kwargs={
            "job_log_path": job_log_path,
            "stop_event": stop_event,
            "terminal_stream": sys.stderr,
            "poll_interval_s": DEFAULT_JOB_LOG_POLL_INTERVAL_S,
        },
        name="gage-harbor-job-log-tail",
        daemon=True,
    )
    thread.start()
    return thread


def _tail_job_log(
    *,
    job_log_path: Path,
    stop_event: threading.Event,
    terminal_stream: TextIO,
    poll_interval_s: float,
) -> None:
    """Poll Harbor's ``job.log`` and forward new lines to the operator terminal.

    Harbor 0.6.6 routes its loguru logger to ``<job_dir>/job.log`` and does
    not emit to subprocess stdout/stderr; this tailer is the only practical
    way to give operators real-time visibility into Harbor's internal progress
    (e.g. "Sending keys: ls -la").
    """

    handle: BinaryIO | None = None
    leftover = bytearray()
    try:
        while not stop_event.is_set():
            if handle is None:
                try:
                    handle = job_log_path.open("rb")
                except FileNotFoundError:
                    if stop_event.wait(poll_interval_s):
                        break
                    continue
                except OSError:
                    return
            chunk = handle.read()
            if chunk:
                leftover.extend(chunk)
                while True:
                    newline_index = leftover.find(b"\n")
                    if newline_index < 0:
                        break
                    line_bytes = bytes(leftover[: newline_index + 1])
                    del leftover[: newline_index + 1]
                    _emit_live_line(terminal_stream, LIVE_LOG_JOB_PREFIX, line_bytes)
            elif stop_event.wait(poll_interval_s):
                break
        # Drain one more pass after stop to capture trailing lines written
        # between the last poll and process termination.
        if handle is None:
            try:
                handle = job_log_path.open("rb")
            except FileNotFoundError:
                return
            except OSError:
                return
        if handle is not None:
            tail = handle.read()
            if tail:
                leftover.extend(tail)
            while True:
                newline_index = leftover.find(b"\n")
                if newline_index < 0:
                    break
                line_bytes = bytes(leftover[: newline_index + 1])
                del leftover[: newline_index + 1]
                _emit_live_line(terminal_stream, LIVE_LOG_JOB_PREFIX, line_bytes)
            if leftover:
                _emit_live_line(terminal_stream, LIVE_LOG_JOB_PREFIX, bytes(leftover))
    finally:
        if handle is not None:
            try:
                handle.close()
            except Exception:
                pass


def _join_threads(threads: Sequence[threading.Thread]) -> None:
    for thread in threads:
        thread.join()


async def _run_harbor_job(job_config_payload: Mapping[str, Any]) -> dict[str, Any]:
    from harbor.job import Job
    from harbor.models.job.config import JobConfig

    _install_harbor_compat_patches()
    config = JobConfig.model_validate(dict(job_config_payload))
    job = await Job.create(config)
    result = await job.run()
    return redact_for_artifact(_model_dump_jsonable(result))


def _install_harbor_compat_patches() -> None:
    """Install small runtime compatibility shims for the pinned Harbor package.

    Harbor 0.6.6 exposes SWE-agent cost/token flags but not SWE-agent's
    ``agent.model.per_instance_call_limit`` flag. Without this shim, GAGE can
    pass the setting through JobConfig but Harbor's installed-client base class
    silently ignores it when building the SWE-agent command line.
    """

    try:
        from harbor.agents.installed.base import CliFlag
        from harbor.agents.installed.swe_agent import SweAgent
    except Exception as exc:
        print(
            "GAGE Harbor compatibility patch skipped: Harbor SWE-agent module "
            f"unavailable ({type(exc).__name__}: {exc})",
            file=sys.stderr,
        )
        return

    if any(flag.kwarg == "per_instance_call_limit" for flag in SweAgent.CLI_FLAGS):
        return
    SweAgent.CLI_FLAGS = [
        *SweAgent.CLI_FLAGS,
        CliFlag(
            "per_instance_call_limit",
            cli="--agent.model.per_instance_call_limit",
            type="int",
        ),
    ]


def _read_launcher_input(config_path: Path) -> dict[str, Any]:
    with config_path.open("r", encoding="utf-8") as fh:
        payload = json.load(fh)
    if not isinstance(payload, dict):
        raise ValueError("launcher input must be a JSON object")
    return payload


def _job_config_payload(payload: Mapping[str, Any]) -> dict[str, Any]:
    raw_job_config = payload.get("job_config")
    if isinstance(raw_job_config, Mapping):
        job_config = dict(raw_job_config)
    else:
        job_config = dict(payload)
    if payload.get("job_name") is not None:
        job_config.setdefault("job_name", payload["job_name"])
    if payload.get("jobs_dir") is not None:
        job_config.setdefault("jobs_dir", payload["jobs_dir"])
    return job_config


def _model_dump_jsonable(value: Any) -> dict[str, Any]:
    if hasattr(value, "model_dump"):
        dumped = value.model_dump(mode="json")
    elif hasattr(value, "model_dump_json"):
        dumped = json.loads(value.model_dump_json())
    else:
        dumped = value
    if isinstance(dumped, dict):
        return dumped
    return {"value": dumped}


def _write_timeout_result(
    *,
    config_path: Path,
    result_file: Path,
    started_at: str,
    finished_at: str,
    pid: int,
    python: str,
    timeout_s: float | None,
) -> None:
    launcher_input = _safe_read_launcher_input(config_path)
    job_config_payload = _job_config_payload(launcher_input)
    job_name = str(job_config_payload.get("job_name") or launcher_input.get("job_name") or "harbor_job")
    jobs_dir = Path(str(job_config_payload.get("jobs_dir") or launcher_input.get("jobs_dir") or "jobs")).resolve()
    _write_text(
        result_file.parent / TRACEBACK_REF,
        f"Harbor launcher timed out after {timeout_s} seconds; process group was killed.\n",
    )
    result = _result_payload(
        exit_code=-signal.SIGKILL,
        started_at=started_at,
        finished_at=finished_at,
        pid=pid,
        python=python,
        config_path=config_path,
        job_name=job_name,
        jobs_dir=jobs_dir,
        job_dir=jobs_dir / job_name,
        harbor_job_result=None,
        launcher_error=_launcher_error(
            error_type="TimeoutExpired",
            message=f"Harbor launcher timed out after {timeout_s} seconds",
            traceback_ref=TRACEBACK_REF,
        ),
    )
    _write_json(result_file, result)


def _write_missing_result_error(
    *,
    config_path: Path,
    result_file: Path,
    started_at: str,
    finished_at: str,
    pid: int,
    python: str,
    exit_code: int,
) -> None:
    launcher_input = _safe_read_launcher_input(config_path)
    job_config_payload = _job_config_payload(launcher_input)
    job_name = str(job_config_payload.get("job_name") or launcher_input.get("job_name") or "harbor_job")
    jobs_dir = Path(str(job_config_payload.get("jobs_dir") or launcher_input.get("jobs_dir") or "jobs")).resolve()
    _write_text(
        result_file.parent / TRACEBACK_REF,
        f"Harbor launcher exited with code {exit_code} without a valid result JSON.\n",
    )
    result = _result_payload(
        exit_code=exit_code,
        started_at=started_at,
        finished_at=finished_at,
        pid=pid,
        python=python,
        config_path=config_path,
        job_name=job_name,
        jobs_dir=jobs_dir,
        job_dir=jobs_dir / job_name,
        harbor_job_result=None,
        launcher_error=_launcher_error(
            error_type="MissingLauncherResult",
            message="Harbor launcher exited without a valid result JSON",
            traceback_ref=TRACEBACK_REF,
        ),
    )
    _write_json(result_file, result)


def _result_payload(
    *,
    exit_code: int,
    started_at: str,
    finished_at: str,
    pid: int,
    python: str,
    config_path: Path,
    job_name: str,
    jobs_dir: Path,
    job_dir: Path,
    harbor_job_result: dict[str, Any] | None,
    launcher_error: dict[str, Any] | None,
) -> dict[str, Any]:
    return {
        "schema_version": RESULT_SCHEMA_VERSION,
        "exit_code": exit_code,
        "started_at": started_at,
        "finished_at": finished_at,
        "pid": pid,
        "python": redact_text(python),
        "config_path": redact_text(str(config_path)),
        "job_name": redact_text(job_name),
        "jobs_dir": redact_text(str(jobs_dir)),
        "job_dir": redact_text(str(job_dir)),
        "harbor_job_result": redact_for_artifact(harbor_job_result),
        "launcher_error": launcher_error,
        "stdout_ref": STDOUT_REF,
        "stderr_ref": STDERR_REF,
    }


def _launcher_error(*, error_type: str, message: str, traceback_ref: str) -> dict[str, str]:
    return {
        "type": redact_text(error_type),
        "message": redact_text(message),
        "traceback_ref": traceback_ref,
    }


def _safe_read_launcher_input(config_path: Path) -> dict[str, Any]:
    try:
        return _read_launcher_input(config_path)
    except Exception:
        return {}


def _resolve_from_workdir(path: Path, workdir: Path) -> Path:
    if path.is_absolute():
        return path
    return workdir / path


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(redact_for_artifact(payload), indent=2, sort_keys=True)
    _atomic_write_text(path, text)


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    _atomic_write_text(path, redact_text(text))


def _atomic_write_text(path: Path, text: str) -> None:
    tmp_path = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    try:
        with tmp_path.open("w", encoding="utf-8") as fh:
            fh.write(text)
            fh.flush()
            os.fsync(fh.fileno())
        tmp_path.replace(path)
    finally:
        _safe_unlink(tmp_path)


def _safe_unlink(path: Path) -> None:
    try:
        path.unlink()
    except FileNotFoundError:
        return


def _launcher_result_file_valid(path: Path, *, expected_pid: int, expected_exit_code: int) -> bool:
    try:
        with path.open("r", encoding="utf-8") as fh:
            payload = json.load(fh)
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return False
    if not isinstance(payload, Mapping):
        return False
    return (
        payload.get("schema_version") == RESULT_SCHEMA_VERSION
        and payload.get("pid") == expected_pid
        and payload.get("exit_code") == expected_exit_code
        and payload.get("stdout_ref") == STDOUT_REF
        and payload.get("stderr_ref") == STDERR_REF
        and isinstance(payload.get("exit_code"), int)
    )


def _reject_secret_like_argv_path(path: Path | str, *, environ: Mapping[str, str] | None) -> None:
    if contains_secret_like_text(str(path), environ=environ):
        raise ValueError("launcher argv paths must not contain secret-like values")


def _terminate_process_group_gracefully(
    process: subprocess.Popen[bytes],
    *,
    stderr_fh: BinaryIO,
    graceful_timeout_s: float,
    force_timeout_s: float,
) -> int | None:
    if process.poll() is not None:
        return process.returncode

    message = (
        f"Launcher interrupted while subprocess pgid={process.pid} still running; "
        f"sending SIGTERM (graceful_s={graceful_timeout_s})"
    )
    logger.warning(message)
    _write_stderr_notice(stderr_fh, message)
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        return process.poll()

    try:
        return process.wait(timeout=max(0.0, float(graceful_timeout_s)))
    except subprocess.TimeoutExpired:
        message = "Subprocess did not exit after SIGTERM; escalating to SIGKILL"
        logger.warning(message)
        _write_stderr_notice(stderr_fh, message)
        _kill_process_group(process.pid)
        try:
            return process.wait(timeout=max(0.0, float(force_timeout_s)))
        except subprocess.TimeoutExpired:
            return process.poll()


def _write_stderr_notice(stderr_fh: BinaryIO, message: str) -> None:
    try:
        stderr_fh.write(f"{message}\n".encode("utf-8", errors="replace"))
        stderr_fh.flush()
    except Exception:
        return


def _kill_process_group(pid: int) -> None:
    try:
        os.killpg(pid, signal.SIGKILL)
    except ProcessLookupError:
        return


def _cleanup_harbor_docker_resources_for_config(
    config_path: Path,
    *,
    stderr_fh: BinaryIO | None,
    reason: str,
) -> None:
    projects = _harbor_compose_projects_from_config(config_path)
    if not projects:
        return
    if shutil.which("docker") is None:
        _write_cleanup_notice(
            stderr_fh,
            f"Skipping Harbor Docker cleanup after {reason}: docker is not on PATH",
        )
        return

    cleaned: list[str] = []
    for project in projects:
        try:
            container_ids = _docker_cli_lines(
                [
                    "docker",
                    "ps",
                    "-a",
                    "--filter",
                    f"label={DOCKER_COMPOSE_PROJECT_LABEL}={project}",
                    "--format",
                    "{{.ID}}",
                ]
            )
            if container_ids:
                _run_docker_cli(
                    ["docker", "rm", "-f", *container_ids],
                    timeout_s=DOCKER_CLEANUP_TIMEOUT_S,
                )
            network_names = _docker_cli_lines(
                [
                    "docker",
                    "network",
                    "ls",
                    "--filter",
                    f"label={DOCKER_COMPOSE_PROJECT_LABEL}={project}",
                    "--format",
                    "{{.Name}}",
                ]
            )
            if network_names:
                _run_docker_cli(
                    ["docker", "network", "rm", *network_names],
                    timeout_s=DOCKER_CLEANUP_TIMEOUT_S,
                )
            if container_ids or network_names:
                cleaned.append(project)
        except Exception as exc:
            _write_cleanup_notice(
                stderr_fh,
                f"Harbor Docker cleanup failed for compose project {project!r} after {reason}: {exc}",
            )
    if cleaned:
        _write_cleanup_notice(
            stderr_fh,
            f"Harbor Docker cleanup after {reason}: removed compose resources for {', '.join(cleaned)}",
        )


def _harbor_compose_projects_from_config(config_path: Path) -> list[str]:
    launcher_input = _safe_read_launcher_input(config_path)
    if not launcher_input:
        return []
    job_config_payload = _job_config_payload(launcher_input)
    job_name = str(
        job_config_payload.get("job_name") or launcher_input.get("job_name") or ""
    )
    jobs_dir_raw = job_config_payload.get("jobs_dir") or launcher_input.get("jobs_dir")
    if not job_name or jobs_dir_raw is None:
        return []
    jobs_dir = Path(str(jobs_dir_raw))
    if not jobs_dir.is_absolute():
        jobs_dir = config_path.parent / jobs_dir
    return _harbor_compose_projects_from_job_dir(jobs_dir.resolve() / job_name)


def _harbor_compose_projects_from_job_dir(job_dir: Path) -> list[str]:
    try:
        children = list(job_dir.iterdir())
    except OSError:
        return []
    seen: set[str] = set()
    projects: list[str] = []
    for child in children:
        if not child.is_dir():
            continue
        project = _sanitize_harbor_docker_compose_project_name(child.name)
        if project and project not in seen:
            seen.add(project)
            projects.append(project)
    return projects


def _sanitize_harbor_docker_compose_project_name(name: str) -> str:
    lowered = name.lower()
    if not re.match(r"^[a-z0-9]", lowered):
        lowered = "0" + lowered
    return re.sub(r"[^a-z0-9_-]", "-", lowered)


def _docker_cli_lines(args: list[str]) -> list[str]:
    output = _run_docker_cli(args, timeout_s=DOCKER_CLEANUP_TIMEOUT_S)
    return [line.strip() for line in output.splitlines() if line.strip()]


def _run_docker_cli(args: list[str], *, timeout_s: float) -> str:
    result = subprocess.run(
        args,
        capture_output=True,
        check=False,
        text=True,
        timeout=max(0.1, float(timeout_s)),
    )
    if result.returncode != 0:
        detail = (result.stderr or result.stdout or "").strip()
        raise RuntimeError(detail or f"docker command exited {result.returncode}")
    return result.stdout or ""


def _write_cleanup_notice(stderr_fh: BinaryIO | None, message: str) -> None:
    logger.warning(message)
    if stderr_fh is not None:
        _write_stderr_notice(stderr_fh, message)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run a Harbor JobConfig through the GAGE launcher protocol.")
    parser.add_argument("--config", required=True, help="Path to Harbor launcher input JSON.")
    parser.add_argument("--result-file", required=True, help="Path to write launcher result JSON.")
    args = parser.parse_args(argv)
    return run_launcher(config_path=args.config, result_file=args.result_file)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = [
    "LauncherSubprocessResult",
    "RESULT_SCHEMA_VERSION",
    "build_launcher_argv",
    "main",
    "run_launcher",
    "run_launcher_subprocess",
]
