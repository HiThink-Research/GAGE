"""SkillsBench judge implementation using official task-local tests."""

from __future__ import annotations

from pathlib import Path
import shutil
import subprocess
from typing import Any, Dict, Mapping, Optional
import uuid

from loguru import logger

from gage_eval.registry import registry
from gage_eval.role.judge.base import JudgeImplementation
from gage_eval.sandbox.docker_runtime import (
    build_docker_run_command,
    ensure_docker_image,
    normalize_runtime_configs,
)


@registry.asset(
    "judge_impls",
    "skillsbench_evaluate",
    desc="SkillsBench verifier using official tests/test.sh in the task image",
    tags=("skillsbench", "docker", "judge"),
)
class SkillsBenchEvaluate(JudgeImplementation):
    """Run official SkillsBench verifier scripts against the captured workspace."""

    def __init__(
        self,
        *,
        docker_bin: str = "docker",
        default_timeout_s: int = 1800,
    ) -> None:
        self._docker_bin = docker_bin
        self._default_timeout_s = max(1, int(default_timeout_s))

    def invoke(self, payload: Dict[str, Any], state: Any = None) -> Dict[str, Any]:
        sample = payload.get("sample") if isinstance(payload.get("sample"), dict) else {}
        params = payload.get("params") if isinstance(payload.get("params"), dict) else {}
        artifact_paths = payload.get("artifact_paths") if isinstance(payload.get("artifact_paths"), dict) else {}
        metadata = sample.get("metadata") if isinstance(sample.get("metadata"), dict) else {}
        skillsbench_meta = metadata.get("skillsbench") if isinstance(metadata.get("skillsbench"), dict) else {}
        if not skillsbench_meta:
            return _failure_result("missing_skillsbench_metadata")

        task_id = str(skillsbench_meta.get("task_id") or sample.get("id") or "unknown")
        image = str(skillsbench_meta.get("image") or "")
        workdir = str(skillsbench_meta.get("workdir") or sample.get("cwd") or "/app")
        tests_dir = Path(str(skillsbench_meta.get("tests_dir") or "")).expanduser()
        runtime_configs = _coerce_mapping(
            _coerce_mapping(sample.get("sandbox")).get("runtime_configs")
        )
        merged_runtime_configs = normalize_runtime_configs(runtime_configs)
        merged_runtime_configs.setdefault("docker_bin", str(params.get("docker_bin") or self._docker_bin))
        merged_runtime_configs.setdefault("image", image)
        merged_runtime_configs.setdefault("workdir", workdir)
        merged_runtime_configs.setdefault("exec_workdir", workdir)
        merged_runtime_configs.setdefault("command", ["sleep", "infinity"])
        merged_runtime_configs.setdefault("wait_for_ready", False)
        merged_runtime_configs["env"] = _merge_env_mappings(
            _coerce_mapping(merged_runtime_configs.get("env")),
            _coerce_mapping(skillsbench_meta.get("verifier_env")),
            _coerce_mapping(params.get("env")),
        )
        resources = _coerce_mapping(sample.get("resources"))
        timeout_sec = _coerce_timeout(
            params.get("timeout_sec") or skillsbench_meta.get("verifier_timeout_sec"),
            default=self._default_timeout_s,
        )

        if not image:
            return _failure_result("missing_image", task_id=task_id)
        if not tests_dir.exists():
            return _failure_result("missing_tests_dir", task_id=task_id)

        workspace_source = _resolve_workspace_source(payload, artifact_paths)
        if workspace_source is None or not workspace_source.exists():
            return _failure_result("missing_agent_workspace", task_id=task_id)

        verifier_workspace_dir = Path(
            artifact_paths.get("verifier_workspace_dir")
            or artifact_paths.get("verifier_workspace_path")
            or workspace_source.parent / "verifier-workspace"
        )
        verifier_logs_dir = Path(
            artifact_paths.get("verifier_logs_dir")
            or artifact_paths.get("log_dir")
            or verifier_workspace_dir.parent / "logs"
        )
        verifier_stdout_file = artifact_paths.get("verifier_stdout_file") or artifact_paths.get("stdout_path")
        verifier_stderr_file = artifact_paths.get("verifier_stderr_file") or artifact_paths.get("stderr_path")

        _prepare_workspace_copy(workspace_source, verifier_workspace_dir)
        verifier_logs_dir.mkdir(parents=True, exist_ok=True)

        try:
            ensure_docker_image(merged_runtime_configs)
        except Exception as exc:
            logger.warning("SkillsBench image build failed for {}: {}", task_id, exc)
            return _failure_result("image_build_failed", task_id=task_id, error=str(exc))

        container_name = _build_container_name(task_id)
        run_command = build_docker_run_command(
            image=image,
            container_name=container_name,
            runtime_configs={
                **merged_runtime_configs,
                "docker_bin": str(params.get("docker_bin") or self._docker_bin),
                "detach": True,
                "auto_remove": True,
                "command": ["sleep", "infinity"],
                "workdir": workdir,
            },
            resources=resources,
        )
        try:
            started = subprocess.run(
                run_command,
                capture_output=True,
                text=True,
                check=False,
            )
        except FileNotFoundError as exc:
            return _failure_result("docker_binary_not_found", task_id=task_id, error=str(exc))
        if started.returncode != 0:
            return _failure_result(
                "container_start_failed",
                task_id=task_id,
                error=(started.stderr or started.stdout).strip(),
            )
        container_id = started.stdout.strip() or container_name

        try:
            _docker_exec(
                docker_bin=str(params.get("docker_bin") or self._docker_bin),
                container=container_id,
                command=(
                    f"mkdir -p {workdir} {workdir}/output /tests /logs/verifier "
                    f"&& rm -rf /output && ln -sfn {workdir}/output /output"
                ),
                timeout_sec=min(timeout_sec, 120),
            )
            _docker_copy_to_container(
                docker_bin=str(params.get("docker_bin") or self._docker_bin),
                source=verifier_workspace_dir,
                container=container_id,
                destination=workdir,
            )
            _docker_copy_to_container(
                docker_bin=str(params.get("docker_bin") or self._docker_bin),
                source=tests_dir,
                container=container_id,
                destination="/tests",
            )
            completed = _docker_exec(
                docker_bin=str(params.get("docker_bin") or self._docker_bin),
                container=container_id,
                command="bash /tests/test.sh",
                timeout_sec=timeout_sec,
                workdir=workdir,
                capture_output=True,
            )
            _write_optional_text(verifier_stdout_file, completed.stdout)
            _write_optional_text(verifier_stderr_file, completed.stderr)
            _docker_copy_from_container(
                docker_bin=str(params.get("docker_bin") or self._docker_bin),
                container=container_id,
                source="/logs/verifier",
                destination=verifier_logs_dir,
            )
        except subprocess.TimeoutExpired:
            _write_optional_text(verifier_stderr_file, "verifier_timeout")
            return _failure_result(
                "verifier_timeout",
                task_id=task_id,
                log_dir=str(verifier_logs_dir),
                workspace_dir=str(verifier_workspace_dir),
            )
        except Exception as exc:
            logger.warning("SkillsBench verifier execution failed for {}: {}", task_id, exc)
            return _failure_result(
                "verifier_execution_failed",
                task_id=task_id,
                error=str(exc),
                log_dir=str(verifier_logs_dir),
                workspace_dir=str(verifier_workspace_dir),
            )
        finally:
            _stop_container(str(params.get("docker_bin") or self._docker_bin), container_id)

        reward = _read_reward(verifier_logs_dir / "reward.txt")
        resolved = reward is not None and reward >= 1.0
        return {
            "resolved": resolved,
            "score": reward,
            "task_id": task_id,
            "failure_reason": None if resolved else "tests_failed",
            "stdout_path": verifier_stdout_file,
            "stderr_path": verifier_stderr_file,
            "log_dir": str(verifier_logs_dir),
            "workspace_dir": str(verifier_workspace_dir),
            "reward": reward,
            "reward_file": str(verifier_logs_dir / "reward.txt"),
        }


def _resolve_workspace_source(payload: Dict[str, Any], artifact_paths: Mapping[str, Any]) -> Optional[Path]:
    for candidate in (
        payload.get("agent_workspace_dir"),
        artifact_paths.get("agent_workspace_dir"),
        artifact_paths.get("agent_workspace_path"),
    ):
        if isinstance(candidate, str) and candidate.strip():
            return Path(candidate).expanduser().resolve()
    return None


def _prepare_workspace_copy(source: Path, destination: Path) -> None:
    if destination.exists():
        shutil.rmtree(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(source, destination)


def _docker_copy_to_container(*, docker_bin: str, source: Path, container: str, destination: str) -> None:
    subprocess.run(
        [docker_bin, "cp", f"{source}/.", f"{container}:{destination}"],
        capture_output=True,
        text=True,
        check=True,
    )


def _docker_copy_from_container(*, docker_bin: str, container: str, source: str, destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    completed = subprocess.run(
        [docker_bin, "cp", f"{container}:{source}/.", str(destination)],
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.returncode != 0:
        logger.debug(
            "SkillsBench verifier logs copy skipped: {}",
            (completed.stderr or completed.stdout).strip(),
        )


def _docker_exec(
    *,
    docker_bin: str,
    container: str,
    command: str,
    timeout_sec: int,
    workdir: Optional[str] = None,
    capture_output: bool = False,
    strict: bool = True,
) -> subprocess.CompletedProcess[str]:
    args = [docker_bin, "exec"]
    if workdir:
        args.extend(["-w", workdir])
    args.extend([container, "/bin/sh", "-lc", command])
    completed = subprocess.run(
        args,
        capture_output=capture_output,
        text=True,
        timeout=timeout_sec,
        check=False,
    )
    if strict and completed.returncode != 0:
        raise RuntimeError((completed.stderr or completed.stdout).strip() or "docker_exec_failed")
    return completed


def _stop_container(docker_bin: str, container: str) -> None:
    subprocess.run(
        [docker_bin, "stop", "-t", "5", container],
        capture_output=True,
        text=True,
        check=False,
    )


def _read_reward(path: Path) -> Optional[float]:
    if not path.exists():
        return None
    try:
        return float(path.read_text(encoding="utf-8").strip())
    except (OSError, ValueError):
        return None


def _failure_result(reason: str, **extra: Any) -> Dict[str, Any]:
    payload = {
        "resolved": False,
        "score": 0.0,
        "failure_reason": reason,
    }
    payload.update({str(key): value for key, value in extra.items() if value is not None})
    return payload


def _coerce_mapping(value: Any) -> Dict[str, Any]:
    if isinstance(value, dict):
        return dict(value)
    return {}


def _coerce_timeout(value: Any, *, default: int) -> int:
    try:
        return max(1, int(float(value)))
    except (TypeError, ValueError):
        return int(default)


def _merge_env_mappings(*mappings: Dict[str, Any]) -> Dict[str, Any]:
    merged: Dict[str, Any] = {}
    for mapping in mappings:
        merged.update(_coerce_mapping(mapping))
    return merged


def _build_container_name(task_id: str) -> str:
    sanitized = "".join(ch if ch.isalnum() or ch in {"-", "_", "."} else "-" for ch in task_id)
    return f"gage-skillsbench-verify-{sanitized[:32]}-{uuid.uuid4().hex[:8]}"


def _write_optional_text(path: Optional[str], content: str) -> None:
    if not path:
        return
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(content, encoding="utf-8")


__all__ = ["SkillsBenchEvaluate"]
