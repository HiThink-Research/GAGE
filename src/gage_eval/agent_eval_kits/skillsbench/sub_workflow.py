"""SkillsBench benchmark-specific workflow helpers."""

from __future__ import annotations

from io import BytesIO
from pathlib import Path
import shlex
import shutil
import subprocess
import tarfile
from typing import Any, Dict

from gage_eval.agent_eval_kits.skillsbench.units import (
    extract_artifact_paths,
    resolve_agent_workspace_dir,
    resolve_cwd,
    resolve_env,
    resolve_instruction,
    resolve_sample_id,
    resolve_skillsbench_meta,
    serialize_scheduler_result,
)

_WORKSPACE_ARCHIVE_PATH = "/tmp/gage-skillsbench-workspace.tar"


def prepare_inputs(sample: dict, session: Any) -> Dict[str, Any]:
    """Prepare SkillsBench inputs for the installed-client scheduler."""

    metadata = dict(sample.get("metadata") or {})
    artifact_paths = extract_artifact_paths(getattr(session, "artifacts", None))
    metadata.setdefault("artifact_paths", artifact_paths)
    metadata.setdefault(
        "benchmark_kit_id",
        getattr(getattr(session, "plan", None), "benchmark_kit_id", "skillsbench"),
    )
    return {
        "sample_id": resolve_sample_id(sample),
        "sample": sample,
        "instruction": resolve_instruction(sample),
        "cwd": resolve_cwd(sample, session),
        "env": resolve_env(sample, session),
        "metadata": metadata,
        "artifacts": artifact_paths,
        "artifact_paths": artifact_paths,
        "session": {
            "run_id": getattr(getattr(session, "trace", None), "run_id", None),
            "benchmark_kit_id": getattr(getattr(session, "plan", None), "benchmark_kit_id", None),
        },
    }


def prepare_environment(sample: dict, session: Any, environment: Any, request: Any | None = None) -> Dict[str, Any]:
    """Prime the task container before invoking Codex."""

    skillsbench_meta = resolve_skillsbench_meta(sample)
    workdir = resolve_cwd(sample, session)
    timeout_sec = int(skillsbench_meta.get("agent_timeout_sec") or 1800)
    raw_code_home = (request.metadata or {}).get("code_home") if request is not None else None
    code_home = str(raw_code_home).strip() if raw_code_home is not None else ""
    if not code_home or code_home.lower() == "none":
        code_home = "/root/.codex"
    # SkillsBench runs inside task-provided images instead of the standard
    # gage-codex-sandbox image/entrypoint. We only inject the Codex runtime into
    # those images at build time, so auth material and a git baseline must be
    # normalized just before `codex exec`.
    command = f"""
CODEX_HOME={shlex.quote(code_home)}
# Recreate the auth bootstrap because task images do not inherit the standard
# bootstrap_codex.sh entrypoint used by the shared Codex sandbox image.
mkdir -p "$CODEX_HOME"
chmod 700 "$CODEX_HOME" || true
if [ ! -f "$CODEX_HOME/auth.json" ] && [ -n "${{GAGE_CODEX_HOST_HOME:-}}" ] && [ -d "${{GAGE_CODEX_HOST_HOME}}" ]; then
  cp -R "${{GAGE_CODEX_HOST_HOME}}/." "$CODEX_HOME/" 2>/dev/null || true
  chmod 700 "$CODEX_HOME" || true
  chmod 600 "$CODEX_HOME/auth.json" 2>/dev/null || true
fi
if [ ! -f "$CODEX_HOME/auth.json" ] && [ -n "${{OPENAI_API_KEY:-}}" ]; then
  umask 077
  printf '{{\\n  "OPENAI_API_KEY": "%s"\\n}}\\n' "$OPENAI_API_KEY" > "$CODEX_HOME/auth.json"
fi
# Many SkillsBench workdirs are plain task folders rather than git repos. Create
# a baseline commit so patch capture can later use git diff consistently.
if command -v git >/dev/null 2>&1; then
  if [ ! -d .git ]; then
    git init >/dev/null 2>&1 || true
    git config user.email gage@example.invalid >/dev/null 2>&1 || true
    git config user.name gage >/dev/null 2>&1 || true
    git add -A >/dev/null 2>&1 || true
    git commit -m baseline >/dev/null 2>&1 || true
  fi
fi
"""
    result = environment.exec(command, cwd=workdir, env=None, timeout_sec=timeout_sec)
    return {"prepare_environment_exit_code": getattr(result, "exit_code", 1)}


def capture_environment_artifacts(
    sample: dict,
    session: Any,
    environment: Any,
    client_result: Any,
    request: Any | None = None,
) -> Dict[str, Any]:
    """Capture the mutated workspace from the running task container."""

    workdir = resolve_cwd(sample, session)
    skillsbench_meta = resolve_skillsbench_meta(sample)
    timeout_sec = int(skillsbench_meta.get("agent_timeout_sec") or 1800)
    runtime_configs = ((sample.get("sandbox") or {}).get("runtime_configs") or {})
    docker_bin = str(runtime_configs.get("docker_bin") or "docker")
    workspace_dir = resolve_agent_workspace_dir(getattr(session, "artifacts", None))
    if workspace_dir is None:
        return {}
    workspace_dir.mkdir(parents=True, exist_ok=True)
    direct_copy_error = _copy_workspace_via_docker_cp(
        environment=environment,
        docker_bin=docker_bin,
        workdir=workdir,
        workspace_dir=workspace_dir,
    )
    if direct_copy_error is None:
        return {"agent_workspace_dir": str(workspace_dir)}
    command = f"rm -f {shlex.quote(_WORKSPACE_ARCHIVE_PATH)} && tar -C {shlex.quote(workdir)} -cf {shlex.quote(_WORKSPACE_ARCHIVE_PATH)} ."
    result = environment.exec(command, cwd=workdir, env=None, timeout_sec=timeout_sec)
    if int(getattr(result, "exit_code", 1) or 1) != 0:
        error_detail = str(
            getattr(result, "stderr", "") or getattr(result, "stdout", "") or ""
        ).strip()
        if direct_copy_error:
            error_detail = f"{direct_copy_error}; {error_detail}".strip("; ")
        return {
            "agent_workspace_capture_error": error_detail or "capture_failed",
        }
    payload = environment.read_file(_WORKSPACE_ARCHIVE_PATH)
    _extract_tar_bytes(payload, workspace_dir)
    return {"agent_workspace_dir": str(workspace_dir)}


def finalize_result(sample: dict, scheduler_result: Any, artifacts: Any) -> Dict[str, Any]:
    """Normalize SkillsBench scheduler output for reporting."""

    artifact_paths = extract_artifact_paths(artifacts)
    payload = serialize_scheduler_result(scheduler_result)
    if isinstance(payload.get("artifacts"), dict):
        payload["artifacts"] = dict(payload["artifacts"])
    payload.update(
        {
            "sample_id": resolve_sample_id(sample),
            "sample_metadata": dict(sample.get("metadata") or {}),
            "artifacts": artifact_paths,
            "artifact_paths": artifact_paths,
            "patch_path": payload.get("patch_path") or artifact_paths.get("patch_path") or artifact_paths.get("patch_file"),
            "stdout_path": payload.get("stdout_path") or artifact_paths.get("stdout_path") or artifact_paths.get("stdout_file"),
            "trajectory_path": payload.get("trajectory_path") or artifact_paths.get("trajectory_path") or artifact_paths.get("trajectory_file"),
        }
    )
    return payload


def _extract_tar_bytes(payload: bytes, target_dir: Path) -> None:
    with tarfile.open(fileobj=BytesIO(payload), mode="r:") as archive:
        for member in archive.getmembers():
            resolved = (target_dir / member.name).resolve()
            if not str(resolved).startswith(str(target_dir.resolve())):
                raise RuntimeError(f"unsafe_tar_member: {member.name}")
        archive.extractall(target_dir)


def _copy_workspace_via_docker_cp(
    *,
    environment: Any,
    docker_bin: str,
    workdir: str,
    workspace_dir: Path,
) -> str | None:
    handle_getter = getattr(environment, "runtime_handle", None)
    if not callable(handle_getter):
        return "missing_runtime_handle"
    runtime_handle = dict(handle_getter() or {})
    container = runtime_handle.get("container_id") or runtime_handle.get("container_name")
    if not container:
        return "missing_container_handle"
    shutil.rmtree(workspace_dir, ignore_errors=True)
    workspace_dir.mkdir(parents=True, exist_ok=True)
    completed = subprocess.run(
        [docker_bin, "cp", f"{container}:{workdir}/.", str(workspace_dir)],
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.returncode == 0:
        return None
    return str(completed.stderr or completed.stdout or "docker_cp_failed").strip()


__all__ = [
    "capture_environment_artifacts",
    "finalize_result",
    "prepare_environment",
    "prepare_inputs",
]
