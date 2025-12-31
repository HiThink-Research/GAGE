"""SWE-bench Pro judge implementation using local Docker."""

from __future__ import annotations

import ast
import json
import os
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from loguru import logger

from gage_eval.registry import registry
from gage_eval.role.judge.base import JudgeImplementation
from gage_eval.utils.swebench import get_dockerhub_image_uri, resolve_docker_platform


@registry.asset(
    "judge_impls",
    "swebench_docker",
    desc="SWE-bench Pro docker judge implementation",
    tags=("swebench", "docker"),
)
class SwebenchDocker(JudgeImplementation):
    def __init__(
        self,
        *,
        scripts_dir: Optional[str] = None,
        dockerhub_username: str = "jefzda",
        registry_prefix: Optional[str] = None,
        block_network: bool = True,
        test_timeout_s: int = 900,
        mem_limit: Optional[str] = None,
        pids_limit: Optional[int] = None,
        allow_pull: bool = True,
        image_lock_dir: Optional[str] = None,
        dockerfiles_dir: Optional[str] = None,
        docker_platform: Optional[str] = None,
    ) -> None:
        self._scripts_dir = Path(scripts_dir) if scripts_dir else _default_scripts_dir()
        self._dockerhub_username = dockerhub_username
        self._registry_prefix = registry_prefix
        self._block_network = bool(block_network)
        self._test_timeout_s = max(1, int(test_timeout_s))
        self._mem_limit = mem_limit
        self._pids_limit = pids_limit
        self._allow_pull = bool(allow_pull)
        self._image_lock_dir = Path(image_lock_dir) if image_lock_dir else None
        self._dockerfiles_dir = Path(dockerfiles_dir) if dockerfiles_dir else None
        self._docker_platform = docker_platform

    def invoke(self, payload: Dict[str, Any], state: Any = None) -> Dict[str, Any]:
        params = payload.get("params") or {}
        sample = payload.get("sample") or {}
        model_output = payload.get("model_output") or {}

        patch = _resolve_patch(model_output)
        if not patch:
            return {"resolved": False, "failure_reason": "missing_patch"}

        instance_id = _resolve_instance_id(sample)
        repo = _get_meta(sample, "repo")
        base_commit = _get_meta(sample, "base_commit")
        if not instance_id or not repo or not base_commit:
            return {"resolved": False, "failure_reason": "missing_metadata"}

        run_id = _resolve_run_id(payload)
        log_dir = _host_log_dir(run_id, instance_id)
        workspace_dir = log_dir / "workspace"
        workspace_dir.mkdir(parents=True, exist_ok=True)

        scripts_dir = Path(params.get("scripts_dir", self._scripts_dir))
        run_script_dir = _resolve_script_dir(scripts_dir, instance_id)

        try:
            run_script = _read_text(run_script_dir / "run_script.sh")
            parser_script = _read_text(run_script_dir / "parser.py")
        except FileNotFoundError as exc:
            logger.error("SWE-bench run_scripts missing: {}", exc)
            return {"resolved": False, "failure_reason": "missing_run_scripts"}

        test_patch = _get_meta(sample, "test_patch")
        entryscript = _create_entryscript(
            sample=sample,
            base_commit=base_commit,
            dockerfiles_dir=params.get("dockerfiles_dir") or self._dockerfiles_dir,
            test_patch=bool(test_patch),
        )

        _write_text(workspace_dir / "patch.diff", patch)
        _write_text(workspace_dir / "run_script.sh", run_script)
        _write_text(workspace_dir / "parser.py", parser_script)
        _write_text(workspace_dir / "entryscript.sh", entryscript)
        if test_patch:
            _write_text(workspace_dir / "test_patch.diff", test_patch)

        image_uri = _resolve_image_uri(
            sample=sample,
            params=params,
            dockerhub_username=self._dockerhub_username,
            registry_prefix=self._registry_prefix,
        )

        run_result = self._run_container(
            image_uri=image_uri,
            workspace_dir=workspace_dir,
            params=params,
            run_id=run_id,
            instance_id=instance_id,
        )
        if run_result.get("status") != "ok":
            return {
                "resolved": False,
                "failure_reason": run_result.get("failure_reason", "container_error"),
                "log_dir": str(log_dir),
            }

        output = _load_output_json(workspace_dir / "output.json")
        if output is None:
            return {
                "resolved": False,
                "failure_reason": "missing_output",
                "log_dir": str(log_dir),
            }

        resolved, failure_reason = _evaluate_resolution(
            output,
            _parse_list(_get_meta(sample, "fail_to_pass")),
            _parse_list(_get_meta(sample, "pass_to_pass")),
        )
        return {
            "resolved": resolved,
            "failure_reason": failure_reason,
            "tests": output.get("tests", []),
            "log_dir": str(log_dir),
        }

    def _run_container(
        self,
        *,
        image_uri: str,
        workspace_dir: Path,
        params: Dict[str, Any],
        run_id: str,
        instance_id: str,
    ) -> Dict[str, str]:
        client = _get_docker_client()
        docker_platform = resolve_docker_platform(params.get("docker_platform", self._docker_platform))
        if not self._ensure_image(client, image_uri, docker_platform=docker_platform):
            return {"status": "error", "failure_reason": "image_missing"}

        labels = {
            "gage_eval.run_id": run_id,
            "swebench.instance_id": instance_id,
        }
        container = None
        try:
            run_kwargs = {
                "command": ["-c", "bash /workspace/entryscript.sh"],
                "entrypoint": "/bin/bash",
                "volumes": {str(workspace_dir): {"bind": "/workspace", "mode": "rw"}},
                "detach": True,
                "remove": True,
                "network_mode": "none" if params.get("block_network", self._block_network) else None,
                "mem_limit": params.get("mem_limit", self._mem_limit),
                "pids_limit": params.get("pids_limit", self._pids_limit),
                "labels": labels,
            }
            if docker_platform:
                run_kwargs["platform"] = docker_platform
            container = client.containers.run(image_uri, **run_kwargs)
            container.wait(timeout=int(params.get("test_timeout_s", self._test_timeout_s)))
            return {"status": "ok"}
        except Exception as exc:
            logger.warning("SWE-bench docker run failed: {}", exc)
            if container is not None:
                try:
                    container.kill()
                except Exception:
                    pass
            return {"status": "error", "failure_reason": "test_execution_error"}
        finally:
            if container is not None:
                try:
                    container.remove(force=True)
                except Exception:
                    pass

    def _ensure_image(self, client, image_uri: str, *, docker_platform: Optional[str]) -> bool:
        if _has_image(client, image_uri):
            return True
        if not self._allow_pull:
            return False
        lock_dir = self._image_lock_dir or Path(os.environ.get("GAGE_EVAL_IMAGE_LOCK_DIR", ".cache/swebench_images"))
        lock_dir.mkdir(parents=True, exist_ok=True)
        lock_path = lock_dir / f"{_safe_lock_name(image_uri)}.lock"
        try:
            from filelock import FileLock  # type: ignore
        except Exception:
            return self._pull_image(client, image_uri, docker_platform=docker_platform)
        with FileLock(str(lock_path)):
            if _has_image(client, image_uri):
                return True
            return self._pull_image(client, image_uri, docker_platform=docker_platform)

    def _pull_image(self, client, image_uri: str, *, docker_platform: Optional[str]) -> bool:
        try:
            if docker_platform:
                client.images.pull(image_uri, platform=docker_platform)
            else:
                client.images.pull(image_uri)
        except Exception as exc:
            logger.warning("Failed to pull image {}: {}", image_uri, exc)
            return _has_image(client, image_uri)
        return True


def _default_scripts_dir() -> Path:
    repo_root = Path(__file__).resolve().parents[4]
    return repo_root / "third_party" / "swebench_pro" / "run_scripts"


def _resolve_script_dir(scripts_dir: Path, instance_id: str) -> Path:
    candidate = scripts_dir / instance_id
    if candidate.exists():
        return candidate
    if not instance_id.startswith("instance_"):
        prefixed = scripts_dir / f"instance_{instance_id}"
        if prefixed.exists():
            return prefixed
    return candidate


def _resolve_patch(model_output: Dict[str, Any]) -> str:
    for key in ("patch", "diff", "answer", "text", "content"):
        value = model_output.get(key)
        if isinstance(value, str) and value.strip():
            return value
    message = model_output.get("message")
    if isinstance(message, dict):
        content = message.get("content")
        if isinstance(content, str) and content.strip():
            return content
    return ""


def _resolve_instance_id(sample: Dict[str, Any]) -> str:
    metadata = sample.get("metadata") or {}
    return (
        metadata.get("instance_id")
        or sample.get("instance_id")
        or sample.get("id")
        or ""
    )


def _resolve_image_uri(
    *,
    sample: Dict[str, Any],
    params: Dict[str, Any],
    dockerhub_username: str,
    registry_prefix: Optional[str],
) -> str:
    metadata = sample.get("metadata") or {}
    image_uri = params.get("image_uri") or metadata.get("image_uri") or sample.get("image_uri")
    if not image_uri:
        instance_id = _resolve_instance_id(sample)
        repo = _get_meta(sample, "repo")
        image_uri = get_dockerhub_image_uri(instance_id, dockerhub_username, repo)
    if registry_prefix:
        image_name = image_uri.split("/", 1)[-1]
        image_uri = f"{registry_prefix.rstrip('/')}/{image_name}"
    return image_uri


def _get_meta(sample: Dict[str, Any], key: str) -> Any:
    metadata = sample.get("metadata") or {}
    return metadata.get(key) or sample.get(key)


def _resolve_run_id(payload: Dict[str, Any]) -> str:
    trace = payload.get("trace")
    if hasattr(trace, "run_id"):
        return str(trace.run_id)
    return os.environ.get("GAGE_EVAL_RUN_ID", "") or "unknown"


def _host_log_dir(run_id: str, instance_id: str) -> Path:
    save_root = Path(os.environ.get("GAGE_EVAL_SAVE_DIR", "./runs")).expanduser().resolve()
    return save_root / run_id / "logs" / instance_id


def _create_entryscript(
    *,
    sample: Dict[str, Any],
    base_commit: str,
    dockerfiles_dir: Optional[Path],
    test_patch: bool,
) -> str:
    before_repo_set_cmd = _get_meta(sample, "before_repo_set_cmd") or ""
    before_repo_set_cmd = before_repo_set_cmd.strip()
    if before_repo_set_cmd:
        before_repo_set_cmd = before_repo_set_cmd.split("\n")[-1]
    selected = _parse_list(_get_meta(sample, "selected_test_files_to_run"))
    selected_arg = ",".join(selected) if selected else ""

    env_cmds = ""
    if dockerfiles_dir:
        env_cmds = _extract_env_exports(dockerfiles_dir, _resolve_instance_id(sample))

    lines = []
    if env_cmds:
        lines.append(env_cmds)
    lines.extend(
        [
            "cd /app",
            f"git reset --hard {base_commit}",
            f"git checkout {base_commit}",
            "git apply -v /workspace/patch.diff",
        ]
    )
    if test_patch:
        lines.append("git apply -v /workspace/test_patch.diff")
    if before_repo_set_cmd:
        lines.append(before_repo_set_cmd)
    if selected_arg:
        lines.append(f"bash /workspace/run_script.sh {selected_arg} > /workspace/stdout.log 2> /workspace/stderr.log")
    else:
        lines.append("bash /workspace/run_script.sh > /workspace/stdout.log 2> /workspace/stderr.log")
    lines.append("python /workspace/parser.py /workspace/stdout.log /workspace/stderr.log /workspace/output.json")
    return "\n".join(lines) + "\n"


def _extract_env_exports(dockerfiles_dir: Path, instance_id: str) -> str:
    env_cmds: List[str] = []
    for kind in ("base_dockerfile", "instance_dockerfile"):
        dockerfile_path = dockerfiles_dir / kind / instance_id / "Dockerfile"
        if not dockerfile_path.exists():
            continue
        for line in dockerfile_path.read_text(encoding="utf-8", errors="replace").splitlines():
            line = line.strip()
            if line.startswith("ENV"):
                env_cmds.append(line.replace("ENV", "export", 1))
    return "\n".join(env_cmds)


def _load_output_json(path: Path) -> Optional[Dict[str, Any]]:
    try:
        with path.open("r", encoding="utf-8", errors="replace") as handle:
            return json.load(handle)
    except Exception:
        return None


def _evaluate_resolution(
    output: Dict[str, Any],
    fail_to_pass: Sequence[str],
    pass_to_pass: Sequence[str],
) -> Tuple[bool, Optional[str]]:
    tests = output.get("tests") or []
    passed = {t.get("name") for t in tests if t.get("status") == "PASSED"}
    target = set(fail_to_pass) | set(pass_to_pass)
    if target and not target.issubset(passed):
        return False, "assertion_error"
    if not target:
        return False, "missing_targets"
    return True, None


def _parse_list(value: Any) -> List[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(v) for v in value]
    if isinstance(value, str):
        raw = value.strip()
        if not raw:
            return []
        try:
            parsed = ast.literal_eval(raw)
            if isinstance(parsed, list):
                return [str(v) for v in parsed]
        except Exception:
            pass
        if "\n" in raw:
            return [ln.strip() for ln in raw.splitlines() if ln.strip()]
        if "," in raw:
            return [seg.strip() for seg in raw.split(",") if seg.strip()]
        return [raw]
    return [str(value)]


def _read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace")


def _write_text(path: Path, content: str) -> None:
    path.write_text(content or "", encoding="utf-8")


def _get_docker_client():
    try:
        import docker  # type: ignore
    except Exception as exc:  # pragma: no cover - optional dependency
        raise RuntimeError("docker SDK is required for swebench_docker") from exc
    return docker.from_env()


def _has_image(client, image_uri: str) -> bool:
    try:
        client.images.get(image_uri)
        return True
    except Exception:
        return False


def _safe_lock_name(image_uri: str) -> str:
    return image_uri.replace("/", "_").replace(":", "_")
