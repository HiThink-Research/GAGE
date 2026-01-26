"""SWE-bench Pro judge implementation using local Docker."""

from __future__ import annotations

import ast
import base64
import json
import os
import re
import shlex
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from loguru import logger

from gage_eval.registry import registry
from gage_eval.evaluation.sample_envelope import resolve_model_output
from gage_eval.role.judge.base import JudgeImplementation
from gage_eval.utils.swebench import get_dockerhub_image_uri, resolve_docker_platform


class _SandboxFallback(RuntimeError):
    """Signals that sandbox-based execution should fall back to Docker SDK."""


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
        model_output = resolve_model_output(sample, payload.get("model_output"))
        sandbox_provider = payload.get("sandbox_provider")

        patch = _clean_patch_content(_resolve_patch(model_output))
        if not patch and sandbox_provider is not None:
            patch = _load_submission_patch(sandbox_provider, payload, sample)
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
        _write_text(workspace_dir / "patch.diff", patch)
        _write_text(workspace_dir / "run_script.sh", run_script)
        _write_text(workspace_dir / "parser.py", parser_script)
        if test_patch:
            _write_text(workspace_dir / "test_patch.diff", test_patch)

        if sandbox_provider is not None:
            try:
                return self._run_with_sandbox(
                    sandbox_provider=sandbox_provider,
                    params=params,
                    sample=sample,
                    patch=patch,
                    run_script=run_script,
                    parser_script=parser_script,
                    test_patch=test_patch,
                    base_commit=base_commit,
                    run_id=run_id,
                    instance_id=instance_id,
                    log_dir=log_dir,
                    workspace_dir=workspace_dir,
                )
            except _SandboxFallback as exc:
                logger.warning("SWE-bench sandbox path fallback: {}", exc)

        entryscript = _create_entryscript(
            sample=sample,
            base_commit=base_commit,
            dockerfiles_dir=params.get("dockerfiles_dir") or self._dockerfiles_dir,
            test_patch=bool(test_patch),
            run_script_path="/workspace/run_script.sh",
            parser_path="/workspace/parser.py",
        )
        _write_text(workspace_dir / "entryscript.sh", entryscript)

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

        patch_status = _load_patch_status_file(workspace_dir / "patch_apply_status.json")
        if patch_status:
            failure_reason = _resolve_patch_failure_reason(patch_status)
            if failure_reason:
                return {
                    "resolved": False,
                    "failure_reason": failure_reason,
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

    def _run_with_sandbox(
        self,
        *,
        sandbox_provider: Any,
        params: Dict[str, Any],
        sample: Dict[str, Any],
        patch: str,
        run_script: str,
        parser_script: str,
        test_patch: Optional[str],
        base_commit: str,
        run_id: str,
        instance_id: str,
        log_dir: Path,
        workspace_dir: Path,
    ) -> Dict[str, Any]:
        handle = getattr(sandbox_provider, "get_handle", lambda: None)()
        sandbox = getattr(handle, "sandbox", None) if handle else None
        if sandbox is None:
            raise _SandboxFallback("sandbox_unavailable")
        config = getattr(handle, "config", {}) or {}
        runtime_configs = dict(config.get("runtime_configs") or {})
        run_scripts_mount = _resolve_run_scripts_mount(runtime_configs.get("volumes"))

        use_volumes = params.get("use_volumes")
        if use_volumes is None:
            use_volumes = bool(run_scripts_mount)
        if use_volumes and not run_scripts_mount:
            use_volumes = False

        run_script_path = "/workspace/run_script.sh"
        parser_path = "/workspace/parser.py"
        if use_volumes:
            run_script_path = f"{run_scripts_mount}/{instance_id}/run_script.sh"
            parser_path = f"{run_scripts_mount}/{instance_id}/parser.py"

        entryscript = _create_entryscript(
            sample=sample,
            base_commit=base_commit,
            dockerfiles_dir=params.get("dockerfiles_dir") or self._dockerfiles_dir,
            test_patch=bool(test_patch),
            run_script_path=run_script_path,
            parser_path=parser_path,
        )
        _write_text(workspace_dir / "entryscript.sh", entryscript)

        self._ensure_workspace(sandbox)
        self._write_sandbox_file(sandbox, "/workspace/patch.diff", patch)
        self._write_sandbox_file(sandbox, "/workspace/entryscript.sh", entryscript)
        if test_patch:
            self._write_sandbox_file(sandbox, "/workspace/test_patch.diff", test_patch)
        if not use_volumes:
            self._write_sandbox_file(sandbox, "/workspace/run_script.sh", run_script)
            self._write_sandbox_file(sandbox, "/workspace/parser.py", parser_script)

        try:
            sandbox.exec(
                "bash /workspace/entryscript.sh",
                timeout=int(params.get("test_timeout_s", self._test_timeout_s)),
            )
        except Exception as exc:
            raise _SandboxFallback("sandbox_exec_failed") from exc

        patch_status_bytes = self._read_sandbox_file(sandbox, "/workspace/patch_apply_status.json")
        if patch_status_bytes:
            patch_status = _parse_patch_status(patch_status_bytes)
            _write_text(
                workspace_dir / "patch_apply_status.json",
                patch_status_bytes.decode("utf-8", errors="replace"),
            )
            log_path = patch_status.get("log")
            if log_path:
                log_bytes = self._read_sandbox_file(sandbox, str(log_path))
                if log_bytes is not None:
                    _write_text(
                        workspace_dir / Path(str(log_path)).name,
                        log_bytes.decode("utf-8", errors="replace"),
                    )
            failure_reason = _resolve_patch_failure_reason(patch_status)
            if failure_reason:
                return {
                    "resolved": False,
                    "failure_reason": failure_reason,
                    "log_dir": str(log_dir),
                }

        stdout_bytes = self._read_sandbox_file(sandbox, "/workspace/stdout.log")
        stderr_bytes = self._read_sandbox_file(sandbox, "/workspace/stderr.log")
        output_bytes = self._read_sandbox_file(sandbox, "/workspace/output.json")
        if stdout_bytes is not None:
            _write_text(workspace_dir / "stdout.log", stdout_bytes.decode("utf-8", errors="replace"))
        if stderr_bytes is not None:
            _write_text(workspace_dir / "stderr.log", stderr_bytes.decode("utf-8", errors="replace"))
        if output_bytes is None:
            return {
                "resolved": False,
                "failure_reason": "missing_output",
                "log_dir": str(log_dir),
            }
        try:
            output = json.loads(output_bytes.decode("utf-8", errors="replace"))
        except Exception:
            return {
                "resolved": False,
                "failure_reason": "invalid_output",
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

    @staticmethod
    def _ensure_workspace(sandbox: Any) -> None:
        try:
            result = sandbox.exec("mkdir -p /workspace", timeout=10)
        except Exception as exc:
            raise _SandboxFallback("sandbox_workspace_failed") from exc
        if getattr(result, "exit_code", 1) not in (0, None):
            raise _SandboxFallback("sandbox_workspace_failed")

    @staticmethod
    def _write_sandbox_file(sandbox: Any, path: str, content: str) -> None:
        payload = content.encode("utf-8") if isinstance(content, str) else content
        writer = getattr(sandbox, "write_file", None)
        if callable(writer):
            try:
                writer(path, payload)
                return
            except NotImplementedError:
                pass
            except Exception as exc:
                raise _SandboxFallback("sandbox_write_failed") from exc
        command = _build_write_command(path, payload)
        try:
            result = sandbox.exec(command, timeout=30)
        except Exception as exc:
            raise _SandboxFallback("sandbox_write_failed") from exc
        if getattr(result, "exit_code", 1) not in (0, None):
            raise _SandboxFallback("sandbox_write_failed")

    @staticmethod
    def _read_sandbox_file(sandbox: Any, path: str) -> Optional[bytes]:
        reader = getattr(sandbox, "read_file", None)
        if callable(reader):
            try:
                return reader(path)
            except NotImplementedError:
                pass
            except Exception as exc:
                if _is_missing_file_error(exc):
                    return None
                raise _SandboxFallback("sandbox_read_failed") from exc
        try:
            result = sandbox.exec(f"cat {shlex.quote(path)}", timeout=10)
        except Exception as exc:
            raise _SandboxFallback("sandbox_read_failed") from exc
        if getattr(result, "exit_code", 1) not in (0, None):
            return None
        return str(result.stdout or "").encode("utf-8")

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


def _is_missing_file_error(exc: Exception) -> bool:
    message = str(exc).lower()
    tokens = (
        "no such file",
        "not found",
        "does not exist",
        "cannot stat",
        "file not found",
    )
    return any(token in message for token in tokens)


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
    trace_patch = _resolve_patch_from_agent_trace(model_output)
    if trace_patch:
        if not str(model_output.get("answer") or "").strip():
            model_output["answer"] = trace_patch
        return trace_patch
    return ""


def _resolve_patch_from_agent_trace(model_output: Dict[str, Any]) -> str:
    agent_trace = model_output.get("agent_trace")
    if not isinstance(agent_trace, list):
        return ""
    for entry in reversed(agent_trace):
        if not isinstance(entry, dict):
            continue
        if entry.get("name") != "submit_patch_tool":
            continue
        output = entry.get("output")
        patch = _extract_tool_stdout(output)
        if patch.strip():
            return patch
    return ""


def _extract_tool_stdout(output: Any) -> str:
    if isinstance(output, dict):
        for key in ("stdout", "content", "text", "answer", "patch", "diff"):
            value = output.get(key)
            if isinstance(value, str) and value.strip():
                return value
        nested = output.get("output")
        if nested is not None and nested is not output:
            return _extract_tool_stdout(nested)
    if isinstance(output, str):
        return output
    return ""


def _clean_patch_content(raw: str) -> str:
    if not raw:
        return ""
    cleaned = raw.strip()
    fenced = _extract_code_block(cleaned)
    if fenced:
        cleaned = fenced.strip()
    cleaned = _strip_apply_patch_markers(cleaned)
    diff_index = cleaned.find("diff --git")
    if diff_index != -1:
        cleaned = cleaned[diff_index:]
    if _has_diff_markers(cleaned):
        cleaned = _trim_diff_tail(cleaned)
        cleaned = _normalize_hunk_context_lines(cleaned)
    cleaned = cleaned.strip()
    if cleaned and not cleaned.endswith("\n"):
        cleaned += "\n"
    return cleaned


def _load_submission_patch(
    sandbox_provider: Any,
    payload: Dict[str, Any],
    sample: Dict[str, Any],
) -> str:
    handle = getattr(sandbox_provider, "get_handle", lambda: None)()
    sandbox = getattr(handle, "sandbox", None) if handle else None
    if sandbox is None:
        return ""
    raw = _read_submission_patch(sandbox)
    if not raw:
        return ""
    cleaned = _clean_patch_content(raw)
    if cleaned:
        _emit_patch_fallback_event(payload, sample)
    return cleaned


def _read_submission_patch(sandbox: Any) -> str:
    reader = getattr(sandbox, "read_file", None)
    if callable(reader):
        try:
            payload = reader("/workspace/submission.patch")
        except Exception:
            return ""
        if isinstance(payload, (bytes, bytearray)):
            return payload.decode("utf-8", errors="replace")
        return str(payload)
    try:
        result = sandbox.exec("cat /workspace/submission.patch", timeout=5)
    except Exception:
        return ""
    if getattr(result, "exit_code", 1) != 0:
        return ""
    return str(getattr(result, "stdout", ""))


def _emit_patch_fallback_event(payload: Dict[str, Any], sample: Dict[str, Any]) -> None:
    trace = payload.get("trace")
    if not hasattr(trace, "emit"):
        return
    instance_id = _resolve_instance_id(sample)
    event_payload = {"source": "submission_patch", "path": "/workspace/submission.patch"}
    if instance_id:
        event_payload["instance_id"] = instance_id
    sample_id = sample.get("id") or instance_id
    trace.emit("swebench_patch_fallback", event_payload, sample_id=sample_id)
    logger.warning("SWE-bench patch fallback used for instance_id={}", instance_id)


_CODE_FENCE_RE = re.compile(
    r"```(?P<lang>[a-zA-Z0-9_-]*)\s*\n(?P<body>.*?)(?:\n)?```",
    re.DOTALL,
)
_UNCLOSED_CODE_FENCE_RE = re.compile(
    r"```(?P<lang>[a-zA-Z0-9_-]*)\s*\n(?P<body>.*)\Z",
    re.DOTALL,
)


def _extract_code_block(text: str) -> Optional[str]:
    matches = list(_CODE_FENCE_RE.finditer(text))
    if matches:
        for match in matches:
            lang = match.group("lang").strip().lower()
            if lang in {"diff", "patch"}:
                return match.group("body")
        return matches[0].group("body")
    match = _UNCLOSED_CODE_FENCE_RE.search(text)
    if match:
        return match.group("body")
    return None


def _strip_apply_patch_markers(text: str) -> str:
    lines = text.splitlines()
    begin_idx = None
    end_idx = None
    for idx, line in enumerate(lines):
        if line.strip().startswith("*** Begin Patch"):
            begin_idx = idx
            break
    if begin_idx is not None:
        for idx in range(begin_idx + 1, len(lines)):
            if lines[idx].strip().startswith("*** End Patch"):
                end_idx = idx
                break
    if begin_idx is not None and end_idx is not None and end_idx > begin_idx:
        return "\n".join(lines[begin_idx + 1 : end_idx])
    return "\n".join(
        line
        for line in lines
        if not line.strip().startswith("*** Begin Patch")
        and not line.strip().startswith("*** End Patch")
    )


def _normalize_hunk_context_lines(text: str) -> str:
    lines = text.splitlines()
    if not lines:
        return text
    output: List[str] = []
    in_hunk = False
    for line in lines:
        if line.startswith("diff --git "):
            in_hunk = False
            output.append(line)
            continue
        if line.startswith("@@"):
            in_hunk = True
            output.append(line)
            continue
        if in_hunk:
            if line.startswith((" ", "+", "-", "\\")):
                output.append(line)
            else:
                output.append(f" {line}")
            continue
        output.append(line)
    return "\n".join(output)


_DIFF_PREFIXES = (
    "diff --git ",
    "index ",
    "--- ",
    "+++ ",
    "@@ ",
    "new file mode ",
    "deleted file mode ",
    "rename from ",
    "rename to ",
    "similarity index ",
    "dissimilarity index ",
    "old mode ",
    "new mode ",
    "GIT binary patch",
    "Binary files ",
)


def _has_diff_markers(text: str) -> bool:
    for line in text.splitlines():
        if line.startswith(("diff --git ", "--- ", "+++ ", "@@ ")):
            return True
    return False


def _trim_diff_tail(text: str) -> str:
    lines = text.splitlines()
    if not lines:
        return text
    last_idx = None
    for idx in range(len(lines) - 1, -1, -1):
        line = lines[idx]
        if line.startswith(_DIFF_PREFIXES) or line[:1] in (" ", "+", "-", "\\"):
            last_idx = idx
            break
    if last_idx is None:
        return text
    return "\n".join(lines[: last_idx + 1])




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
    run_script_path: str,
    parser_path: str,
) -> str:
    if dockerfiles_dir is not None and not isinstance(dockerfiles_dir, Path):
        dockerfiles_dir = Path(str(dockerfiles_dir))
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
            "PATCH_STATUS=/workspace/patch_apply_status.json",
            "apply_patch() {",
            "  patch_path=\"$1\"",
            "  log_path=\"$2\"",
            "  label=\"$3\"",
            "  if [ ! -f \"$patch_path\" ]; then",
            "    return 0",
            "  fi",
            "  # Attempt 1: Recount + ignore whitespace",
            "  if ! git apply --recount --ignore-space-change --ignore-whitespace -v \"$patch_path\" > \"$log_path\" 2>&1; then",
            "    echo \"Recount+whitespace-ignored apply failed, trying patch tool...\" >> \"$log_path\"",
            "    # Attempt 2: GNU patch fallback",
            "      if command -v patch >/dev/null 2>&1; then",
            "        if patch -p1 --force --ignore-whitespace --batch -i \"$patch_path\" >> \"$log_path\" 2>&1; then",
            "          return 0",
            "        fi",
            "      fi",
            "      # All attempts failed",
            "      printf '{\"status\":\"failed\",\"patch\":\"%s\",\"log\":\"%s\"}\\n' \"$label\" \"$log_path\" > \"$PATCH_STATUS\"",
            "      exit 0",
            "  fi",
            "}",
        ]
    )
    lines.extend(
        [
            "cd /app",
            f"git reset --hard {base_commit}",
            f"git checkout {base_commit}",
            "apply_patch /workspace/patch.diff /workspace/patch_apply.log patch.diff",
        ]
    )
    if test_patch:
        lines.append("apply_patch /workspace/test_patch.diff /workspace/test_patch_apply.log test_patch.diff")
    if before_repo_set_cmd:
        lines.append(before_repo_set_cmd)
    if selected_arg:
        lines.append(
            f"bash {shlex.quote(run_script_path)} {selected_arg} > /workspace/stdout.log 2> /workspace/stderr.log"
        )
    else:
        lines.append(f"bash {shlex.quote(run_script_path)} > /workspace/stdout.log 2> /workspace/stderr.log")
    lines.extend(
        [
            "PYTHON_BIN=python",
            "if ! command -v \"$PYTHON_BIN\" >/dev/null 2>&1; then",
            "  if command -v python3 >/dev/null 2>&1; then",
            "    PYTHON_BIN=python3",
            "  else",
            "    echo \"python_missing\" >> /workspace/stderr.log",
            "    exit 0",
            "  fi",
            "fi",
            f"$PYTHON_BIN {shlex.quote(parser_path)} /workspace/stdout.log /workspace/stderr.log /workspace/output.json",
        ]
    )
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


def _parse_patch_status(payload: bytes) -> Dict[str, Any]:
    try:
        return json.loads(payload.decode("utf-8", errors="replace"))
    except Exception:
        return {"status": "failed"}


def _load_patch_status_file(path: Path) -> Optional[Dict[str, Any]]:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8", errors="replace"))
    except Exception:
        return {"status": "failed"}


def _resolve_patch_failure_reason(status: Dict[str, Any]) -> Optional[str]:
    if status.get("status") != "failed":
        return None
    patch_label = str(status.get("patch") or "")
    if patch_label == "test_patch.diff":
        return "test_patch_apply_failed"
    return "patch_apply_failed"


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


def _resolve_run_scripts_mount(volumes: Any) -> Optional[str]:
    if not volumes:
        return None
    if isinstance(volumes, dict):
        volumes = list(volumes.keys())
    if not isinstance(volumes, list):
        return None
    for volume in volumes:
        if not isinstance(volume, str):
            continue
        parts = volume.split(":")
        if len(parts) < 2:
            continue
        container_path = parts[1].rstrip("/")
        if container_path == "/run_scripts":
            return "/run_scripts"
    return None


def _build_write_command(path: str, content: bytes) -> str:
    encoded = base64.b64encode(content).decode("ascii")
    return (
        "python - <<'PY'\n"
        "import base64\n"
        "from pathlib import Path\n"
        f"path = {path!r}\n"
        f"data = base64.b64decode({encoded!r})\n"
        "target = Path(path)\n"
        "target.parent.mkdir(parents=True, exist_ok=True)\n"
        "target.write_bytes(data)\n"
        "PY\n"
    )
