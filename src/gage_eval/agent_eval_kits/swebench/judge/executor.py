from __future__ import annotations

import base64
import shlex
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from gage_eval.environment.contracts import DEFAULT_READ_FILE_LIMIT_BYTES, BaseEnvironment

from .scoring import load_output_json, parse_list, score_output

_VERIFIER_LOG_PATHS = {
    "checkout": "/workspace/checkout.log",
    "patch_apply": "/workspace/patch_apply.log",
    "test_patch_apply": "/workspace/test_patch_apply.log",
    "stdout": "/workspace/stdout.log",
    "stderr": "/workspace/stderr.log",
}
_VERIFIER_LOG_MAX_BYTES = 64 * 1024
_VERIFIER_LOG_READ_MAX_BYTES = DEFAULT_READ_FILE_LIMIT_BYTES


@dataclass(frozen=True)
class SwebenchExecutionRequest:
    sample: dict[str, Any]
    patch: str
    run_script: str
    parser_script: str
    test_patch: str | None = None
    timeout_s: int = 900
    dockerfiles_dir: str | Path | None = None
    strict_patch_apply: bool = False


async def execute_swebench_verifier(
    *,
    environment: BaseEnvironment,
    request: SwebenchExecutionRequest,
) -> dict[str, Any]:
    sample = request.sample
    base_commit = str(_get_meta(sample, "base_commit") or "")
    entryscript = create_entryscript(
        sample=sample,
        base_commit=base_commit,
        dockerfiles_dir=(
            Path(request.dockerfiles_dir) if request.dockerfiles_dir else _default_dockerfiles_dir()
        ),
        test_patch=bool(request.test_patch),
        strict_patch_apply=request.strict_patch_apply,
        run_script_path="/workspace/run_script.sh",
        parser_path="/workspace/parser.py",
    )
    await environment.write_file("/workspace/patch.diff", request.patch)
    await environment.write_file("/workspace/run_script.sh", request.run_script)
    await environment.write_file("/workspace/parser.py", request.parser_script)
    await environment.write_file("/workspace/entryscript.sh", entryscript)
    if request.test_patch:
        await environment.write_file("/workspace/test_patch.diff", request.test_patch)

    result = await environment.exec(
        "bash /workspace/entryscript.sh",
        timeout_s=max(1, int(request.timeout_s)),
    )
    if getattr(result, "timed_out", False):
        return await _with_verifier_logs(
            environment,
            {
                "resolved": False,
                "score": 0.0,
                "failure_reason": "test_execution_error",
                "failure_code": "verifier.executor.timeout",
            },
        )
    verifier_logs = await _collect_verifier_logs(environment)
    if getattr(result, "exit_code", 0) not in (0, None):
        return _attach_verifier_logs(
            {
                "resolved": False,
                "score": 0.0,
                "failure_reason": "test_execution_error",
                "failure_code": "verifier.executor.failed",
            },
            verifier_logs,
        )

    patch_status = await _read_json_file(environment, "/workspace/patch_apply_status.json")
    if patch_status and patch_status.get("status") == "failed":
        if patch_status.get("stage") == "checkout":
            return _attach_verifier_logs(
                {
                    "resolved": False,
                    "score": 0.0,
                    "failure_reason": "test_execution_error",
                    "failure_code": str(
                        patch_status.get("failure_code") or "verifier.checkout_failed"
                    ),
                },
                verifier_logs,
            )
        patch_label = str(patch_status.get("patch") or "")
        failure_reason = (
            "test_patch_apply_failed" if patch_label == "test_patch.diff" else "patch_apply_failed"
        )
        return _attach_verifier_logs(
            {"resolved": False, "score": 0.0, "failure_reason": failure_reason},
            verifier_logs,
        )

    output_payload = await _read_optional_file(environment, "/workspace/output.json")
    if output_payload is None:
        return _attach_verifier_logs(
            {"resolved": False, "score": 0.0, "failure_reason": "missing_output"},
            verifier_logs,
        )
    output = load_output_json(output_payload)
    if output is None:
        return _attach_verifier_logs(
            {"resolved": False, "score": 0.0, "failure_reason": "invalid_output"},
            verifier_logs,
        )

    scored = score_output(
        output,
        fail_to_pass=parse_list(_get_meta(sample, "fail_to_pass")),
        pass_to_pass=parse_list(_get_meta(sample, "pass_to_pass")),
    )
    if patch_status and patch_status.get("status") == "applied" and patch_status.get("stage"):
        scored["patch_applied_via"] = str(patch_status["stage"])
    return _attach_verifier_logs(scored, verifier_logs)


async def _with_verifier_logs(environment: BaseEnvironment, payload: dict[str, Any]) -> dict[str, Any]:
    return _attach_verifier_logs(payload, await _collect_verifier_logs(environment))


def _attach_verifier_logs(payload: dict[str, Any], verifier_logs: Mapping[str, str]) -> dict[str, Any]:
    if verifier_logs:
        payload["verifier_logs"] = dict(verifier_logs)
    return payload


async def _collect_verifier_logs(environment: BaseEnvironment) -> dict[str, str]:
    logs: dict[str, str] = {}
    for key, path in _VERIFIER_LOG_PATHS.items():
        payload = await _read_optional_file(environment, path, max_bytes=_VERIFIER_LOG_READ_MAX_BYTES)
        if payload is None:
            continue
        text = _truncate_utf8(_decode_payload(payload), _VERIFIER_LOG_MAX_BYTES).strip()
        if text:
            logs[key] = text
    return logs


def create_entryscript(
    *,
    sample: Mapping[str, Any],
    base_commit: str,
    dockerfiles_dir: Path | None,
    test_patch: bool,
    strict_patch_apply: bool = False,
    run_script_path: str,
    parser_path: str,
) -> str:
    before_repo_set_cmd = str(_get_meta(sample, "before_repo_set_cmd") or "").strip()
    if before_repo_set_cmd:
        before_repo_set_cmd = before_repo_set_cmd.split("\n")[-1]
    selected = parse_list(_get_meta(sample, "selected_test_files_to_run"))
    selected_arg = ",".join(selected) if selected else ""
    env_cmds = (
        extract_env_exports(dockerfiles_dir, _resolve_instance_id(sample)) if dockerfiles_dir else ""
    )

    lines: list[str] = []
    if env_cmds:
        lines.append(env_cmds)
    lines.extend(
        [
            "PATCH_STATUS=/workspace/patch_apply_status.json",
            "write_patch_status() {",
            "  status=\"$1\"",
            "  label=\"$2\"",
            "  stage=\"$3\"",
            "  log_path=\"$4\"",
            "  printf '{\"status\":\"%s\",\"patch\":\"%s\",\"stage\":\"%s\",\"log\":\"%s\"}\\n' "
            "\"$status\" \"$label\" \"$stage\" \"$log_path\" > \"$PATCH_STATUS\"",
            "}",
            "apply_patch() {",
            "  patch_path=\"$1\"",
            "  log_path=\"$2\"",
            "  label=\"$3\"",
            "  if [ ! -f \"$patch_path\" ]; then",
            "    return 0",
            "  fi",
            "  if git apply -v \"$patch_path\" > \"$log_path\" 2>&1; then",
            "    write_patch_status applied \"$label\" git_apply \"$log_path\"",
            "    return 0",
            "  fi",
            *(
                [
                    "  echo \"strict git apply failed; official strict mode is enabled.\" >> \"$log_path\"",
                    "  write_patch_status failed \"$label\" git_apply \"$log_path\"",
                    "  exit 0",
                ]
                if strict_patch_apply
                else []
            ),
            *(
                []
                if strict_patch_apply
                else [
                    "  echo \"strict git apply failed, trying lenient git apply...\" >> \"$log_path\"",
                    "  if git apply --recount --ignore-space-change --ignore-whitespace -v \"$patch_path\" >> \"$log_path\" 2>&1; then",
                    "    write_patch_status applied \"$label\" git_apply_lenient \"$log_path\"",
                    "    return 0",
                    "  fi",
                    "  echo \"lenient git apply failed, trying patch tool...\" >> \"$log_path\"",
                    "  if command -v patch >/dev/null 2>&1; then",
                    "    if patch -p1 --force --ignore-whitespace --batch -i \"$patch_path\" >> \"$log_path\" 2>&1; then",
                    "      write_patch_status applied \"$label\" patch_fallback \"$log_path\"",
                    "      return 0",
                    "    fi",
                    "  fi",
                    "  write_patch_status failed \"$label\" patch_fallback \"$log_path\"",
                ]
            ),
            "  exit 0",
            "}",
            "write_checkout_failure() {",
            "  failure_code=\"$1\"",
            "  printf '{\"status\":\"failed\",\"stage\":\"checkout\",\"failure_code\":\"%s\",\"log\":\"/workspace/checkout.log\"}\\n' \"$failure_code\" > \"$PATCH_STATUS\"",
            "  exit 0",
            "}",
            "checkout_base() {",
            "  if [ ! -d /app/.git ]; then",
            "    echo \"/app is not a git repository\" > /workspace/checkout.log",
            "    write_checkout_failure verifier.checkout_failed",
            "  fi",
            "  cd /app || {",
            "    echo \"cannot cd /app\" > /workspace/checkout.log",
            "    write_checkout_failure verifier.checkout_failed",
            "  }",
            f"  if ! git reset --hard {shlex.quote(base_commit)} > /workspace/checkout.log 2>&1; then",
            "    write_checkout_failure verifier.checkout_failed",
            "  fi",
            f"  if ! git checkout {shlex.quote(base_commit)} >> /workspace/checkout.log 2>&1; then",
            "    write_checkout_failure verifier.checkout_failed",
            "  fi",
            "}",
            "checkout_base",
            "apply_patch /workspace/patch.diff /workspace/patch_apply.log patch.diff",
        ]
    )
    if test_patch:
        lines.append("apply_patch /workspace/test_patch.diff /workspace/test_patch_apply.log test_patch.diff")
    if before_repo_set_cmd:
        lines.append(before_repo_set_cmd)
    command = f"bash {shlex.quote(run_script_path)}"
    if selected_arg:
        command = f"{command} {shlex.quote(selected_arg)}"
    lines.append(f"{command} > /workspace/stdout.log 2> /workspace/stderr.log")
    lines.extend(
        [
            "PYTHON_BIN=python",
            "if ! command -v \"$PYTHON_BIN\" >/dev/null 2>&1; then",
            "  PYTHON_BIN=python3",
            "fi",
            f"$PYTHON_BIN {shlex.quote(parser_path)} /workspace/stdout.log /workspace/stderr.log /workspace/output.json",
        ]
    )
    return "\n".join(lines) + "\n"


def extract_env_exports(dockerfiles_dir: Path | None, instance_id: str) -> str:
    if dockerfiles_dir is None:
        return ""
    env_cmds: list[str] = []
    for kind in ("base_dockerfile", "instance_dockerfile"):
        dockerfile_path = dockerfiles_dir / kind / instance_id / "Dockerfile"
        if not dockerfile_path.exists():
            continue
        for line in dockerfile_path.read_text(encoding="utf-8", errors="replace").splitlines():
            stripped = line.strip()
            if stripped.startswith("ENV"):
                env_cmds.append(stripped.replace("ENV", "export", 1))
    return "\n".join(env_cmds)


def _default_dockerfiles_dir() -> Path:
    return Path(__file__).resolve().parents[5] / "third_party" / "swebench_pro" / "dockerfiles"


def build_write_command(path: str, content: bytes) -> str:
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


async def _read_optional_file(
    environment: BaseEnvironment,
    path: str,
    *,
    max_bytes: int = 16 * 1024 * 1024,
) -> bytes | None:
    try:
        return await environment.read_file(path, max_bytes=max_bytes)
    except Exception:
        return None


async def _read_json_file(environment: BaseEnvironment, path: str) -> dict[str, Any] | None:
    payload = await _read_optional_file(environment, path)
    if payload is None:
        return None
    parsed = load_output_json(payload)
    return parsed


def _get_meta(sample: Mapping[str, Any], key: str) -> Any:
    metadata = sample.get("metadata") or {}
    if isinstance(metadata, Mapping):
        value = metadata.get(key)
        if value is not None:
            return value
    return sample.get(key)


def _decode_payload(payload: Any) -> str:
    if isinstance(payload, (bytes, bytearray)):
        return payload.decode("utf-8", errors="replace")
    if payload is None:
        return ""
    return str(payload)


def _truncate_utf8(text: str, max_bytes: int) -> str:
    encoded = text.encode("utf-8")
    if len(encoded) <= max_bytes:
        return text
    return encoded[:max_bytes].decode("utf-8", errors="ignore") + f"\n...[truncated to {max_bytes} bytes]"


def _resolve_instance_id(sample: Mapping[str, Any]) -> str:
    return str(_get_meta(sample, "instance_id") or sample.get("id") or "")


_create_entryscript = create_entryscript
_extract_env_exports = extract_env_exports
_build_write_command = build_write_command
