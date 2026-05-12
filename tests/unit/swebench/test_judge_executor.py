from __future__ import annotations

import asyncio
import inspect
import json
from pathlib import Path

from gage_eval.agent_eval_kits.swebench.judge import executor as executor_module
from gage_eval.agent_eval_kits.swebench.judge.executor import (
    SwebenchExecutionRequest,
    build_write_command,
    create_entryscript,
    execute_swebench_verifier,
    extract_env_exports,
)
from gage_eval.environment.contracts import ExecResult


DIFF = "diff --git a/foo b/foo\n--- a/foo\n+++ b/foo\n@@ -1 +1 @@\n-old\n+new\n"


class RecordingEnvironment:
    env_id = "env-1"
    name = "fake"
    provider = "fake"
    metadata: dict[str, str] = {}
    capabilities = {}

    def __init__(
        self,
        *,
        output_json: dict | None = None,
        timed_out: bool = False,
        patch_status: dict | None = None,
        extra_files: dict[str, bytes | str] | None = None,
    ) -> None:
        self.files: dict[str, bytes] = {}
        self.exec_calls: list[tuple[str, dict]] = []
        self.output_json = output_json or {"tests": [{"name": "tests/test_fix.py::test_fix", "status": "PASSED"}]}
        self.timed_out = timed_out
        self.patch_status = patch_status
        for path, payload in (extra_files or {}).items():
            self.files[path] = payload.encode("utf-8") if isinstance(payload, str) else payload

    async def write_file(self, path: str, content: bytes | str) -> None:
        self.files[path] = content.encode("utf-8") if isinstance(content, str) else content

    async def read_file(self, path: str, *, max_bytes: int = 16 * 1024 * 1024) -> bytes:
        del max_bytes
        if path == "/workspace/output.json":
            return json.dumps(self.output_json).encode("utf-8")
        if path == "/workspace/patch_apply_status.json" and self.patch_status is not None:
            return json.dumps(self.patch_status).encode("utf-8")
        if path in self.files:
            return self.files[path]
        raise FileNotFoundError(path)

    async def exec(self, command: str, **kwargs) -> ExecResult:
        self.exec_calls.append((command, kwargs))
        return ExecResult(command=command, exit_code=124 if self.timed_out else 0, timed_out=self.timed_out)


def test_executor_runs_entryscript_through_base_environment() -> None:
    env = RecordingEnvironment()

    result = asyncio.run(
        execute_swebench_verifier(
            environment=env,
            request=SwebenchExecutionRequest(
                sample={
                    "id": "instance_1",
                    "metadata": {
                        "instance_id": "instance_1",
                        "base_commit": "abc123",
                        "fail_to_pass": ["tests/test_fix.py::test_fix"],
                        "pass_to_pass": [],
                    },
                },
                patch=DIFF,
                run_script="#!/bin/bash\npytest\n",
                parser_script="print('parse')\n",
                timeout_s=30,
            ),
        )
    )

    assert result["resolved"] is True
    assert env.files["/workspace/patch.diff"].decode("utf-8") == DIFF
    assert "bash /workspace/entryscript.sh" in env.exec_calls[0][0]
    assert env.exec_calls[0][1]["timeout_s"] == 30


def test_executor_reports_patch_apply_stage_for_successful_runs() -> None:
    result = asyncio.run(
        execute_swebench_verifier(
            environment=RecordingEnvironment(patch_status={"status": "applied", "patch": "patch.diff", "stage": "git_apply_lenient"}),
            request=SwebenchExecutionRequest(
                sample={
                    "id": "instance_1",
                    "metadata": {
                        "instance_id": "instance_1",
                        "base_commit": "abc123",
                        "fail_to_pass": ["tests/test_fix.py::test_fix"],
                        "pass_to_pass": [],
                    },
                },
                patch=DIFF,
                run_script="pytest\n",
                parser_script="print('parse')\n",
                timeout_s=30,
            ),
        )
    )

    assert result["resolved"] is True
    assert result["patch_applied_via"] == "git_apply_lenient"


def test_verifier_executor_timeout_maps_failure_code() -> None:
    result = asyncio.run(
        execute_swebench_verifier(
            environment=RecordingEnvironment(timed_out=True),
            request=SwebenchExecutionRequest(
                sample={"id": "instance_1", "metadata": {"instance_id": "instance_1", "base_commit": "abc123"}},
                patch=DIFF,
                run_script="pytest\n",
                parser_script="print('parse')\n",
                timeout_s=1,
            ),
        )
    )

    assert result["resolved"] is False
    assert result["failure_reason"] == "test_execution_error"
    assert result["failure_code"] == "verifier.executor.timeout"


def test_entryscript_extracts_env_exports_and_safe_write_command(tmp_path: Path) -> None:
    dockerfiles = tmp_path / "dockerfiles"
    dockerfile = dockerfiles / "base_dockerfile" / "instance_1" / "Dockerfile"
    dockerfile.parent.mkdir(parents=True)
    dockerfile.write_text("FROM python:3.11\nENV FOO=bar\n", encoding="utf-8")

    assert extract_env_exports(dockerfiles, "instance_1") == "export FOO=bar"
    script = create_entryscript(
        sample={"id": "instance_1"},
        base_commit="abc123",
        dockerfiles_dir=dockerfiles,
        test_patch=False,
        run_script_path="/workspace/run_script.sh",
        parser_path="/workspace/parser.py",
    )
    assert "export FOO=bar" in script
    assert "git reset --hard abc123" in script
    assert "checkout_base()" in script
    assert "if ! git reset --hard abc123" in script

    command = build_write_command("/workspace/file.txt", b"hello")
    assert "base64.b64decode" in command
    assert "/workspace/file.txt" in command


def test_entryscript_uses_strict_then_lenient_patch_apply_and_records_stage() -> None:
    script = create_entryscript(
        sample={"id": "instance_1"},
        base_commit="abc123",
        dockerfiles_dir=None,
        test_patch=False,
        run_script_path="/workspace/run_script.sh",
        parser_path="/workspace/parser.py",
    )

    strict_idx = script.index('git apply -v "$patch_path"')
    lenient_idx = script.index("git apply --recount --ignore-space-change --ignore-whitespace -v")
    fallback_idx = script.index("patch -p1 --force --ignore-whitespace --batch")
    assert strict_idx < lenient_idx < fallback_idx
    assert "write_patch_status applied \"$label\" git_apply" in script
    assert "write_patch_status applied \"$label\" git_apply_lenient" in script
    assert "write_patch_status applied \"$label\" patch_fallback" in script
    assert "write_patch_status failed \"$label\" patch_fallback" in script


def test_entryscript_can_disable_lenient_patch_apply_for_official_strict_mode() -> None:
    script = create_entryscript(
        sample={"id": "instance_1"},
        base_commit="abc123",
        dockerfiles_dir=None,
        test_patch=False,
        strict_patch_apply=True,
        run_script_path="/workspace/run_script.sh",
        parser_path="/workspace/parser.py",
    )

    assert 'git apply -v "$patch_path"' in script
    assert "git apply --recount --ignore-space-change --ignore-whitespace -v" not in script
    assert "patch -p1 --force --ignore-whitespace --batch" not in script
    assert "write_patch_status failed \"$label\" git_apply" in script


def test_selected_test_files_empty_list_omits_run_script_argument() -> None:
    script = create_entryscript(
        sample={"id": "instance_1", "metadata": {"selected_test_files_to_run": "[]"}},
        base_commit="abc123",
        dockerfiles_dir=None,
        test_patch=False,
        run_script_path="/workspace/run_script.sh",
        parser_path="/workspace/parser.py",
    )

    assert "bash /workspace/run_script.sh > /workspace/stdout.log" in script
    assert "bash /workspace/run_script.sh  >" not in script


def test_test_patch_apply_is_executed_after_main_patch() -> None:
    script = create_entryscript(
        sample={"id": "instance_1"},
        base_commit="abc123",
        dockerfiles_dir=None,
        test_patch=True,
        run_script_path="/workspace/run_script.sh",
        parser_path="/workspace/parser.py",
    )

    assert script.index("apply_patch /workspace/patch.diff") < script.index(
        "apply_patch /workspace/test_patch.diff"
    )


def test_test_patch_apply_failure_maps_failure_reason() -> None:
    result = asyncio.run(
        execute_swebench_verifier(
            environment=RecordingEnvironment(
                patch_status={
                    "status": "failed",
                    "stage": "patch_fallback",
                    "patch": "test_patch.diff",
                    "log": "/workspace/test_patch_apply.log",
                }
            ),
            request=SwebenchExecutionRequest(
                sample={"id": "instance_1", "metadata": {"instance_id": "instance_1", "base_commit": "abc123"}},
                patch=DIFF,
                test_patch=DIFF,
                run_script="pytest\n",
                parser_script="print('parse')\n",
                timeout_s=30,
            ),
        )
    )

    assert result["resolved"] is False
    assert result["failure_reason"] == "test_patch_apply_failed"


def test_executor_uses_default_dockerfiles_dir_when_resource_omitted(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dockerfiles = tmp_path / "dockerfiles"
    dockerfile = dockerfiles / "instance_dockerfile" / "instance_1" / "Dockerfile"
    dockerfile.parent.mkdir(parents=True)
    dockerfile.write_text("FROM scratch\nENV NODE_VERSION=20\n", encoding="utf-8")
    monkeypatch.setattr(executor_module, "_default_dockerfiles_dir", lambda: dockerfiles)
    env = RecordingEnvironment()

    result = asyncio.run(
        execute_swebench_verifier(
            environment=env,
            request=SwebenchExecutionRequest(
                sample={
                    "id": "instance_1",
                    "metadata": {
                        "instance_id": "instance_1",
                        "base_commit": "abc123",
                        "fail_to_pass": ["tests/test_fix.py::test_fix"],
                    },
                },
                patch=DIFF,
                run_script="pytest\n",
                parser_script="print('parse')\n",
                timeout_s=30,
            ),
        )
    )

    assert result["resolved"] is True
    assert "export NODE_VERSION=20" in env.files["/workspace/entryscript.sh"].decode("utf-8")


def test_verifier_checkout_failure_maps_stable_failure_code() -> None:
    result = asyncio.run(
        execute_swebench_verifier(
            environment=RecordingEnvironment(
                patch_status={
                    "status": "failed",
                    "stage": "checkout",
                    "failure_code": "verifier.checkout_failed",
                    "log": "/workspace/checkout.log",
                },
                extra_files={"/workspace/checkout.log": "unknown revision"},
            ),
            request=SwebenchExecutionRequest(
                sample={"id": "instance_1", "metadata": {"instance_id": "instance_1", "base_commit": "abc123"}},
                patch=DIFF,
                run_script="pytest\n",
                parser_script="print('parse')\n",
                timeout_s=30,
            ),
        )
    )

    assert result["resolved"] is False
    assert result["failure_reason"] == "test_execution_error"
    assert result["failure_code"] == "verifier.checkout_failed"
    assert result["verifier_logs"]["checkout"] == "unknown revision"


def test_executor_preserves_verifier_logs_in_result_payload() -> None:
    result = asyncio.run(
        execute_swebench_verifier(
            environment=RecordingEnvironment(
                extra_files={
                    "/workspace/stdout.log": "stdout body",
                    "/workspace/stderr.log": "stderr body",
                    "/workspace/patch_apply.log": "patch body",
                }
            ),
            request=SwebenchExecutionRequest(
                sample={
                    "id": "instance_1",
                    "metadata": {
                        "instance_id": "instance_1",
                        "base_commit": "abc123",
                        "fail_to_pass": ["tests/test_fix.py::test_fix"],
                    },
                },
                patch=DIFF,
                run_script="pytest\n",
                parser_script="print('parse')\n",
                timeout_s=30,
            ),
        )
    )

    assert result["verifier_logs"] == {
        "patch_apply": "patch body",
        "stdout": "stdout body",
        "stderr": "stderr body",
    }


def test_executor_truncates_oversized_verifier_logs_instead_of_dropping_them() -> None:
    result = asyncio.run(
        execute_swebench_verifier(
            environment=RecordingEnvironment(extra_files={"/workspace/stdout.log": "x" * (64 * 1024 + 10)}),
            request=SwebenchExecutionRequest(
                sample={"id": "instance_1", "metadata": {"instance_id": "instance_1", "base_commit": "abc123"}},
                patch=DIFF,
                run_script="pytest\n",
                parser_script="print('parse')\n",
                timeout_s=30,
            ),
        )
    )

    stdout_log = result["verifier_logs"]["stdout"]
    assert stdout_log.startswith("x" * 100)
    assert stdout_log.endswith("...[truncated to 65536 bytes]")


def test_executor_module_does_not_import_docker_sdk() -> None:
    source = inspect.getsource(executor_module)

    assert "import docker" not in source
    assert "docker.from_env" not in source
