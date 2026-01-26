from __future__ import annotations

import json

import pytest

from gage_eval.role.judge.swebench_docker import SwebenchDocker


class FakeSandbox:
    def __init__(self, outputs: dict[str, bytes] | None = None) -> None:
        self.outputs = outputs or {}
        self.writes: dict[str, bytes] = {}
        self.exec_calls: list[str] = []

    def exec(self, command: str, timeout: int = 30):
        self.exec_calls.append(command)
        return type("ExecResult", (), {"exit_code": 0, "stdout": "", "stderr": ""})()

    def write_file(self, path: str, content: bytes) -> None:
        if isinstance(content, str):
            content = content.encode("utf-8")
        self.writes[path] = content

    def read_file(self, path: str) -> bytes:
        if path in self.outputs:
            return self.outputs[path]
        return self.writes.get(path, b"")


class FakeHandle:
    def __init__(self, sandbox: FakeSandbox, runtime_configs: dict | None = None) -> None:
        self.sandbox = sandbox
        self.config = {"runtime_configs": runtime_configs or {}}


class FakeProvider:
    def __init__(self, sandbox: FakeSandbox, runtime_configs: dict | None = None) -> None:
        self._handle = FakeHandle(sandbox, runtime_configs)

    def get_handle(self) -> FakeHandle:
        return self._handle


class MissingFileSandbox(FakeSandbox):
    def read_file(self, path: str) -> bytes:  # type: ignore[override]
        if path == "/workspace/patch_apply_status.json":
            raise RuntimeError("docker_cp_failed: no such file or directory")
        return super().read_file(path)


@pytest.mark.io
def test_swebench_judge_sandbox_path(tmp_path, temp_workspace) -> None:
    instance_id = "instance_1"
    scripts_root = tmp_path / "run_scripts"
    run_dir = scripts_root / instance_id
    run_dir.mkdir(parents=True)
    (run_dir / "run_script.sh").write_text("#!/bin/bash\necho ok\n", encoding="utf-8")
    (run_dir / "parser.py").write_text("print('ok')\n", encoding="utf-8")

    output = {"tests": [{"name": "tests/test_bar.py::test_bar", "status": "PASSED"}]}
    sandbox = FakeSandbox(outputs={"/workspace/output.json": json.dumps(output).encode("utf-8")})
    provider = FakeProvider(sandbox)

    judge = SwebenchDocker(scripts_dir=str(scripts_root))
    sample = {
        "id": instance_id,
        "messages": [{"role": "user", "content": "fix bug"}],
        "metadata": {
            "instance_id": instance_id,
            "repo": "repo/name",
            "base_commit": "abc",
            "fail_to_pass": ["tests/test_bar.py::test_bar"],
            "pass_to_pass": [],
        },
    }
    payload = {
        "sample": sample,
        "model_output": {"answer": "patch"},
        "params": {},
        "sandbox_provider": provider,
    }

    result = judge.invoke(payload)

    assert result["resolved"] is True
    assert "/workspace/patch.diff" in sandbox.writes
    entryscript = sandbox.writes["/workspace/entryscript.sh"].decode("utf-8")
    assert "bash /workspace/run_script.sh" in entryscript


@pytest.mark.io
def test_swebench_judge_patch_apply_failure(tmp_path, temp_workspace) -> None:
    instance_id = "instance_1"
    scripts_root = tmp_path / "run_scripts"
    run_dir = scripts_root / instance_id
    run_dir.mkdir(parents=True)
    (run_dir / "run_script.sh").write_text("#!/bin/bash\necho ok\n", encoding="utf-8")
    (run_dir / "parser.py").write_text("print('ok')\n", encoding="utf-8")

    status = {"status": "failed", "patch": "patch.diff", "log": "/workspace/patch_apply.log"}
    sandbox = FakeSandbox(
        outputs={
            "/workspace/patch_apply_status.json": json.dumps(status).encode("utf-8"),
            "/workspace/patch_apply.log": b"patch failed",
        }
    )
    provider = FakeProvider(sandbox)

    judge = SwebenchDocker(scripts_dir=str(scripts_root))
    sample = {
        "id": instance_id,
        "messages": [{"role": "user", "content": "fix bug"}],
        "metadata": {
            "instance_id": instance_id,
            "repo": "repo/name",
            "base_commit": "abc",
            "fail_to_pass": ["tests/test_bar.py::test_bar"],
            "pass_to_pass": [],
        },
    }
    payload = {
        "sample": sample,
        "model_output": {"answer": "patch"},
        "params": {},
        "sandbox_provider": provider,
    }

    result = judge.invoke(payload)

    assert result["resolved"] is False
    assert result["failure_reason"] == "patch_apply_failed"


@pytest.mark.io
def test_swebench_judge_missing_patch_status_is_ok(tmp_path, temp_workspace) -> None:
    instance_id = "instance_1"
    scripts_root = tmp_path / "run_scripts"
    run_dir = scripts_root / instance_id
    run_dir.mkdir(parents=True)
    (run_dir / "run_script.sh").write_text("#!/bin/bash\necho ok\n", encoding="utf-8")
    (run_dir / "parser.py").write_text("print('ok')\n", encoding="utf-8")

    output = {"tests": [{"name": "tests/test_bar.py::test_bar", "status": "PASSED"}]}
    sandbox = MissingFileSandbox(outputs={"/workspace/output.json": json.dumps(output).encode("utf-8")})
    provider = FakeProvider(sandbox)

    judge = SwebenchDocker(scripts_dir=str(scripts_root))
    sample = {
        "id": instance_id,
        "messages": [{"role": "user", "content": "fix bug"}],
        "metadata": {
            "instance_id": instance_id,
            "repo": "repo/name",
            "base_commit": "abc",
            "fail_to_pass": ["tests/test_bar.py::test_bar"],
            "pass_to_pass": [],
        },
    }
    payload = {
        "sample": sample,
        "model_output": {"answer": "patch"},
        "params": {},
        "sandbox_provider": provider,
    }

    result = judge.invoke(payload)

    assert result["resolved"] is True


@pytest.mark.io
def test_swebench_judge_cleans_patch_markdown(tmp_path, temp_workspace) -> None:
    instance_id = "instance_1"
    scripts_root = tmp_path / "run_scripts"
    run_dir = scripts_root / instance_id
    run_dir.mkdir(parents=True)
    (run_dir / "run_script.sh").write_text("#!/bin/bash\necho ok\n", encoding="utf-8")
    (run_dir / "parser.py").write_text("print('ok')\n", encoding="utf-8")

    output = {"tests": [{"name": "tests/test_bar.py::test_bar", "status": "PASSED"}]}
    sandbox = FakeSandbox(outputs={"/workspace/output.json": json.dumps(output).encode("utf-8")})
    provider = FakeProvider(sandbox)

    raw_patch = (
        "Here is the patch you requested:\n"
        "```diff\n"
        "diff --git a/foo.txt b/foo.txt\n"
        "index 1111111..2222222 100644\n"
        "--- a/foo.txt\n"
        "+++ b/foo.txt\n"
        "@@ -1,1 +1,1 @@\n"
        "-old\n"
        "+new\n"
        "```\n"
        "Thanks!\n"
    )

    judge = SwebenchDocker(scripts_dir=str(scripts_root))
    sample = {
        "id": instance_id,
        "messages": [{"role": "user", "content": "fix bug"}],
        "metadata": {
            "instance_id": instance_id,
            "repo": "repo/name",
            "base_commit": "abc",
            "fail_to_pass": ["tests/test_bar.py::test_bar"],
            "pass_to_pass": [],
        },
    }
    payload = {
        "sample": sample,
        "model_output": {"answer": raw_patch},
        "params": {},
        "sandbox_provider": provider,
    }

    result = judge.invoke(payload)

    assert result["resolved"] is True
    cleaned_patch = sandbox.writes["/workspace/patch.diff"].decode("utf-8")
    expected = (
        "diff --git a/foo.txt b/foo.txt\n"
        "index 1111111..2222222 100644\n"
        "--- a/foo.txt\n"
        "+++ b/foo.txt\n"
        "@@ -1,1 +1,1 @@\n"
        "-old\n"
        "+new\n"
    )
    assert cleaned_patch.strip() == expected.strip()


@pytest.mark.io
def test_swebench_judge_uses_agent_trace_patch_when_answer_missing(tmp_path, temp_workspace) -> None:
    instance_id = "instance_1"
    scripts_root = tmp_path / "run_scripts"
    run_dir = scripts_root / instance_id
    run_dir.mkdir(parents=True)
    (run_dir / "run_script.sh").write_text("#!/bin/bash\necho ok\n", encoding="utf-8")
    (run_dir / "parser.py").write_text("print('ok')\n", encoding="utf-8")

    output = {"tests": [{"name": "tests/test_bar.py::test_bar", "status": "PASSED"}]}
    sandbox = FakeSandbox(outputs={"/workspace/output.json": json.dumps(output).encode("utf-8")})
    provider = FakeProvider(sandbox)

    trace_patch = (
        "diff --git a/foo.txt b/foo.txt\n"
        "index 1111111..2222222 100644\n"
        "--- a/foo.txt\n"
        "+++ b/foo.txt\n"
        "@@ -1,1 +1,1 @@\n"
        "-old\n"
        "+new\n"
    )

    judge = SwebenchDocker(scripts_dir=str(scripts_root))
    sample = {
        "id": instance_id,
        "messages": [{"role": "user", "content": "fix bug"}],
        "metadata": {
            "instance_id": instance_id,
            "repo": "repo/name",
            "base_commit": "abc",
            "fail_to_pass": ["tests/test_bar.py::test_bar"],
            "pass_to_pass": [],
        },
    }
    payload = {
        "sample": sample,
        "model_output": {
            "answer": "",
            "agent_trace": [
                {
                    "trace_role": "tool",
                    "name": "submit_patch_tool",
                    "output": {"stdout": trace_patch},
                }
            ],
        },
        "params": {},
        "sandbox_provider": provider,
    }

    result = judge.invoke(payload)

    assert result["resolved"] is True
    cleaned_patch = sandbox.writes["/workspace/patch.diff"].decode("utf-8")
    assert cleaned_patch.strip() == trace_patch.strip()


@pytest.mark.io
def test_swebench_judge_normalizes_hunk_context_lines(tmp_path, temp_workspace) -> None:
    instance_id = "instance_1"
    scripts_root = tmp_path / "run_scripts"
    run_dir = scripts_root / instance_id
    run_dir.mkdir(parents=True)
    (run_dir / "run_script.sh").write_text("#!/bin/bash\necho ok\n", encoding="utf-8")
    (run_dir / "parser.py").write_text("print('ok')\n", encoding="utf-8")

    output = {"tests": [{"name": "tests/test_bar.py::test_bar", "status": "PASSED"}]}
    sandbox = FakeSandbox(outputs={"/workspace/output.json": json.dumps(output).encode("utf-8")})
    provider = FakeProvider(sandbox)

    raw_patch = (
        "diff --git a/foo.txt b/foo.txt\n"
        "index 1111111..2222222 100644\n"
        "--- a/foo.txt\n"
        "+++ b/foo.txt\n"
        "@@ -1,3 +1,3 @@\n"
        "line1\n"
        "-old\n"
        "+new\n"
        " line3\n"
    )

    judge = SwebenchDocker(scripts_dir=str(scripts_root))
    sample = {
        "id": instance_id,
        "messages": [{"role": "user", "content": "fix bug"}],
        "metadata": {
            "instance_id": instance_id,
            "repo": "repo/name",
            "base_commit": "abc",
            "fail_to_pass": ["tests/test_bar.py::test_bar"],
            "pass_to_pass": [],
        },
    }
    payload = {
        "sample": sample,
        "model_output": {"answer": raw_patch},
        "params": {},
        "sandbox_provider": provider,
    }

    result = judge.invoke(payload)

    assert result["resolved"] is True
    cleaned_patch = sandbox.writes["/workspace/patch.diff"].decode("utf-8")
    assert "\n line1\n" in cleaned_patch
    assert "\n line3\n" in cleaned_patch


@pytest.mark.io
def test_swebench_judge_trims_diff_tail(tmp_path, temp_workspace) -> None:
    instance_id = "instance_1"
    scripts_root = tmp_path / "run_scripts"
    run_dir = scripts_root / instance_id
    run_dir.mkdir(parents=True)
    (run_dir / "run_script.sh").write_text("#!/bin/bash\necho ok\n", encoding="utf-8")
    (run_dir / "parser.py").write_text("print('ok')\n", encoding="utf-8")

    output = {"tests": [{"name": "tests/test_bar.py::test_bar", "status": "PASSED"}]}
    sandbox = FakeSandbox(outputs={"/workspace/output.json": json.dumps(output).encode("utf-8")})
    provider = FakeProvider(sandbox)

    raw_patch = (
        "diff --git a/foo.txt b/foo.txt\n"
        "index 1111111..2222222 100644\n"
        "--- a/foo.txt\n"
        "+++ b/foo.txt\n"
        "@@ -1,1 +1,1 @@\n"
        "-old\n"
        "+new\n"
        "Note: apply this fix ASAP.\n"
    )

    judge = SwebenchDocker(scripts_dir=str(scripts_root))
    sample = {
        "id": instance_id,
        "messages": [{"role": "user", "content": "fix bug"}],
        "metadata": {
            "instance_id": instance_id,
            "repo": "repo/name",
            "base_commit": "abc",
            "fail_to_pass": ["tests/test_bar.py::test_bar"],
            "pass_to_pass": [],
        },
    }
    payload = {
        "sample": sample,
        "model_output": {"answer": raw_patch},
        "params": {},
        "sandbox_provider": provider,
    }

    result = judge.invoke(payload)

    assert result["resolved"] is True
    cleaned_patch = sandbox.writes["/workspace/patch.diff"].decode("utf-8")
    assert "Note: apply this fix ASAP." not in cleaned_patch
