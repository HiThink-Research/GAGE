from __future__ import annotations

import json

import pytest

from gage_eval.role.judge.swebench_docker import SwebenchDocker


class FakeSandbox:
    def __init__(self, outputs: dict[str, bytes] | None = None) -> None:
        self.outputs = outputs or {}
        self.writes: dict[str, bytes] = {}

    def exec(self, command: str, timeout: int = 30):
        if command.startswith("cat /workspace/submission.patch"):
            payload = self.outputs.get("/workspace/submission.patch", b"")
            return type("ExecResult", (), {"exit_code": 0, "stdout": payload.decode("utf-8"), "stderr": ""})()
        return type("ExecResult", (), {"exit_code": 0, "stdout": "", "stderr": ""})()

    def write_file(self, path: str, content: bytes) -> None:
        if isinstance(content, str):
            content = content.encode("utf-8")
        self.writes[path] = content

    def read_file(self, path: str) -> bytes:
        return self.outputs.get(path, b"")


class FakeHandle:
    def __init__(self, sandbox: FakeSandbox) -> None:
        self.sandbox = sandbox
        self.config = {"runtime_configs": {}}


class FakeProvider:
    def __init__(self, sandbox: FakeSandbox) -> None:
        self._handle = FakeHandle(sandbox)

    def get_handle(self) -> FakeHandle:
        return self._handle


@pytest.mark.io
def test_swebench_judge_fallback_reads_submission_patch(tmp_path, temp_workspace, mock_trace) -> None:
    instance_id = "instance_1"
    scripts_root = tmp_path / "run_scripts"
    run_dir = scripts_root / instance_id
    run_dir.mkdir(parents=True)
    (run_dir / "run_script.sh").write_text("#!/bin/bash\necho ok\n", encoding="utf-8")
    (run_dir / "parser.py").write_text("print('ok')\n", encoding="utf-8")

    patch = (
        "diff --git a/foo.txt b/foo.txt\n"
        "index 1111111..2222222 100644\n"
        "--- a/foo.txt\n"
        "+++ b/foo.txt\n"
        "@@ -1,1 +1,1 @@\n"
        "-old\n"
        "+new\n"
    )
    output = {"tests": [{"name": "tests/test_bar.py::test_bar", "status": "PASSED"}]}
    sandbox = FakeSandbox(
        outputs={
            "/workspace/submission.patch": patch.encode("utf-8"),
            "/workspace/output.json": json.dumps(output).encode("utf-8"),
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
        "model_output": {},
        "params": {},
        "sandbox_provider": provider,
        "trace": mock_trace,
    }

    result = judge.invoke(payload)

    assert result["resolved"] is True
    assert "/workspace/patch.diff" in sandbox.writes
    events = [item["event"] for item in mock_trace.events]
    assert "swebench_patch_fallback" in events
