from __future__ import annotations

import json

import pytest

from gage_eval.role.judge.swebench_docker import SwebenchDocker


class FakeSandbox:
    def __init__(self, outputs: dict[str, bytes] | None = None) -> None:
        self.outputs = outputs or {}
        self.writes: dict[str, bytes] = {}

    def exec(self, command: str, timeout: int = 30):
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
    def __init__(self, sandbox: FakeSandbox, runtime_configs: dict) -> None:
        self.sandbox = sandbox
        self.config = {"runtime_configs": runtime_configs}


class FakeProvider:
    def __init__(self, sandbox: FakeSandbox, runtime_configs: dict) -> None:
        self._handles = [FakeHandle(sandbox, runtime_configs)]
        self._current = 0

    @classmethod
    def with_restart(
        cls,
        agent_sandbox: FakeSandbox,
        judge_sandbox: FakeSandbox,
        runtime_configs: dict,
    ) -> "FakeProvider":
        provider = cls(agent_sandbox, runtime_configs)
        provider._handles = [
            FakeHandle(agent_sandbox, runtime_configs),
            FakeHandle(judge_sandbox, runtime_configs),
        ]
        return provider

    def get_handle(self) -> FakeHandle:
        return self._handles[self._current]

    def release(self) -> None:
        if self._current < len(self._handles) - 1:
            self._current += 1


@pytest.mark.io
def test_swebench_judge_volume_fastpath(tmp_path, temp_workspace) -> None:
    instance_id = "instance_2"
    scripts_root = tmp_path / "run_scripts"
    run_dir = scripts_root / instance_id
    run_dir.mkdir(parents=True)
    (run_dir / "run_script.sh").write_text("#!/bin/bash\necho ok\n", encoding="utf-8")
    (run_dir / "parser.py").write_text("print('ok')\n", encoding="utf-8")

    output = {"tests": [{"name": "tests/test_bar.py::test_bar", "status": "PASSED"}]}
    agent_sandbox = FakeSandbox()
    judge_sandbox = FakeSandbox(outputs={"/workspace/output.json": json.dumps(output).encode("utf-8")})
    provider = FakeProvider.with_restart(
        agent_sandbox,
        judge_sandbox,
        {"volumes": ["host:/run_scripts:ro"]},
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
        "model_output": {"answer": "patch"},
        "params": {},
        "sandbox_provider": provider,
    }

    result = judge.invoke(payload)

    assert result["resolved"] is True
    assert "/workspace/entryscript.sh" not in agent_sandbox.writes
    entryscript = judge_sandbox.writes["/workspace/entryscript.sh"].decode("utf-8")
    assert "/run_scripts/instance_2/run_script.sh" in entryscript
    assert "/run_scripts/instance_2/parser.py" in entryscript
    assert "/workspace/run_script.sh" not in judge_sandbox.writes
    assert "/workspace/parser.py" not in judge_sandbox.writes
