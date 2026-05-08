from __future__ import annotations

import json
from pathlib import Path

import pytest

from gage_eval.agent_eval_kits.swebench.judge.adapters import SwebenchVerifierAdapter
from gage_eval.agent_runtime.verifier.contracts import VerifierInput
from gage_eval.environment.contracts import ExecResult


DIFF = "diff --git a/foo b/foo\n--- a/foo\n+++ b/foo\n@@ -1 +1 @@\n-old\n+new\n"


class FakeEnvironment:
    env_id = "env-1"
    name = "fake"
    provider = "fake"
    metadata = {}
    capabilities = {}

    def __init__(self) -> None:
        self.writes: dict[str, bytes] = {}
        self.exec_calls: list[str] = []

    async def write_file(self, path: str, content: bytes | str) -> None:
        self.writes[path] = content.encode("utf-8") if isinstance(content, str) else content

    async def read_file(self, path: str, *, max_bytes: int = 16 * 1024 * 1024) -> bytes:
        del max_bytes
        if path == "/workspace/output.json":
            return json.dumps({"tests": [{"name": "tests/test_fix.py::test_fix", "status": "PASSED"}]}).encode()
        if path in self.writes:
            return self.writes[path]
        raise FileNotFoundError(path)

    async def exec(self, command: str, **kwargs) -> ExecResult:
        del kwargs
        self.exec_calls.append(command)
        return ExecResult(command=command, exit_code=0)


class FakeLease:
    def __init__(self, environment: FakeEnvironment) -> None:
        self.environment = environment
        self.lease_id = "lease-1"
        self.provider = "fake"
        self.profile_id = "profile"

    async def write_file(self, path: str, content: bytes | str) -> None:
        await self.environment.write_file(path, content)

    async def read_file(self, path: str, *, max_bytes: int = 16 * 1024 * 1024) -> bytes:
        return await self.environment.read_file(path, max_bytes=max_bytes)

    async def exec(self, command: str, **kwargs) -> ExecResult:
        return await self.environment.exec(command, **kwargs)


class FakeTrace:
    def __init__(self) -> None:
        self.events: list[tuple[str, dict, str | None]] = []

    def emit(self, event: str, payload: dict, sample_id: str | None = None) -> None:
        self.events.append((event, payload, sample_id))


def _write_scripts(root: Path, instance_id: str = "instance_1") -> None:
    run_dir = root / instance_id
    run_dir.mkdir(parents=True)
    (run_dir / "run_script.sh").write_text("pytest\n", encoding="utf-8")
    (run_dir / "parser.py").write_text("print('parse')\n", encoding="utf-8")


def test_adapter_runs_kit_owned_verifier_and_emits_result_trace(tmp_path: Path) -> None:
    scripts_dir = tmp_path / "run_scripts"
    _write_scripts(scripts_dir)
    trace = FakeTrace()
    env = FakeEnvironment()

    result = SwebenchVerifierAdapter(scripts_dir=str(scripts_dir)).run(
        VerifierInput(
            benchmark_kit_id="swebench",
            scheduler_type="framework_loop",
            sample_id="sample-1",
            sample={
                "id": "instance_1",
                "metadata": {
                    "instance_id": "instance_1",
                    "repo": "repo/name",
                    "base_commit": "abc123",
                    "fail_to_pass": ["tests/test_fix.py::test_fix"],
                    "pass_to_pass": [],
                },
            },
            scheduler_result={"agent_output": {"answer": DIFF}},
            runtime_context={"environment_lease": FakeLease(env), "trace": trace, "trial_id": "trial_0001"},
            verifier_resources={},
        )
    )

    assert result.status == "completed"
    assert result.payload["resolved"] is True
    assert result.payload["metric"]["score"] == 1.0
    event = next(item for item in trace.events if item[0] == "verifier.result")
    assert event[1]["metric"]["score"] == 1.0
    assert event[1]["artifact_refs"]
    source_event = next(item for item in trace.events if item[0] == "patch.source.resolved")
    assert source_event[1]["source"] == "model_output"


def test_adapter_suppresses_test_patch_for_swebench_pro_samples(tmp_path: Path) -> None:
    scripts_dir = tmp_path / "run_scripts"
    _write_scripts(scripts_dir)
    env = FakeEnvironment()

    result = SwebenchVerifierAdapter(scripts_dir=str(scripts_dir)).run(
        VerifierInput(
            benchmark_kit_id="swebench",
            scheduler_type="framework_loop",
            sample_id="sample-1",
            sample={
                "id": "instance_1",
                "_gage_dataset_id": "swebench_pro_smoke11",
                "metadata": {
                    "instance_id": "instance_1",
                    "repo": "repo/name",
                    "base_commit": "abc123",
                    "test_patch": DIFF,
                    "fail_to_pass": ["tests/test_fix.py::test_fix"],
                    "pass_to_pass": [],
                },
            },
            scheduler_result={"agent_output": {"answer": DIFF}},
            runtime_context={"environment_lease": FakeLease(env)},
            verifier_resources={},
        )
    )

    assert result.payload["resolved"] is True
    assert "/workspace/test_patch.diff" not in env.writes
    entryscript = env.writes["/workspace/entryscript.sh"].decode("utf-8")
    assert "test_patch.diff" not in entryscript
    assert "git_apply_lenient" not in entryscript


def test_adapter_pro_mode_does_not_depend_on_dataset_name_heuristics(tmp_path: Path) -> None:
    scripts_dir = tmp_path / "run_scripts"
    _write_scripts(scripts_dir)
    env = FakeEnvironment()

    result = SwebenchVerifierAdapter(scripts_dir=str(scripts_dir)).run(
        VerifierInput(
            benchmark_kit_id="swebench",
            scheduler_type="framework_loop",
            sample_id="sample-1",
            sample={
                "id": "instance_1",
                "dataset_id": "scaleai/swe-bench-pro",
                "metadata": {
                    "instance_id": "instance_1",
                    "repo": "repo/name",
                    "base_commit": "abc123",
                    "test_patch": DIFF,
                    "fail_to_pass": ["tests/test_fix.py::test_fix"],
                    "pass_to_pass": [],
                },
            },
            scheduler_result={"agent_output": {"answer": DIFF}},
            runtime_context={"environment_lease": FakeLease(env)},
            verifier_resources={},
        )
    )

    assert result.payload["resolved"] is True
    assert "/workspace/test_patch.diff" not in env.writes
    entryscript = env.writes["/workspace/entryscript.sh"].decode("utf-8")
    assert "test_patch.diff" not in entryscript
    assert "git_apply_lenient" not in entryscript


def test_adapter_non_pro_mode_applies_test_patch_and_allows_lenient_apply(tmp_path: Path) -> None:
    scripts_dir = tmp_path / "run_scripts"
    _write_scripts(scripts_dir)
    env = FakeEnvironment()

    result = SwebenchVerifierAdapter(scripts_dir=str(scripts_dir), swebench_pro_mode=False).run(
        VerifierInput(
            benchmark_kit_id="swebench",
            scheduler_type="framework_loop",
            sample_id="sample-1",
            sample={
                "id": "instance_1",
                "metadata": {
                    "instance_id": "instance_1",
                    "repo": "repo/name",
                    "base_commit": "abc123",
                    "test_patch": DIFF,
                    "fail_to_pass": ["tests/test_fix.py::test_fix"],
                    "pass_to_pass": [],
                },
            },
            scheduler_result={"agent_output": {"answer": DIFF}},
            runtime_context={"environment_lease": FakeLease(env)},
            verifier_resources={},
        )
    )

    assert result.payload["resolved"] is True
    assert env.writes["/workspace/test_patch.diff"].decode("utf-8") == DIFF
    entryscript = env.writes["/workspace/entryscript.sh"].decode("utf-8")
    assert "test_patch.diff" in entryscript
    assert "git_apply_lenient" in entryscript


def test_missing_patch_maps_to_artifact_capture_patch_missing(tmp_path: Path) -> None:
    scripts_dir = tmp_path / "run_scripts"
    _write_scripts(scripts_dir)

    result = SwebenchVerifierAdapter(scripts_dir=str(scripts_dir)).run(
        VerifierInput(
            benchmark_kit_id="swebench",
            scheduler_type="framework_loop",
            sample_id="sample-1",
            sample={"id": "instance_1", "metadata": {"instance_id": "instance_1", "repo": "repo/name", "base_commit": "abc"}},
            scheduler_result={"agent_output": {}},
            runtime_context={"environment_lease": FakeLease(FakeEnvironment())},
            verifier_resources={},
        )
    )

    assert result.status == "failed"
    assert result.payload["failure_reason"] == "missing_patch"
    assert result.payload["failure_code"] == "artifact_capture.patch_missing"


def test_missing_base_commit_fails_fast_before_environment_or_patch_resolution(tmp_path: Path) -> None:
    scripts_dir = tmp_path / "run_scripts"
    _write_scripts(scripts_dir)
    env = FakeEnvironment()

    result = SwebenchVerifierAdapter(scripts_dir=str(scripts_dir)).run(
        VerifierInput(
            benchmark_kit_id="swebench",
            scheduler_type="framework_loop",
            sample_id="sample-1",
            sample={"id": "instance_1", "metadata": {"instance_id": "instance_1", "repo": "repo/name"}},
            scheduler_result={"agent_output": {"answer": DIFF}},
            runtime_context={"environment_lease": FakeLease(env)},
            verifier_resources={},
        )
    )

    assert result.status == "failed"
    assert result.payload["failure_code"] == "config.kit_schema.validation_failed"
    assert result.payload["failure_reason"] == "missing_base_commit"
    assert env.writes == {}
    assert env.exec_calls == []


def test_missing_base_commit_preflight_returns_config_failure() -> None:
    result = SwebenchVerifierAdapter().preflight(
        VerifierInput(
            benchmark_kit_id="swebench",
            scheduler_type="framework_loop",
            sample_id="sample-1",
            sample={"id": "instance_1", "metadata": {"instance_id": "instance_1", "repo": "repo/name"}},
            scheduler_result={"agent_output": {"answer": DIFF}},
            runtime_context={},
            verifier_resources={},
        )
    )

    assert result is not None
    assert result.status == "failed"
    assert result.payload["failure_code"] == "config.kit_schema.validation_failed"
    assert result.payload["failure_reason"] == "missing_base_commit"


def test_workflow_prepare_failure_maps_failure_code(tmp_path: Path) -> None:
    missing_scripts_dir = tmp_path / "missing"

    result = SwebenchVerifierAdapter(scripts_dir=str(missing_scripts_dir)).run(
        VerifierInput(
            benchmark_kit_id="swebench",
            scheduler_type="framework_loop",
            sample_id="sample-1",
            sample={
                "id": "instance_1",
                "metadata": {"instance_id": "instance_1", "repo": "repo/name", "base_commit": "abc"},
            },
            scheduler_result={"agent_output": {"answer": DIFF}},
            runtime_context={"environment_lease": FakeLease(FakeEnvironment())},
            verifier_resources={},
        )
    )

    assert result.status == "failed"
    assert result.payload["failure_reason"] == "missing_run_scripts"
    assert result.payload["failure_code"] == "input_projection.workflow.prepare_failed"
