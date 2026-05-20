from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

from gage_eval.agent_eval_kits.swebench.artifacts import (
    persist_swebench_artifacts,
    update_swebench_diagnostics_post_verifier,
)
from gage_eval.agent_runtime.compiled_plan import CompiledRuntimePlan, SchedulerWorkflowBundle
from gage_eval.agent_runtime.contracts.scheduler import SchedulerResult
from gage_eval.agent_runtime.executor import DefaultVerifierRunner
from gage_eval.agent_runtime.session import AgentRuntimeSession
from gage_eval.agent_runtime.verifier.binding import JudgeBinding
from gage_eval.agent_runtime.verifier.contracts import VerifierInput, VerifierResult


DIFF = "diff --git a/foo b/foo\n--- a/foo\n+++ b/foo\n@@ -1 +1 @@\n-old\n+new\n"


class _FakeSandbox:
    def read_file(self, path: str) -> bytes:
        assert path == "/workspace/submission.patch"
        return b"diff --git a/foo b/foo\n"

    def exec(self, command: str, timeout: int) -> SimpleNamespace:
        return SimpleNamespace(exit_code=0, stdout="", stderr="")


class _FakeProvider:
    def get_handle(self) -> SimpleNamespace:
        return SimpleNamespace(sandbox=_FakeSandbox())


class _AssertionErrorVerifier:
    judge_source = "swebench.test"

    def run(self, verifier_input: VerifierInput) -> VerifierResult:
        del verifier_input
        return VerifierResult(
            status="failed",
            payload={
                "status": "failed",
                "resolved": False,
                "score": 0.0,
                "failure_reason": "assertion_error",
                "patch_applied_via": "git_apply",
            },
        )


def _build_session(tmp_path: Path) -> AgentRuntimeSession:
    sample_root = tmp_path / "sample"
    artifacts_dir = sample_root / "artifacts"
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    return AgentRuntimeSession(
        session_id="session-1",
        run_id="run-1",
        task_id="task-1",
        sample_id="sample-1",
        benchmark_kit_id="swebench",
        scheduler_type="framework_loop",
        artifact_layout={
            "sample_root": str(sample_root),
            "artifacts_dir": str(artifacts_dir),
            "verifier_result": str(sample_root / "verifier" / "verifier_result.json"),
        },
    )


def test_persist_swebench_artifacts_materializes_patch_and_trace(tmp_path: Path) -> None:
    session = _build_session(tmp_path)

    artifact_paths = persist_swebench_artifacts(
        session=session,
        scheduler_output={
            "answer": "final explanation",
            "agent_trace": [{"trace_step": 1, "name": "submit_patch_tool", "output": {"stdout": "diff --git a/foo b/foo"}}],
        },
        sandbox_provider=_FakeProvider(),
    )

    assert artifact_paths["submission_patch"] == "artifacts/submission.patch"
    assert artifact_paths["agent_trace"] == "artifacts/agent_trace.json"
    assert artifact_paths["final_response"] == "artifacts/final_response.txt"
    assert (tmp_path / "sample" / "artifacts" / "submission.patch").exists()
    assert json.loads((tmp_path / "sample" / "artifacts" / "agent_trace.json").read_text(encoding="utf-8"))[0]["name"] == "submit_patch_tool"


def test_persist_swebench_artifacts_materializes_diff_from_answer_field(tmp_path: Path) -> None:
    session = _build_session(tmp_path)

    artifact_paths = persist_swebench_artifacts(
        session=session,
        scheduler_output={"answer": f"```diff\n{DIFF}```\n"},
    )

    assert artifact_paths["submission_patch"] == "artifacts/submission.patch"
    assert (tmp_path / "sample" / "artifacts" / "submission.patch").read_text(encoding="utf-8") == DIFF
    diagnostics = json.loads(
        (tmp_path / "sample" / "artifacts" / "swebench_diagnostics.json").read_text(encoding="utf-8")
    )
    assert diagnostics["submission_patch_present"] is True


def test_persist_swebench_artifacts_redacts_secret_text(tmp_path: Path) -> None:
    session = _build_session(tmp_path)

    persist_swebench_artifacts(
        session=session,
        scheduler_output={
            "answer": "api_key=abc123",
            "agent_trace": [
                {
                    "name": "run_shell",
                    "output": {"stderr": "Authorization: Bearer abc123"},
                }
            ],
        },
        sandbox_provider=None,
    )

    artifact_dir = tmp_path / "sample" / "artifacts"
    serialized = "\n".join(
        [
            (artifact_dir / "agent_trace.json").read_text(encoding="utf-8"),
            (artifact_dir / "final_response.txt").read_text(encoding="utf-8"),
            (artifact_dir / "swebench_diagnostics.json").read_text(encoding="utf-8"),
        ]
    )
    assert "Bearer abc123" not in serialized
    assert "api_key=abc123" not in serialized
    assert "<redacted:" in serialized


def test_persist_swebench_artifacts_surfaces_missing_commands(tmp_path: Path) -> None:
    session = _build_session(tmp_path)

    persist_swebench_artifacts(
        session=session,
        scheduler_output={
            "answer": "final explanation",
            "agent_trace": [
                {
                    "name": "run_shell",
                    "output": {
                        "stderr": "/bin/sh: 1: jq: not found\n/bin/sh: 1: trivy-to-vuls: not found\n"
                    },
                }
            ],
        },
        sandbox_provider=None,
    )

    diagnostics = json.loads(
        (tmp_path / "sample" / "artifacts" / "swebench_diagnostics.json").read_text(encoding="utf-8")
    )

    assert diagnostics["missing_commands"] == ["jq", "trivy-to-vuls"]
    assert diagnostics["recent_errors"][-1].endswith("trivy-to-vuls: not found")


def test_persist_swebench_artifacts_counts_nested_output_artifact_refs(tmp_path: Path) -> None:
    session = _build_session(tmp_path)

    persist_swebench_artifacts(
        session=session,
        scheduler_output={
            "agent_trace": [
                {
                    "name": "run_shell",
                    "output": {
                        "stdout": "<spilled to artifact: stdout.txt>",
                        "output_artifact_refs": [{"owner": "agent", "path": "stdout.txt", "size_bytes": 8105}],
                    },
                }
            ],
        },
        sandbox_provider=None,
    )

    diagnostics = json.loads(
        (tmp_path / "sample" / "artifacts" / "swebench_diagnostics.json").read_text(encoding="utf-8")
    )

    assert diagnostics["artifact_spillover_count"] == 1
    assert diagnostics["max_tool_output_bytes"] == 8105


def test_persist_swebench_artifacts_classifies_failure_from_collected_trace_stats(tmp_path: Path) -> None:
    session = _build_session(tmp_path)

    persist_swebench_artifacts(
        session=session,
        scheduler_output={
            "status": "failed",
            "failure_code": "client_execution.run_scheduler.swebench.agent_loop_failed",
            "loop_exit_reason": "max_turns",
            "agent_trace": [
                {
                    "name": "run_shell",
                    "output": {
                        "command": "cat src/large.py",
                        "stdout": "<spilled to artifact: stdout.txt>",
                        "output_artifact_refs": [
                            {"owner": "agent", "path": f"stdout-{index}.txt", "size_bytes": 120000}
                        ],
                    },
                }
                for index in range(3)
            ],
        },
        sandbox_provider=None,
    )

    diagnostics = json.loads(
        (tmp_path / "sample" / "artifacts" / "swebench_diagnostics.json").read_text(encoding="utf-8")
    )

    assert diagnostics["artifact_spillover_count"] == 3
    assert diagnostics["max_tool_output_bytes"] == 120000
    assert diagnostics["final_failure_category"] == "context_overflow_from_listing"


def test_update_swebench_diagnostics_post_verifier_merges_judge_verdict(tmp_path: Path) -> None:
    session = _build_session(tmp_path)
    persist_swebench_artifacts(
        session=session,
        scheduler_output={
            "status": "completed",
            "answer": DIFF,
        },
        sandbox_provider=None,
    )

    update_swebench_diagnostics_post_verifier(
        session=session,
        verifier_output={
            "status": "failed",
            "resolved": False,
            "score": 0.0,
            "failure_reason": "assertion_error",
            "failure_category": "unknown",
            "patch_applied_via": "git_apply",
        },
    )

    diagnostics = json.loads(
        (tmp_path / "sample" / "artifacts" / "swebench_diagnostics.json").read_text(encoding="utf-8")
    )

    assert diagnostics["failure_reason"] == "assertion_error"
    assert diagnostics["patch_applied_via"] == "git_apply"
    assert diagnostics["score"] == 0.0
    assert diagnostics["final_failure_category"] == "wrong_solution"


def test_default_verifier_runner_updates_swebench_diagnostics_after_judge(tmp_path: Path) -> None:
    session = _build_session(tmp_path)
    artifact_paths = persist_swebench_artifacts(
        session=session,
        scheduler_output={
            "status": "completed",
            "answer": DIFF,
        },
        sandbox_provider=None,
    )
    scheduler_result = SchedulerResult(
        scheduler_type="framework_loop",
        benchmark_kit_id="swebench",
        status="completed",
        agent_output={"answer": DIFF, "artifact_paths": artifact_paths},
        artifact_paths=artifact_paths,
    )

    DefaultVerifierRunner().run(
        plan=_compiled_swebench_plan(_AssertionErrorVerifier()),
        session=session,
        sample={"id": "sample-1"},
        scheduler_result=scheduler_result,
    )

    diagnostics = json.loads(
        (tmp_path / "sample" / "artifacts" / "swebench_diagnostics.json").read_text(encoding="utf-8")
    )

    assert diagnostics["failure_reason"] == "assertion_error"
    assert diagnostics["patch_applied_via"] == "git_apply"
    assert diagnostics["final_failure_category"] == "wrong_solution"


def _compiled_swebench_plan(adapter: object) -> CompiledRuntimePlan:
    workflow = SchedulerWorkflowBundle(
        bundle_id="swebench.framework_loop",
        benchmark_kit_id="swebench",
        scheduler_type="framework_loop",
    )
    return CompiledRuntimePlan(
        run_id="run-1",
        dut_id="dut-1",
        agent_id="agent-1",
        env_id="env-1",
        benchmark_id="bench-1",
        trial_policy={},
        kit_id="swebench",
        kit_entry=None,
        kit_config={},
        agent_config={},
        scheduler_type="framework_loop",
        scheduler_config={},
        environment_provider="docker",
        environment_profile_id="swebench_runtime",
        environment_profile={},
        lifecycle="per_sample",
        provider_config={},
        startup_env={},
        resources={},
        verifier_environment_policy="reuse",
        verifier_environment_profile_id=None,
        workflow_bundle=workflow,
        tool_registry=None,
        tool_provider_adapter=None,
        verifier_adapter=adapter,
        artifact_sink=None,
        judge_binding=JudgeBinding(
            judge_mode="runtime_verifier",
            benchmark_kit_id="swebench",
            verifier_kind="native",
            verifier_resource_refs={"adapter": adapter},
        ),
    )
