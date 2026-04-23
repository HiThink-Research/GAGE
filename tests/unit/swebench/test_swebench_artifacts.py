from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

from gage_eval.agent_eval_kits.swebench.artifacts import persist_swebench_artifacts
from gage_eval.agent_runtime.session import AgentRuntimeSession


class _FakeSandbox:
    def read_file(self, path: str) -> bytes:
        assert path == "/workspace/submission.patch"
        return b"diff --git a/foo b/foo\n"

    def exec(self, command: str, timeout: int) -> SimpleNamespace:
        return SimpleNamespace(exit_code=0, stdout="", stderr="")


class _FakeProvider:
    def get_handle(self) -> SimpleNamespace:
        return SimpleNamespace(sandbox=_FakeSandbox())


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


def test_persist_swebench_artifacts_collects_missing_commands_from_full_trace(tmp_path: Path) -> None:
    session = _build_session(tmp_path)

    persist_swebench_artifacts(
        session=session,
        scheduler_output={
            "agent_trace": [
                {
                    "name": "run_shell",
                    "output": {"stderr": "/bin/sh: 1: earlycmd: not found\n"},
                },
                {"name": "noop", "output": {"stderr": ""}},
                {"name": "noop", "output": {"stderr": ""}},
                {"name": "noop", "output": {"stderr": ""}},
                {"name": "noop", "output": {"stderr": ""}},
                {"name": "run_shell", "output": {"stderr": "/bin/sh: 1: latecmd: not found\n"}},
            ],
        },
        sandbox_provider=None,
    )

    diagnostics = json.loads(
        (tmp_path / "sample" / "artifacts" / "swebench_diagnostics.json").read_text(encoding="utf-8")
    )

    assert diagnostics["missing_commands"] == ["earlycmd", "latecmd"]
    assert diagnostics["recent_errors"] == ["/bin/sh: 1: latecmd: not found"]


def test_persist_swebench_artifacts_resolves_prompt_metadata_across_context_fallbacks(tmp_path: Path) -> None:
    session = _build_session(tmp_path)
    session.prompt_context = {"prompt_present": True}
    session.runtime_context = {"prompt_source": "problem_statement"}

    persist_swebench_artifacts(
        session=session,
        scheduler_output={"agent_trace": []},
        sandbox_provider=None,
    )

    diagnostics = json.loads(
        (tmp_path / "sample" / "artifacts" / "swebench_diagnostics.json").read_text(encoding="utf-8")
    )

    assert diagnostics["prompt_present"] is True
    assert diagnostics["prompt_source"] == "problem_statement"


def test_persist_swebench_artifacts_expands_auditable_diagnostics(tmp_path: Path) -> None:
    session = _build_session(tmp_path)
    extra_dir = tmp_path / "sample" / "artifacts" / "tool-output"
    extra_dir.mkdir(parents=True, exist_ok=True)
    spill_path = extra_dir / "stdout.txt"
    spill_path.write_text("captured output", encoding="utf-8")

    persist_swebench_artifacts(
        session=session,
        scheduler_output={
            "answer": "final explanation",
            "prompt_present": True,
            "prompt_source": "problem_statement",
            "input_failure_code": "missing_prompt",
            "failure_category": "client_execution.tool_retry_budget_exhausted",
            "artifact_paths": {
                "tool_stdout": "artifacts/tool-output/stdout.txt",
            },
            "agent_trace": [
                {
                    "trace_step": 1,
                    "trace_role": "assistant",
                    "name": "agent_response",
                    "status": "retry_required_tool_call",
                    "output": {
                        "error": "required_tool_call_missing",
                        "retry_count": 1,
                        "total_invalid_tool_calls": 1,
                        "tool_call_parse_error_type": "missing_function_call",
                    },
                },
                {
                    "trace_step": 2,
                    "trace_role": "tool",
                    "name": "run_shell",
                    "input": {"command": "python -m pytest tests/unit -q"},
                    "output": {
                        "stdout": "head",
                        "stderr": "/bin/sh: 1: jq: not found\n",
                        "stdout_original_length": 2048,
                        "stderr_original_length": 128,
                        "truncated": True,
                        "head_preview": {"stdout": "head"},
                        "tail_preview": {"stdout": "tail"},
                    },
                },
                {
                    "trace_step": 3,
                    "trace_role": "tool",
                    "name": "run_shell",
                    "input": {"cmd": "git status --short"},
                    "output": {
                        "error": "tool_argument_invalid",
                        "error_code": "tool_argument_invalid",
                        "message": "argument 'command' must be string",
                    },
                },
                {
                    "trace_step": 4,
                    "trace_role": "assistant",
                    "name": "agent_response",
                    "status": "retry_required_tool_call",
                    "output": {
                        "error": "required_tool_call_missing",
                        "retry_count": 2,
                        "total_invalid_tool_calls": 3,
                    },
                },
            ],
        },
        sandbox_provider=None,
    )

    diagnostics = json.loads(
        (tmp_path / "sample" / "artifacts" / "swebench_diagnostics.json").read_text(encoding="utf-8")
    )

    assert diagnostics["prompt_present"] is True
    assert diagnostics["prompt_source"] == "problem_statement"
    assert diagnostics["input_failure_code"] == "missing_prompt"
    assert diagnostics["tool_call_retry_count"] == 2
    assert diagnostics["tool_call_retry_total"] == 3
    assert diagnostics["tool_call_parse_error_count"] == 1
    assert diagnostics["largest_tool_output_bytes"] == 2176
    assert diagnostics["largest_tool_name"] == "run_shell"
    assert diagnostics["largest_command_preview"] == "python -m pytest tests/unit -q"
    assert diagnostics["artifact_spillovers"] == 1
    assert diagnostics["final_failure_category"] == "client_execution.tool_retry_budget_exhausted"
    assert diagnostics["submission_patch_present"] is False
    assert diagnostics["agent_trace_step_count"] == 4
    assert diagnostics["answer_present"] is True
    assert diagnostics["missing_commands"] == ["jq"]
    assert diagnostics["recent_errors"][-1].endswith("jq: not found")


def test_persist_swebench_artifacts_infers_input_failure_before_patch_missing(tmp_path: Path) -> None:
    session = _build_session(tmp_path)

    persist_swebench_artifacts(
        session=session,
        scheduler_output={
            "input_failure_code": "input_projection.missing_problem_statement",
            "agent_trace": [],
        },
        sandbox_provider=None,
    )

    diagnostics = json.loads(
        (tmp_path / "sample" / "artifacts" / "swebench_diagnostics.json").read_text(encoding="utf-8")
    )

    assert diagnostics["final_failure_category"] == "input_projection.missing_problem_statement"
    assert diagnostics["submission_patch_present"] is False


def test_persist_swebench_artifacts_infers_tool_argument_invalid_before_patch_missing(tmp_path: Path) -> None:
    session = _build_session(tmp_path)

    persist_swebench_artifacts(
        session=session,
        scheduler_output={
            "agent_trace": [
                {
                    "trace_step": 1,
                    "trace_role": "tool",
                    "name": "run_shell",
                    "status": "error",
                    "output": {
                        "error": "command_missing",
                    },
                }
            ],
        },
        sandbox_provider=None,
    )

    diagnostics = json.loads(
        (tmp_path / "sample" / "artifacts" / "swebench_diagnostics.json").read_text(encoding="utf-8")
    )

    assert diagnostics["final_failure_category"] == "client_execution.tool_argument_invalid"


def test_persist_swebench_artifacts_marks_patch_missing_without_other_failure_signals(tmp_path: Path) -> None:
    session = _build_session(tmp_path)

    persist_swebench_artifacts(
        session=session,
        scheduler_output={"agent_trace": []},
        sandbox_provider=None,
    )

    diagnostics = json.loads(
        (tmp_path / "sample" / "artifacts" / "swebench_diagnostics.json").read_text(encoding="utf-8")
    )

    assert diagnostics["final_failure_category"] == "artifact_capture.patch_missing"


def test_persist_swebench_artifacts_does_not_mark_failure_when_patch_present(tmp_path: Path) -> None:
    session = _build_session(tmp_path)

    persist_swebench_artifacts(
        session=session,
        scheduler_output={
            "agent_trace": [
                {
                    "trace_step": 1,
                    "trace_role": "assistant",
                    "status": "retry_required_tool_call",
                    "output": {"error": "required_tool_call_missing"},
                },
                {
                    "trace_step": 2,
                    "trace_role": "tool",
                    "name": "submit_patch_tool",
                    "output": {"stdout": "diff --git a/foo b/foo\n", "exit_code": 0},
                },
            ],
        },
        sandbox_provider=None,
    )

    diagnostics = json.loads(
        (tmp_path / "sample" / "artifacts" / "swebench_diagnostics.json").read_text(encoding="utf-8")
    )

    assert diagnostics["submission_patch_present"] is True
    assert diagnostics["final_failure_category"] is None
