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
