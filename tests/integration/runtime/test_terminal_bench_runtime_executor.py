from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

from gage_eval.agent_runtime import build_compiled_runtime_executor, compile_agent_runtime_plan


class _InstalledClientStub:
    def invoke(self, payload: dict) -> dict:
        return {
            "answer": "done",
            "agent_trace": [],
            "artifact_paths": {"stdout": "stdout.txt"},
        }


def test_terminal_bench_installed_client_executor(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("GAGE_EVAL_SAVE_DIR", str(tmp_path))

    plan = compile_agent_runtime_plan(agent_runtime_id="terminal_bench_installed_client")
    plan = replace(
        plan,
        resource_plan={"resource_kind": "docker", "sandbox_config": {}},
    )
    executor = build_compiled_runtime_executor(
        compiled_plan=plan,
        agent_backend=_InstalledClientStub(),
        max_turns=4,
    )

    output = executor.compiled_plan.scheduler_handle
    assert output is not None

    result = __import__("asyncio").run(
        executor.aexecute(
            sample={
                "id": "terminal-1",
                "instruction": "say done",
                "expected_answer": "done",
                "messages": [{"role": "user", "content": "say done"}],
            },
            payload={
                "sample": {
                    "id": "terminal-1",
                    "instruction": "say done",
                    "expected_answer": "done",
                    "messages": [{"role": "user", "content": "say done"}],
                },
                "execution_context": {
                    "run_id": "run-terminal",
                    "task_id": "task-terminal",
                    "sample_id": "terminal-1",
                },
            },
        )
    )

    assert result["answer"] == "done"
    assert result["runtime_judge_outcome"]["judge_output"]["resolved"] is True
    assert result["runtime_session"]["runtime_metadata_path"].endswith("runtime_metadata.json")
    sample_root = Path(result["runtime_session"]["sample_root"])
    assert (sample_root / "artifacts" / "tool_trace.json").exists()
    assert (sample_root / "artifacts" / "stdout.log").exists()
    assert (sample_root / "artifacts" / "workspace_diff.json").exists()

    verifier_payload = json.loads(
        Path(result["runtime_session"]["verifier_result_path"]).read_text(encoding="utf-8")
    )
    assert verifier_payload["verifier_input"]["verifier_resources"] == {}
    assert verifier_payload["judge_output"]["artifact_paths"]["tool_trace"] == "artifacts/tool_trace.json"
