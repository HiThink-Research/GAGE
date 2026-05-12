from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

from gage_eval.agent_runtime.artifacts import RuntimeArtifactSink
from gage_eval.agent_runtime.resources.contracts import ResourceLease
from gage_eval.agent_runtime.schedulers.framework_loop import _dispatch_runtime_tool_call
from gage_eval.agent_runtime.session import AgentRuntimeSession
from gage_eval.agent_runtime.tooling.contracts import ToolCallIR, ToolResultIR
from gage_eval.agent_runtime.trace_schema import TRACE_INLINE_TEXT_LIMIT_BYTES


class _LargeOutputRouter:
    def __init__(self, stdout: str) -> None:
        self.stdout = stdout

    async def dispatch(self, call: ToolCallIR, context: Any) -> ToolResultIR:
        return ToolResultIR(
            call_id=call.call_id,
            name=call.name,
            provider=call.provider,
            status="success",
            output_text=self.stdout,
            output_json={
                "command": "cat src/large.js",
                "exit_code": 0,
                "stdout": self.stdout,
                "stderr": "",
                "output_artifact_refs": [],
            },
            raw_output={"stdout": self.stdout},
            metadata={"latency_ms": 1.0},
        )


def test_framework_loop_spills_oversize_tool_stdout_before_trace(tmp_path: Path) -> None:
    stdout = "x" * (TRACE_INLINE_TEXT_LIMIT_BYTES + 128)
    sink = RuntimeArtifactSink(base_dir=str(tmp_path))
    agent_trace: list[dict[str, Any]] = []
    call = ToolCallIR(
        call_id="call-large",
        name="run_shell",
        arguments_json='{"command":"cat src/large.js"}',
        raw_message={"id": "call-large", "function": {"name": "run_shell"}},
    )

    result = asyncio.run(
        _dispatch_runtime_tool_call(
            call=call,
            tool_router=_LargeOutputRouter(stdout),
            session=_session(),
            payload={"artifact_sink": sink},
            sink=sink,
            trial_id="trial_0001",
            turn_index=1,
            agent_trace=agent_trace,
        )
    )

    assert result.output_json["stdout"] == stdout
    assert len(agent_trace) == 1
    trace_step_output = agent_trace[0]["output"]
    assert len(trace_step_output["stdout"].encode("utf-8")) <= TRACE_INLINE_TEXT_LIMIT_BYTES
    assert trace_step_output["output_artifact_refs"]

    trace_path = tmp_path / "run-1/artifacts/task-1/sample-1/trials/trial_0001/infra/trace.jsonl"
    events = [json.loads(line) for line in trace_path.read_text(encoding="utf-8").splitlines()]
    tool_result_event = next(event for event in events if event["event_type"] == "tool.result")
    output_json = tool_result_event["payload"]["tool_result"]["output_json"]

    assert len(output_json["stdout"].encode("utf-8")) <= TRACE_INLINE_TEXT_LIMIT_BYTES
    assert output_json["stdout"].startswith("<spilled to artifact")
    assert "head -n 50" in output_json["stdout"]
    assert "grep -n PATTERN" in output_json["stdout"]
    assert output_json["output_artifact_refs"]
    assert tool_result_event["artifact_refs"] == output_json["output_artifact_refs"]

    artifact_ref = output_json["output_artifact_refs"][0]
    artifact_path = tmp_path / "run-1" / artifact_ref["path"]
    assert artifact_path.read_text(encoding="utf-8") == stdout


def _session() -> AgentRuntimeSession:
    return AgentRuntimeSession(
        session_id="session-1",
        run_id="run-1",
        task_id="task-1",
        sample_id="sample-1",
        benchmark_kit_id="demo",
        scheduler_type="framework_loop",
        resource_lease=ResourceLease(
            lease_id="lease-1",
            resource_kind="local_process",
            profile_id="demo",
            lifecycle="per_sample",
        ),
    )
