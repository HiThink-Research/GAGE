from __future__ import annotations

import json

import pytest

from gage_eval.agent_runtime.artifacts import RuntimeArtifactSink
from gage_eval.agent_runtime.contracts.failure import FailureEnvelope, FailureEnvelopeError
from gage_eval.agent_runtime.session import AgentRuntimeSession


class _DetailedError(RuntimeError):
    def __init__(self) -> None:
        self.details = {
            "sandbox_runtime": "docker",
            "runtime_handle": {"container_id": "cid-123"},
            "runtime_state": {"state_exit_code": 137, "state_oom_killed": True},
        }
        super().__init__("sandbox_crashed: exit_code=137 oom_killed=true")


@pytest.mark.fast
def test_runtime_artifact_sink_persists_error_details(tmp_path) -> None:
    sink = RuntimeArtifactSink(base_dir=str(tmp_path))
    layout = sink.build_layout(run_id="run-1", task_id="task-1", sample_id="sample-1")
    session = AgentRuntimeSession(
        session_id="session-1",
        run_id="run-1",
        task_id="task-1",
        sample_id="sample-1",
        benchmark_kit_id="swebench",
        scheduler_type="framework_loop",
        artifact_layout=layout,
    )

    target = sink.persist_raw_error(session=session, error=_DetailedError())
    payload = json.loads(open(target, encoding="utf-8").read())

    assert payload["error_type"] == "_DetailedError"
    assert payload["error"] == "sandbox_crashed: exit_code=137 oom_killed=true"
    assert payload["details"] == {
        "sandbox_runtime": "docker",
        "runtime_handle": {"container_id": "cid-123"},
        "runtime_state": {"state_exit_code": 137, "state_oom_killed": True},
    }


@pytest.mark.fast
def test_runtime_artifact_sink_persists_failure_envelope_details(tmp_path) -> None:
    sink = RuntimeArtifactSink(base_dir=str(tmp_path))
    layout = sink.build_layout(run_id="run-1", task_id="task-1", sample_id="sample-1")
    session = AgentRuntimeSession(
        session_id="session-1",
        run_id="run-1",
        task_id="task-1",
        sample_id="sample-1",
        benchmark_kit_id="swebench",
        scheduler_type="framework_loop",
        artifact_layout=layout,
    )
    failure = FailureEnvelope(
        failure_domain="environment",
        failure_stage="run_scheduler",
        failure_code="environment.run_scheduler.swebench.framework_loop.sandbox_crashed",
        component_kind="runtime",
        component_id="swebench.framework_loop.sandbox",
        owner="runtime_sandbox_core",
        retryable=False,
        summary="sandbox_crashed: exit_code=137 oom_killed=true error=oom-killed",
        first_bad_step="swebench.framework_loop.agent_loop.run",
        suspect_files=("src/gage_eval/role/agent/loop.py",),
        details={
            "sandbox_runtime": "docker",
            "runtime_handle": {"container_id": "cid-123"},
            "runtime_state": {"state_exit_code": 137, "state_oom_killed": True},
        },
    )

    target = sink.persist_raw_error(session=session, error=FailureEnvelopeError(failure))
    payload = json.loads(open(target, encoding="utf-8").read())

    assert payload["error_type"] == "FailureEnvelopeError"
    assert payload["error"] == "sandbox_crashed: exit_code=137 oom_killed=true error=oom-killed"
    assert payload["details"] == failure.details
