from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest

from gage_eval.agent_runtime.contracts.failure import FailureEnvelopeError
from gage_eval.agent_runtime.schedulers.framework_loop import FrameworkLoopScheduler
from gage_eval.agent_runtime.session import AgentRuntimeSession
from gage_eval.role.agent.tool_router import ToolRouter


class _IdleBackend:
    def invoke(self, payload):
        raise AssertionError("backend should not be invoked after sandbox crash")


class _CrashySandbox:
    def is_alive(self, timeout_s=None):
        return False

    def describe_runtime_state(self, timeout_s=None):
        return {
            "container_id": "cid-123",
            "container_name": "gage-sandbox-test",
            "state_status": "exited",
            "state_exit_code": 137,
            "state_oom_killed": True,
            "state_error": "oom-killed",
            "logs_tail": "fatal: container exited",
        }


class _Provider:
    def get_handle(self):
        return SimpleNamespace(
            sandbox=_CrashySandbox(),
            runtime_handle={"container_id": "cid-123", "container_name": "gage-sandbox-test"},
        )


class _WorkflowBundle:
    bundle_id = "swebench.framework_loop"
    build_loop_inputs = None
    inject_prompt_context = None
    inject_tool_schemas = None
    finalize_loop_result = None
    failure_normalizer = None


@pytest.mark.fast
def test_framework_loop_scheduler_surfaces_structured_sandbox_crash() -> None:
    scheduler = FrameworkLoopScheduler(
        backend=_IdleBackend(),
        tool_router=ToolRouter(),
        max_turns=2,
    )
    session = AgentRuntimeSession(
        session_id="session-1",
        run_id="run-1",
        task_id="task-1",
        sample_id="sample-1",
        benchmark_kit_id="swebench",
        scheduler_type="framework_loop",
    )

    with pytest.raises(FailureEnvelopeError) as exc_info:
        asyncio.run(
            scheduler.arun(
                session=session,
                sample={"messages": [{"role": "user", "content": "hi"}]},
                payload={},
                workflow_bundle=_WorkflowBundle(),
                sandbox_provider=_Provider(),
            )
        )

    failure = exc_info.value.failure
    assert failure.failure_domain == "environment"
    assert failure.failure_stage == "run_scheduler"
    assert failure.failure_code == "environment.run_scheduler.swebench.framework_loop.sandbox_crashed"
    assert failure.summary == "sandbox_crashed: exit_code=137 oom_killed=true error=oom-killed"
    assert failure.details == {
        "sandbox_runtime": "unknown",
        "runtime_handle": {"container_id": "cid-123", "container_name": "gage-sandbox-test"},
        "runtime_state": {
            "container_id": "cid-123",
            "container_name": "gage-sandbox-test",
            "state_status": "exited",
            "state_exit_code": 137,
            "state_oom_killed": True,
            "state_error": "oom-killed",
            "logs_tail": "fatal: container exited",
        },
    }
