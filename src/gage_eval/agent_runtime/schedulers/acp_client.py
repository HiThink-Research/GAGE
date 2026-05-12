from __future__ import annotations

from typing import Any

from gage_eval.agent_runtime.contracts.failure import FailureEnvelope
from gage_eval.agent_runtime.contracts.scheduler import SchedulerResult
from gage_eval.agent_runtime.session import AgentRuntimeSession


class AcpClientScheduler:
    """Phase-1 ACP scheduler stub."""

    async def arun(
        self,
        *,
        session: AgentRuntimeSession,
        sample: dict[str, Any],
        payload: dict[str, Any],
        workflow_bundle,
        sandbox_provider,
    ) -> SchedulerResult:
        del sample, payload, workflow_bundle, sandbox_provider
        failure = FailureEnvelope(
            failure_domain="client_execution",
            failure_stage="run_scheduler",
            failure_code="client_execution.scheduler.acp_unsupported",
            component_kind="scheduler",
            component_id="acp_client.scheduler",
            owner="runtime_scheduler_core",
            retryable=False,
            summary="acp_client scheduler is not supported in Phase 1",
            first_bad_step="acp_client.scheduler.run",
            suspect_files=("src/gage_eval/agent_runtime/schedulers/acp_client.py",),
            details={"scheduler_type": session.scheduler_type},
        )
        return SchedulerResult(
            scheduler_type="acp_client",
            benchmark_kit_id=session.benchmark_kit_id,
            status="failed",
            agent_output={},
            failure=failure,
        )
