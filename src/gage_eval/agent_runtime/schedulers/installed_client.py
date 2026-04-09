from __future__ import annotations

from typing import Any

from gage_eval.agent_runtime.clients.runner import InstalledClientRunner
from gage_eval.agent_runtime.contracts.failure import FailureEnvelopeError
from gage_eval.agent_runtime.contracts.scheduler import SchedulerResult
from gage_eval.agent_runtime.failure_mapper import FailureMapper
from gage_eval.agent_runtime.session import AgentRuntimeSession


class InstalledClientScheduler:
    """Runs one installed client against the prepared benchmark request."""

    def __init__(self, client: Any) -> None:
        self._runner = InstalledClientRunner(client)
        self._failure_mapper = FailureMapper()

    async def arun(
        self,
        *,
        session: AgentRuntimeSession,
        sample: dict[str, Any],
        payload: dict[str, Any],
        workflow_bundle,
        sandbox_provider,
    ) -> SchedulerResult:
        """Execute the installed-client workflow and normalize its result."""

        # STEP 1: Build scheduler inputs and environment-specific context.
        request_payload = {
            "sample": sample,
            "payload": payload,
            "session": session,
            "sandbox_provider": sandbox_provider,
        }
        workflow_path = _resolve_workflow_path(session)
        if callable(workflow_bundle.prepare_environment):
            try:
                environment_state = workflow_bundle.prepare_environment(
                    session=session,
                    sample=sample,
                    sandbox_provider=sandbox_provider,
                )
            except Exception as exc:
                raise FailureEnvelopeError(
                    self._failure_mapper.map_exception(
                        exc,
                        failure_domain="environment",
                        failure_stage="prepare_inputs",
                        component_kind="scheduler",
                        component_id=f"{workflow_bundle.bundle_id}.prepare_environment",
                        owner=f"{session.benchmark_kit_id}_kit",
                        failure_code=(
                            f"environment.prepare_inputs."
                            f"{workflow_bundle.bundle_id}.prepare_environment_failed"
                        ),
                        first_bad_step=f"{workflow_bundle.bundle_id}.prepare_environment",
                        suspect_files=(
                            workflow_path,
                            "src/gage_eval/agent_runtime/schedulers/installed_client.py",
                        ),
                    )
                ) from exc
            if isinstance(environment_state, dict):
                session.scheduler_state.update(environment_state)
        if callable(workflow_bundle.prepare_inputs):
            try:
                request_payload.update(
                    workflow_bundle.prepare_inputs(
                        session=session,
                        sample=sample,
                        payload=payload,
                        sandbox_provider=sandbox_provider,
                    )
                )
            except Exception as exc:
                raise FailureEnvelopeError(
                    self._failure_mapper.map_exception(
                        exc,
                        failure_domain="input_projection",
                        failure_stage="prepare_inputs",
                        component_kind="scheduler",
                        component_id=f"{workflow_bundle.bundle_id}.prepare_inputs",
                        owner=f"{session.benchmark_kit_id}_kit",
                        failure_code=(
                            f"input_projection.prepare_inputs."
                            f"{workflow_bundle.bundle_id}.prepare_inputs_failed"
                        ),
                        first_bad_step=f"{workflow_bundle.bundle_id}.prepare_inputs",
                        suspect_files=(
                            workflow_path,
                            "src/gage_eval/agent_runtime/schedulers/installed_client.py",
                        ),
                    )
                ) from exc

        # STEP 2: Run the external client surface.
        try:
            raw_result = await self._runner.arun(request_payload)
        except Exception as exc:
            raise FailureEnvelopeError(
                self._failure_mapper.map_exception(
                    exc,
                    failure_domain="client_execution",
                    failure_stage="run_scheduler",
                    component_kind="client",
                    component_id=f"{workflow_bundle.bundle_id}.installed_client.run",
                    owner="runtime_scheduler_core",
                    failure_code=(
                        f"client_execution.run_scheduler."
                        f"{workflow_bundle.bundle_id}.installed_client_failed"
                    ),
                    first_bad_step=f"{workflow_bundle.bundle_id}.installed_client.run",
                    suspect_files=(
                        "src/gage_eval/agent_runtime/clients/runner.py",
                        "src/gage_eval/agent_runtime/schedulers/installed_client.py",
                    ),
                )
            ) from exc

        # STEP 3: Collect artifacts and return a stable scheduler result.
        artifact_paths: dict[str, str] = {}
        if callable(workflow_bundle.capture_environment_artifacts):
            try:
                captured = workflow_bundle.capture_environment_artifacts(
                    session=session,
                    sample=sample,
                    scheduler_output=raw_result,
                    sandbox_provider=sandbox_provider,
                )
            except Exception as exc:
                raise FailureEnvelopeError(
                    self._failure_mapper.map_exception(
                        exc,
                        failure_domain="artifact_capture",
                        failure_stage="capture_artifacts",
                        component_kind="scheduler",
                        component_id=f"{workflow_bundle.bundle_id}.capture_environment_artifacts",
                        owner=f"{session.benchmark_kit_id}_kit",
                        failure_code=(
                            f"artifact_capture.capture_artifacts."
                            f"{workflow_bundle.bundle_id}.capture_environment_artifacts_failed"
                        ),
                        first_bad_step=f"{workflow_bundle.bundle_id}.capture_environment_artifacts",
                        suspect_files=(
                            workflow_path,
                            "src/gage_eval/agent_runtime/schedulers/installed_client.py",
                        ),
                    )
                ) from exc
            if isinstance(captured, dict):
                artifact_paths.update({str(k): str(v) for k, v in captured.items() if v})
        finalized_output = raw_result
        if callable(workflow_bundle.finalize_result):
            try:
                finalized = workflow_bundle.finalize_result(
                    session=session,
                    sample=sample,
                    scheduler_output=raw_result,
                    artifact_paths=artifact_paths,
                )
            except Exception as exc:
                raise FailureEnvelopeError(
                    self._failure_mapper.map_exception(
                        exc,
                        failure_domain="artifact_capture",
                        failure_stage="normalize_result",
                        component_kind="scheduler",
                        component_id=f"{workflow_bundle.bundle_id}.finalize_result",
                        owner=f"{session.benchmark_kit_id}_kit",
                        failure_code=(
                            f"artifact_capture.normalize_result."
                            f"{workflow_bundle.bundle_id}.finalize_result_failed"
                        ),
                        first_bad_step=f"{workflow_bundle.bundle_id}.finalize_result",
                        suspect_files=(
                            workflow_path,
                            "src/gage_eval/agent_runtime/schedulers/installed_client.py",
                        ),
                    )
                ) from exc
            if isinstance(finalized, dict):
                finalized_output = finalized
        runtime_state = dict(session.benchmark_state or {})
        status = str(finalized_output.get("status") or raw_result.get("status") or "completed")
        if status not in {"completed", "failed", "aborted"}:
            status = "completed"
        if callable(workflow_bundle.failure_normalizer):
            normalized = workflow_bundle.failure_normalizer(
                session=session,
                sample=sample,
                scheduler_output=finalized_output,
                artifact_paths=artifact_paths,
                sandbox_provider=sandbox_provider,
            )
            if isinstance(normalized, dict):
                status = str(normalized.get("status") or status)
                normalized_artifacts = normalized.get("artifact_paths")
                if isinstance(normalized_artifacts, dict):
                    artifact_paths.update(
                        {str(key): str(value) for key, value in normalized_artifacts.items() if value}
                    )
                normalized_runtime_state = normalized.get("runtime_state")
                if isinstance(normalized_runtime_state, dict):
                    runtime_state.update(normalized_runtime_state)
                normalized_output = normalized.get("agent_output")
                if isinstance(normalized_output, dict):
                    finalized_output = normalized_output
        return SchedulerResult(
            scheduler_type="installed_client",
            benchmark_kit_id=session.benchmark_kit_id,
            status=status,  # type: ignore[arg-type]
            agent_output=finalized_output,
            artifact_paths=artifact_paths,
            runtime_state=runtime_state,
        )


def _resolve_workflow_path(session: AgentRuntimeSession) -> str:
    """Resolve the benchmark-owned workflow implementation path."""

    return (
        f"src/gage_eval/agent_eval_kits/{session.benchmark_kit_id}"
        f"/sub_workflows/{session.scheduler_type}.py"
    )
