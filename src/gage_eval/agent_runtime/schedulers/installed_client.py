from __future__ import annotations

from typing import Any, Mapping

from gage_eval.agent_runtime.clients.contracts import (
    ClientEnvironmentProjectionError,
    build_external_client_environment_handle,
    projection_failure_envelope,
    validate_external_client_environment_handle_payload,
)
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
        request_payload = {"metadata": _build_client_request_metadata(session=session)}
        workflow_path = _resolve_workflow_path(session)
        try:
            environment_payload = _build_client_environment(
                session=session,
                payload=payload,
            )
        except ClientEnvironmentProjectionError as exc:
            raise FailureEnvelopeError(
                projection_failure_envelope(
                    exc,
                    component_id="installed_client.environment_projection",
                    first_bad_step="installed_client.project_environment_handle",
                    suspect_files=(
                        "src/gage_eval/agent_runtime/clients/contracts.py",
                        "src/gage_eval/agent_runtime/schedulers/installed_client.py",
                    ),
                )
            ) from exc
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
                        failure_code=_workflow_prepare_failure_code(
                            session=session,
                            workflow_bundle=workflow_bundle,
                            step="prepare_environment",
                            domain="environment",
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
                environment_payload.setdefault("scheduler_state", {}).update(environment_state)
        if callable(workflow_bundle.prepare_inputs):
            try:
                prepared_inputs = workflow_bundle.prepare_inputs(
                    session=session,
                    sample=sample,
                    payload=payload,
                    sandbox_provider=sandbox_provider,
                )
                request_payload = _merge_client_request_payload(
                    request_payload,
                    prepared_inputs,
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
                        failure_code=_workflow_prepare_failure_code(
                            session=session,
                            workflow_bundle=workflow_bundle,
                            step="prepare_inputs",
                            domain="input_projection",
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
            session.scheduler_state.setdefault("client_id", session.client_id)
            session.scheduler_state.setdefault(
                "client_surface",
                self._runner.client_surface_name,
            )
            environment_payload.setdefault("scheduler_state", {}).setdefault(
                "client_id",
                session.client_id,
            )
            environment_payload.setdefault("scheduler_state", {}).setdefault(
                "client_surface",
                self._runner.client_surface_name,
            )
            raw_result = await self._runner.arun(
                request=request_payload,
                environment=environment_payload,
                session=session,
            )
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


def _workflow_prepare_failure_code(
    *,
    session: AgentRuntimeSession,
    workflow_bundle: Any,
    step: str,
    domain: str,
) -> str:
    if session.benchmark_kit_id == "swebench":
        return "input_projection.workflow.prepare_failed"
    return f"{domain}.prepare_inputs.{workflow_bundle.bundle_id}.{step}_failed"


def _build_client_environment(
    *,
    session: AgentRuntimeSession,
    payload: dict[str, Any],
) -> dict[str, Any]:
    """Project the benchmark-owned runtime state into one client environment payload."""

    handle_payload = session.runtime_context.get("external_environment_handle")
    if isinstance(handle_payload, Mapping):
        handle = validate_external_client_environment_handle_payload(
            handle_payload,
            requested_fields=payload.get("external_client_requested_fields"),
        )
        handle_payload = handle.model_dump(mode="json")
        session.runtime_context["external_environment_handle"] = dict(handle_payload)
    else:
        handle = build_external_client_environment_handle(
            session=session,
            environment_lease=payload.get("environment_lease"),
            requested_fields=payload.get("external_client_requested_fields"),
            proxy_starter=payload.get("environment_proxy_starter"),
        )
        handle_payload = handle.model_dump(mode="json")
        session.runtime_context["external_environment_handle"] = dict(handle_payload)

    return {
        "session_id": session.session_id,
        "run_id": session.run_id,
        "task_id": session.task_id,
        "sample_id": session.sample_id,
        "benchmark_kit_id": session.benchmark_kit_id,
        "scheduler_type": session.scheduler_type,
        "client_id": session.client_id,
        "artifact_layout": dict(session.artifact_layout or {}),
        "environment_handle": dict(handle_payload),
        "external_environment_handle": dict(handle_payload),
        "scheduler_state": dict(session.scheduler_state or {}),
    }


def _build_client_request_metadata(
    *,
    session: AgentRuntimeSession,
) -> dict[str, Any]:
    """Build one benchmark-neutral installed-client metadata payload."""

    return {
        "session_id": session.session_id,
        "run_id": session.run_id,
        "task_id": session.task_id,
        "sample_id": session.sample_id,
        "benchmark_kit_id": session.benchmark_kit_id,
        "scheduler_type": session.scheduler_type,
        "client_id": session.client_id,
    }


def _merge_client_request_payload(
    base_payload: Mapping[str, Any],
    update_payload: Any,
) -> dict[str, Any]:
    """Merge workflow-owned request fields into the client request envelope."""

    merged = dict(base_payload or {})
    if not isinstance(update_payload, Mapping):
        return merged

    for key, value in update_payload.items():
        normalized_key = str(key)
        if (
            normalized_key == "metadata"
            and isinstance(merged.get("metadata"), Mapping)
            and isinstance(value, Mapping)
        ):
            merged["metadata"] = {
                str(existing_key): existing_value
                for existing_key, existing_value in merged["metadata"].items()
            }
            merged["metadata"].update(
                {str(child_key): child_value for child_key, child_value in value.items()}
            )
            continue
        merged[normalized_key] = value
    return merged
