from __future__ import annotations

import hashlib
from pathlib import Path
import time
from typing import Any

from gage_eval.agent_runtime.artifacts import RuntimeArtifactSink, RuntimeTraceEmitter
from gage_eval.agent_runtime.clients.contracts import (
    ClientEnvironmentProjectionError,
    build_external_client_environment_handle,
    projection_failure_envelope,
)
from gage_eval.agent_runtime.compiled_plan import CompiledRuntimePlan
from gage_eval.agent_runtime.contracts.failure import FailureEnvelope, FailureEnvelopeError
from gage_eval.agent_runtime.contracts.scheduler import SchedulerResult
from gage_eval.agent_runtime.failure_mapper import FailureMapper
from gage_eval.agent_runtime.serialization import to_json_compatible
from gage_eval.agent_runtime.session import AgentRuntimeSession
from gage_eval.agent_runtime.trace_schema import ArtifactRef, TrialResult
from gage_eval.agent_runtime.trials import TrialPolicy, aggregate_trial_results
from gage_eval.agent_runtime.verifier.adapters import build_failure_result
from gage_eval.agent_runtime.verifier.contracts import RuntimeJudgeOutcome, VerifierInput, VerifierResult
from gage_eval.environment.lease import EnvironmentLease
from gage_eval.observability.trace import ObservabilityTrace


class DefaultVerifierRunner:
    """Runs the runtime-owned verifier adapter bound in the compiled plan."""

    def preflight(
        self,
        *,
        plan: CompiledRuntimePlan,
        session: AgentRuntimeSession,
        sample: dict[str, Any],
        scheduler_result: SchedulerResult,
    ) -> RuntimeJudgeOutcome | None:
        resources = dict(plan.judge_binding.verifier_resource_refs or {})
        adapter = resources.pop("adapter", None)
        preflight = getattr(adapter, "preflight", None)
        if not callable(preflight):
            return None
        verifier_input = VerifierInput(
            benchmark_kit_id=session.benchmark_kit_id,
            scheduler_type=session.scheduler_type,
            sample_id=session.sample_id,
            sample=sample,
            scheduler_result=scheduler_result.to_dict(),
            runtime_context={
                **dict(session.runtime_context or {}),
                "runtime_handle": session.resource_lease.handle_ref if session.resource_lease is not None else {},
            },
            verifier_resources=resources,
        )
        verifier_result = preflight(verifier_input)
        if verifier_result is None:
            return None
        judge_output = _normalize_judge_output(
            verifier_result.payload,
            scheduler_result=scheduler_result,
            judge_source=getattr(adapter, "judge_source", "runtime_verifier"),
        )
        if session.benchmark_kit_id == "swebench":
            _update_swebench_diagnostics_post_verifier(
                session=session,
                verifier_output=judge_output,
            )
        return RuntimeJudgeOutcome(
            verifier_input=verifier_input,
            verifier_result=verifier_result,
            judge_output=judge_output,
            persisted_path=session.artifact_layout["verifier_result"],
            failure=scheduler_result.failure,
        )

    def run(
        self,
        *,
        plan: CompiledRuntimePlan,
        session: AgentRuntimeSession,
        sample: dict[str, Any],
        scheduler_result: SchedulerResult,
        sandbox_provider=None,
        environment_lease=None,
    ) -> RuntimeJudgeOutcome:
        del sandbox_provider
        resources = dict(plan.judge_binding.verifier_resource_refs or {})
        adapter = resources.pop("adapter", None)
        if adapter is None:
            raise RuntimeError("runtime verifier adapter is not bound")
        resolved_environment_lease = environment_lease or session.runtime_context.get("environment_lease")
        verifier_input = VerifierInput(
            benchmark_kit_id=session.benchmark_kit_id,
            scheduler_type=session.scheduler_type,
            sample_id=session.sample_id,
            sample=sample,
            scheduler_result=scheduler_result.to_dict(),
            runtime_context={
                **dict(session.runtime_context or {}),
                "runtime_handle": session.resource_lease.handle_ref if session.resource_lease is not None else {},
                "environment_lease": resolved_environment_lease,
            },
            verifier_resources=resources,
        )
        verifier_result = adapter.run(verifier_input)
        judge_output = _normalize_judge_output(
            verifier_result.payload,
            scheduler_result=scheduler_result,
            judge_source=getattr(adapter, "judge_source", "runtime_verifier"),
        )
        if session.benchmark_kit_id == "swebench":
            _update_swebench_diagnostics_post_verifier(
                session=session,
                verifier_output=judge_output,
            )
        return RuntimeJudgeOutcome(
            verifier_input=verifier_input,
            verifier_result=verifier_result,
            judge_output=judge_output,
            persisted_path=session.artifact_layout["verifier_result"],
            failure=scheduler_result.failure,
        )

    def build_failed_outcome(
        self,
        *,
        plan: CompiledRuntimePlan,
        session: AgentRuntimeSession,
        sample: dict[str, Any],
        failure: FailureEnvelope,
    ) -> RuntimeJudgeOutcome:
        resources = dict(plan.judge_binding.verifier_resource_refs or {})
        adapter = resources.pop("adapter", None)
        judge_source = getattr(adapter, "judge_source", "runtime_verifier")
        verifier_input = VerifierInput(
            benchmark_kit_id=session.benchmark_kit_id,
            scheduler_type=session.scheduler_type,
            sample_id=session.sample_id,
            sample=sample,
            scheduler_result={},
            runtime_context=dict(session.runtime_context or {}),
            verifier_resources=resources,
        )
        verifier_result = build_failure_result(
            judge_source=judge_source,
            failure=failure,
        )
        judge_output = _normalize_judge_output(
            verifier_result.payload,
            scheduler_result=None,
            judge_source=judge_source,
        )
        return RuntimeJudgeOutcome(
            verifier_input=verifier_input,
            verifier_result=verifier_result,
            judge_output=judge_output,
            persisted_path=session.artifact_layout["verifier_result"],
            failure=failure,
        )


class AgentRuntimeSessionFactory:
    """Builds sample-scoped runtime sessions from the compiled plan."""

    def __init__(self, artifact_sink: RuntimeArtifactSink) -> None:
        self._artifact_sink = artifact_sink

    def create(
        self,
        *,
        plan: CompiledRuntimePlan,
        sample: dict[str, Any],
        payload: dict[str, Any],
    ) -> AgentRuntimeSession:
        """Create one sample-scoped runtime session."""

        execution_context = payload.get("execution_context") or {}
        run_id = str(execution_context.get("run_id") or "runtime-run")
        task_id = str(execution_context.get("task_id") or "runtime-task")
        sample_id = str(
            execution_context.get("sample_id")
            or sample.get("id")
            or sample.get("sample_id")
            or "sample"
        )
        session_hash = hashlib.md5(
            f"{plan.plan_id}:{run_id}:{task_id}:{sample_id}".encode("utf-8")
        ).hexdigest()
        layout = self._artifact_sink.build_layout(
            run_id=run_id,
            task_id=task_id,
            sample_id=sample_id,
        )
        return AgentRuntimeSession(
            session_id=f"session-{session_hash[:12]}",
            run_id=run_id,
            task_id=task_id,
            sample_id=sample_id,
            benchmark_kit_id=plan.runtime_spec.benchmark_kit_id,
            scheduler_type=plan.runtime_spec.scheduler_type,
            client_id=plan.runtime_spec.client_id,
            artifact_layout=layout,
            artifact_sink=self._artifact_sink,
        )


class CompiledRuntimeExecutor:
    """Executes one compiled runtime plan end-to-end for a single sample."""

    def __init__(
        self,
        *,
        compiled_plan: CompiledRuntimePlan,
        resource_manager,
        session_factory: AgentRuntimeSessionFactory,
        verifier_runner: DefaultVerifierRunner,
        artifact_sink: RuntimeArtifactSink,
        trace_emitter: RuntimeTraceEmitter,
        failure_mapper: FailureMapper,
        mcp_processes: list[Any] | None = None,
    ) -> None:
        self.compiled_plan = compiled_plan
        self.resource_manager = resource_manager
        self.session_factory = session_factory
        self.verifier_runner = verifier_runner
        self.artifact_sink = artifact_sink
        self.trace_emitter = trace_emitter
        self.failure_mapper = failure_mapper
        self.mcp_processes = list(mcp_processes or [])

    async def aexecute(
        self,
        *,
        sample: dict[str, Any],
        payload: dict[str, Any],
        trace: ObservabilityTrace | None = None,
    ) -> dict[str, Any]:
        """Execute one runtime-owned sample flow and return model output."""

        trial_policy = TrialPolicy.from_mapping(self.compiled_plan.trial_policy)
        if trial_policy.trials > 1:
            return await self._aexecute_multi_trial(
                sample=sample,
                payload=payload,
                trace=trace,
                trial_policy=trial_policy,
            )

        # STEP 1: Materialize the runtime session and sample-scoped trace context.
        session = self.session_factory.create(
            plan=self.compiled_plan,
            sample=sample,
            payload=payload,
        )
        trial_id = "trial_0001"
        session.runtime_context.setdefault("trial_id", trial_id)
        session.scheduler_state.setdefault("trial_id", trial_id)
        trial_started_at = time.perf_counter()
        self.trace_emitter.emit_session_start(trace, session)
        self.artifact_sink.append_trace_event(
            run_id=session.run_id,
            task_id=session.task_id,
            sample_id=session.sample_id,
            trial_id=trial_id,
            actor="runtime",
            event_type="trial.start",
            payload={
                "trial_id": trial_id,
                "trial_index": 1,
                "trial_policy": trial_policy.to_dict(),
                "scheduler_type": session.scheduler_type,
            },
        )

        # STEP 2: Acquire the resource lease and bootstrap the benchmark runtime.
        lease_binding = None
        verifier_lease_binding = None
        scheduler_result: SchedulerResult | None = None
        artifact_refs: list[ArtifactRef] = []
        environment_descriptor: dict[str, Any] = {}
        final_output: dict[str, Any] | None = None
        final_failure: FailureEnvelope | None = None
        try:
            try:
                bound_environment_lease = _payload_environment_lease(payload)
                if bound_environment_lease is not None:
                    lease_binding = self.resource_manager.bind_existing(
                        session,
                        resource_plan=self.compiled_plan.resource_plan,
                        environment_lease=bound_environment_lease,
                    )
                else:
                    lease_binding = self.resource_manager.acquire(
                        session,
                        resource_plan=self.compiled_plan.resource_plan,
                        trace=trace,
                        sample=sample,
                    )
                session.resource_lease = lease_binding.resource_lease
                environment_lease = _environment_lease_from_binding(
                    lease_binding,
                    artifact_sink=self.artifact_sink,
                )
                if environment_lease is not None:
                    session.runtime_context.setdefault("environment_lease", environment_lease)
                environment_descriptor = _environment_descriptor_from_binding(lease_binding)
                self.artifact_sink.append_trace_event(
                    run_id=session.run_id,
                    task_id=session.task_id,
                    sample_id=session.sample_id,
                    trial_id=trial_id,
                    actor="environment",
                    event_type="environment.acquire",
                    payload={
                        "environment_descriptor": environment_descriptor,
                        "role": "scheduler",
                    },
                )
            except Exception as exc:
                failure = self.failure_mapper.map_exception(
                    exc,
                    failure_domain="environment",
                    failure_stage="acquire_lease",
                    component_kind="resource_manager",
                    component_id=f"{session.benchmark_kit_id}.resource.acquire",
                    owner="runtime_resource_core",
                    failure_code="environment.acquire_lease.resource_manager.acquire_failed",
                    first_bad_step="environment.acquire_lease",
                    suspect_files=("src/gage_eval/agent_runtime/resources/manager.py",),
                )
                raise FailureEnvelopeError(failure) from exc

            try:
                runtime_payload = _payload_with_environment_lease(
                    payload,
                    session=session,
                    lease_binding=lease_binding,
                    artifact_sink=self.artifact_sink,
                )
                runtime_context = self.compiled_plan.kit_runtime_ref.bootstrap(
                    session=session,
                    sample=sample,
                    payload=runtime_payload,
                    sandbox_provider=None,
                )
                if isinstance(runtime_context, dict):
                    session.runtime_context.update(runtime_context.get("runtime_context") or {})
                    session.prompt_context.update(runtime_context.get("prompt_context") or {})
                    session.benchmark_state.update(runtime_context.get("benchmark_state") or {})
                    session.scheduler_state.update(runtime_context.get("scheduler_state") or {})
            except Exception as exc:
                failure = self.failure_mapper.map_exception(
                    exc,
                    failure_domain="environment",
                    failure_stage="bootstrap_runtime",
                    component_kind="runtime",
                    component_id=f"{session.benchmark_kit_id}.runtime.bootstrap",
                    owner=f"{session.benchmark_kit_id}_kit",
                    failure_code=f"environment.bootstrap_runtime.{session.benchmark_kit_id}.runtime.bootstrap_failed",
                    first_bad_step=f"{session.benchmark_kit_id}.runtime.bootstrap",
                    suspect_files=(f"src/gage_eval/agent_eval_kits/{session.benchmark_kit_id}/runtime.py",),
                )
                raise FailureEnvelopeError(failure) from exc

            # STEP 3: Run the bound scheduler and normalize benchmark artifacts.
            try:
                scheduler_payload = _scheduler_payload(
                    payload=payload,
                    session=session,
                    resource_plan=self.compiled_plan.resource_plan,
                    lease_binding=lease_binding,
                    artifact_sink=self.artifact_sink,
                )
                if scheduler_payload.get("environment_lease") is not None:
                    session.runtime_context.setdefault(
                        "environment_lease",
                        scheduler_payload["environment_lease"],
                    )
                _reset_mcp_processes(self.mcp_processes, trial_id=trial_id)
                _append_environment_handle_projected(
                    artifact_sink=self.artifact_sink,
                    session=session,
                    trial_id=trial_id,
                    environment_lease=scheduler_payload.get("environment_lease"),
                    lease_binding=lease_binding,
                    requested_fields=scheduler_payload.get("external_client_requested_fields"),
                    proxy_starter=scheduler_payload.get("environment_proxy_starter"),
                )
                scheduler_result = await self.compiled_plan.scheduler_handle.arun(
                    session=session,
                    sample=sample,
                    payload=scheduler_payload,
                    workflow_bundle=self.compiled_plan.workflow_bundle,
                    sandbox_provider=None,
                )
                scheduler_failure = _scheduler_failure_from_result(
                    session=session,
                    scheduler_result=scheduler_result,
                )
                if scheduler_failure is not None:
                    scheduler_result.failure = scheduler_failure
                    final_failure = scheduler_failure
                artifact_refs.append(
                    self.artifact_sink.write_artifact(
                        run_id=session.run_id,
                        task_id=session.task_id,
                        sample_id=session.sample_id,
                        trial_id=trial_id,
                        owner="agent",
                        name="scheduler_result.json",
                        content=scheduler_result.to_dict(),
                        mime_type="application/json",
                    )
                )
            except FailureEnvelopeError:
                raise
            except Exception as exc:
                failure = self.failure_mapper.map_exception(
                    exc,
                    failure_domain="client_execution",
                    failure_stage="run_scheduler",
                    component_kind="scheduler",
                    component_id=f"{session.scheduler_type}.scheduler.run",
                    owner="runtime_scheduler_core",
                    failure_code=f"client_execution.run_scheduler.{session.scheduler_type}.scheduler_failed",
                    first_bad_step="scheduler.run",
                    suspect_files=(f"src/gage_eval/agent_runtime/schedulers/{session.scheduler_type}.py",),
                )
                raise FailureEnvelopeError(failure) from exc

            # STEP 4: Run the runtime-owned verifier and persist shared evidence.
            if self.compiled_plan.judge_binding.judge_mode == "runtime_verifier":
                try:
                    await _carry_scheduler_submission_patch_for_verifier(
                        session=session,
                        scheduler_result=scheduler_result,
                    )
                    outcome = _build_scheduler_failure_skipped_outcome(
                        plan=self.compiled_plan,
                        session=session,
                        sample=sample,
                        scheduler_result=scheduler_result,
                    )
                    preflight = getattr(self.verifier_runner, "preflight", None)
                    if outcome is None:
                        outcome = (
                            preflight(
                                plan=self.compiled_plan,
                                session=session,
                                sample=sample,
                                scheduler_result=scheduler_result,
                            )
                            if callable(preflight)
                            else None
                        )
                    if outcome is None:
                        verifier_environment_lease = session.runtime_context.get("environment_lease")
                        if self.compiled_plan.verifier_environment_policy == "fresh_from_profile":
                            verifier_lease_binding = self.resource_manager.acquire(
                                session,
                                resource_plan=_verifier_resource_plan(self.compiled_plan),
                                trace=trace,
                                sample=_verifier_acquire_sample(self.compiled_plan, sample),
                            )
                            self.artifact_sink.append_trace_event(
                                run_id=session.run_id,
                                task_id=session.task_id,
                                sample_id=session.sample_id,
                                trial_id=trial_id,
                                actor="environment",
                                event_type="environment.acquire",
                                payload={
                                    "environment_descriptor": _environment_descriptor_from_binding(
                                        verifier_lease_binding
                                    ),
                                    "role": "verifier",
                                },
                            )
                        _prepare_verifier_environment_context(
                            session=session,
                            lease_binding=verifier_lease_binding or lease_binding,
                            artifact_sink=self.artifact_sink,
                        )
                        verifier_environment_lease = session.runtime_context.get("environment_lease")
                        outcome = self.verifier_runner.run(
                            plan=self.compiled_plan,
                            session=session,
                            sample=sample,
                            scheduler_result=scheduler_result,
                            sandbox_provider=None,
                            environment_lease=verifier_environment_lease,
                        )
                except Exception as exc:
                    failure = self.failure_mapper.map_exception(
                        exc,
                        failure_domain="verifier",
                        failure_stage="run_verifier",
                        component_kind="verifier_adapter",
                        component_id=f"{session.benchmark_kit_id}.verifier.run",
                        owner=f"{session.benchmark_kit_id}_kit",
                        failure_code=f"verifier.run_verifier.{session.benchmark_kit_id}.verifier_failed",
                        first_bad_step=f"{session.benchmark_kit_id}.verifier.run",
                        suspect_files=(
                            f"src/gage_eval/agent_eval_kits/{session.benchmark_kit_id}/judge_bridge.py",
                            "src/gage_eval/agent_runtime/verifier/adapters.py",
                            "src/gage_eval/agent_runtime/executor.py",
                        ),
                    )
                    raise FailureEnvelopeError(failure) from exc
                session.judge_outcome = outcome
                verifier_ref = self.artifact_sink.write_artifact(
                    run_id=session.run_id,
                    task_id=session.task_id,
                    sample_id=session.sample_id,
                    trial_id=trial_id,
                    owner="verifier",
                    name="verifier_result.json",
                    content=outcome.to_dict(),
                    mime_type="application/json",
                )
                artifact_refs.append(verifier_ref)
                self.artifact_sink.append_trace_event(
                    run_id=session.run_id,
                    task_id=session.task_id,
                    sample_id=session.sample_id,
                    trial_id=trial_id,
                    actor="verifier",
                    event_type="verifier.result",
                    payload={
                        "metric": _verifier_metric_payload(outcome.judge_output),
                        "verifier_result": outcome.judge_output,
                    },
                    artifact_refs=[verifier_ref],
                )
            final_output = (
                _build_failed_runtime_model_output(session, final_failure)
                if final_failure is not None
                else _build_runtime_model_output(session, scheduler_result)
            )
        except FailureEnvelopeError as exc:
            failure = exc.failure
            final_failure = failure
            raw_error_path = _safe_persist_raw_error(
                artifact_sink=self.artifact_sink,
                session=session,
                error=exc,
            )
            if raw_error_path is not None:
                failure.raw_error_path = raw_error_path
            outcome = self.verifier_runner.build_failed_outcome(
                plan=self.compiled_plan,
                session=session,
                sample=sample,
                failure=failure,
            )
            session.judge_outcome = outcome
            _safe_persist_verifier_result(self.artifact_sink, outcome)
            self.trace_emitter.emit_failure(trace, session, failure=failure)
            final_output = _build_failed_runtime_model_output(session, failure)

        cleanup_error: BaseException | None = None
        if _should_release_binding(verifier_lease_binding):
            try:
                self.resource_manager.release(verifier_lease_binding)
            except Exception as exc:
                cleanup_error = exc
        if _should_release_binding(lease_binding):
            try:
                self.resource_manager.release(lease_binding)
            except Exception as exc:
                cleanup_error = exc

        if cleanup_error is not None and final_failure is None:
            failure = self.failure_mapper.map_exception(
                cleanup_error,
                failure_domain="persistence",
                failure_stage="cleanup",
                component_kind="resource_manager",
                component_id=f"{session.benchmark_kit_id}.resource.cleanup",
                owner="runtime_resource_core",
                failure_code="persistence.cleanup.resource_manager.release_failed",
                first_bad_step="environment.cleanup.release",
                suspect_files=("src/gage_eval/agent_runtime/resources/manager.py",),
            )
            raw_error_path = _safe_persist_raw_error(
                artifact_sink=self.artifact_sink,
                session=session,
                error=cleanup_error,
            )
            if raw_error_path is not None:
                failure.raw_error_path = raw_error_path
            outcome = self.verifier_runner.build_failed_outcome(
                plan=self.compiled_plan,
                session=session,
                sample=sample,
                failure=failure,
            )
            session.judge_outcome = outcome
            _safe_persist_verifier_result(self.artifact_sink, outcome)
            self.trace_emitter.emit_failure(trace, session, failure=failure)
            final_failure = failure
            final_output = _build_failed_runtime_model_output(session, failure)

        trace_ref = self.artifact_sink.append_trace_event(
            run_id=session.run_id,
            task_id=session.task_id,
            sample_id=session.sample_id,
            trial_id=trial_id,
            actor="runtime",
            event_type="trial.end",
            payload={
                "trial_id": trial_id,
                "status": "failed" if final_failure is not None else "completed",
                "failure": final_failure.to_dict() if final_failure is not None else None,
                "duration_ms": (time.perf_counter() - trial_started_at) * 1000.0,
            },
        )
        trial_result = _build_trial_result_from_session(
            trial_id=trial_id,
            scheduler_result=scheduler_result,
            session=session,
            artifact_refs=artifact_refs,
            trace_ref=trace_ref,
            environment_descriptor=environment_descriptor,
            failure=final_failure,
        )
        record_ref = self.artifact_sink.write_trial_record(
            run_id=session.run_id,
            task_id=session.task_id,
            sample_id=session.sample_id,
            trial_id=trial_id,
            trial_result=trial_result,
        )
        trial_result.artifact_refs.append(record_ref)
        aggregate = aggregate_trial_results([trial_result], aggregation=trial_policy.aggregation)
        _write_sample_infra_artifacts(
            artifact_sink=self.artifact_sink,
            plan=self.compiled_plan,
            session=session,
            sample=sample,
            trial_policy=trial_policy,
            trial_results=[trial_result],
            aggregate=aggregate,
            scheduler_result=scheduler_result,
            failure=final_failure,
        )
        _emit_sample_failed_summary(
            trace=trace,
            session=session,
            trial_id=trial_id,
            failure=final_failure,
            judge_outcome=session.judge_outcome,
        )
        if final_failure is None:
            self.trace_emitter.emit_session_end(
                trace,
                session,
                scheduler_result=scheduler_result,
            )
        if final_output is None:
            final_output = _build_runtime_model_output(session, scheduler_result)
        _attach_agent_eval_output(
            final_output,
            aggregate=aggregate,
            trial_results=[trial_result],
        )
        return to_json_compatible(final_output)

    async def _aexecute_multi_trial(
        self,
        *,
        sample: dict[str, Any],
        payload: dict[str, Any],
        trace: ObservabilityTrace | None,
        trial_policy: TrialPolicy,
    ) -> dict[str, Any]:
        """Execute multiple trial-scoped runs while keeping the sample output scalar-compatible."""

        trial_results: list[TrialResult] = []
        primary_session: AgentRuntimeSession | None = None
        primary_scheduler_result: SchedulerResult | None = None

        for trial_index, trial_id in enumerate(trial_policy.trial_ids(), start=1):
            trial_started_at = time.perf_counter()
            session = self.session_factory.create(
                plan=self.compiled_plan,
                sample=sample,
                payload=payload,
            )
            session.runtime_context["trial_id"] = trial_id
            session.scheduler_state["trial_id"] = trial_id
            _scope_session_to_trial(session, artifact_sink=self.artifact_sink, trial_id=trial_id)
            self.trace_emitter.emit_session_start(trace, session)

            trace_ref = self.artifact_sink.append_trace_event(
                run_id=session.run_id,
                task_id=session.task_id,
                sample_id=session.sample_id,
                trial_id=trial_id,
                actor="runtime",
                event_type="trial.start",
                payload={
                    "trial_id": trial_id,
                    "trial_index": trial_index,
                    "trial_policy": trial_policy.to_dict(),
                    "scheduler_type": session.scheduler_type,
                },
            )
            artifact_refs: list[ArtifactRef] = []
            lease_binding = None
            verifier_lease_binding = None
            scheduler_result: SchedulerResult | None = None
            failure: FailureEnvelope | None = None
            environment_descriptor: dict[str, Any] = {}

            try:
                bound_environment_lease = _payload_environment_lease(payload)
                if bound_environment_lease is not None:
                    lease_binding = self.resource_manager.bind_existing(
                        session,
                        resource_plan=self.compiled_plan.resource_plan,
                        environment_lease=bound_environment_lease,
                    )
                else:
                    lease_binding = self.resource_manager.acquire(
                        session,
                        resource_plan=self.compiled_plan.resource_plan,
                        trace=trace,
                        sample=sample,
                    )
                session.resource_lease = lease_binding.resource_lease
                environment_lease = _environment_lease_from_binding(
                    lease_binding,
                    artifact_sink=self.artifact_sink,
                )
                if environment_lease is not None:
                    session.runtime_context.setdefault("environment_lease", environment_lease)
                environment_descriptor = _environment_descriptor_from_binding(lease_binding)
                trace_ref = self.artifact_sink.append_trace_event(
                    run_id=session.run_id,
                    task_id=session.task_id,
                    sample_id=session.sample_id,
                    trial_id=trial_id,
                    actor="environment",
                    event_type="environment.acquire",
                    payload={"environment_descriptor": environment_descriptor, "role": "scheduler"},
                )

                runtime_payload = _payload_with_environment_lease(
                    {**payload, "trial_id": trial_id},
                    session=session,
                    lease_binding=lease_binding,
                    artifact_sink=self.artifact_sink,
                )
                runtime_context = self.compiled_plan.kit_runtime_ref.bootstrap(
                    session=session,
                    sample=sample,
                    payload=runtime_payload,
                    sandbox_provider=None,
                )
                if isinstance(runtime_context, dict):
                    session.runtime_context.update(runtime_context.get("runtime_context") or {})
                    session.prompt_context.update(runtime_context.get("prompt_context") or {})
                    session.benchmark_state.update(runtime_context.get("benchmark_state") or {})
                    session.scheduler_state.update(runtime_context.get("scheduler_state") or {})
                session.runtime_context["trial_id"] = trial_id
                session.scheduler_state["trial_id"] = trial_id

                _reset_mcp_processes(self.mcp_processes, trial_id=trial_id)
                scheduler_payload = _scheduler_payload(
                    payload={**payload, "trial_id": trial_id},
                    session=session,
                    resource_plan=self.compiled_plan.resource_plan,
                    lease_binding=lease_binding,
                    artifact_sink=self.artifact_sink,
                )
                if scheduler_payload.get("environment_lease") is not None:
                    session.runtime_context.setdefault(
                        "environment_lease",
                        scheduler_payload["environment_lease"],
                    )
                scheduler_result = await self.compiled_plan.scheduler_handle.arun(
                    session=session,
                    sample=sample,
                    payload=scheduler_payload,
                    workflow_bundle=self.compiled_plan.workflow_bundle,
                    sandbox_provider=None,
                )
                failure = _scheduler_failure_from_result(
                    session=session,
                    scheduler_result=scheduler_result,
                )
                if failure is not None:
                    scheduler_result.failure = failure
                artifact_refs.append(
                    self.artifact_sink.write_artifact(
                        run_id=session.run_id,
                        task_id=session.task_id,
                        sample_id=session.sample_id,
                        trial_id=trial_id,
                        owner="agent",
                        name="scheduler_result.json",
                        content=scheduler_result.to_dict(),
                        mime_type="application/json",
                    )
                )

                await _carry_scheduler_submission_patch_for_verifier(
                    session=session,
                    scheduler_result=scheduler_result,
                )
                if self.compiled_plan.judge_binding.judge_mode == "runtime_verifier":
                    session.judge_outcome = _build_scheduler_failure_skipped_outcome(
                        plan=self.compiled_plan,
                        session=session,
                        sample=sample,
                        scheduler_result=scheduler_result,
                    )
                    preflight = getattr(self.verifier_runner, "preflight", None)
                    if session.judge_outcome is None:
                        session.judge_outcome = (
                            preflight(
                                plan=self.compiled_plan,
                                session=session,
                                sample=sample,
                                scheduler_result=scheduler_result,
                            )
                            if callable(preflight)
                            else None
                        )
                    if session.judge_outcome is None:
                        verifier_environment_lease = session.runtime_context.get("environment_lease")
                        if self.compiled_plan.verifier_environment_policy == "fresh_from_profile":
                            verifier_lease_binding = self.resource_manager.acquire(
                                session,
                                resource_plan=_verifier_resource_plan(self.compiled_plan),
                                trace=trace,
                                sample=_verifier_acquire_sample(self.compiled_plan, sample),
                            )
                            trace_ref = self.artifact_sink.append_trace_event(
                                run_id=session.run_id,
                                task_id=session.task_id,
                                sample_id=session.sample_id,
                                trial_id=trial_id,
                                actor="environment",
                                event_type="environment.acquire",
                                payload={
                                    "environment_descriptor": _environment_descriptor_from_binding(
                                        verifier_lease_binding
                                    ),
                                    "role": "verifier",
                                },
                            )
                        _prepare_verifier_environment_context(
                            session=session,
                            lease_binding=verifier_lease_binding or lease_binding,
                            artifact_sink=self.artifact_sink,
                        )
                        verifier_environment_lease = session.runtime_context.get("environment_lease")
                        session.judge_outcome = self.verifier_runner.run(
                            plan=self.compiled_plan,
                            session=session,
                            sample=sample,
                            scheduler_result=scheduler_result,
                            sandbox_provider=None,
                            environment_lease=verifier_environment_lease,
                        )
                    verifier_ref = self.artifact_sink.write_artifact(
                        run_id=session.run_id,
                        task_id=session.task_id,
                        sample_id=session.sample_id,
                        trial_id=trial_id,
                        owner="verifier",
                        name="verifier_result.json",
                        content=session.judge_outcome.to_dict(),
                        mime_type="application/json",
                    )
                    artifact_refs.append(verifier_ref)
                    trace_ref = self.artifact_sink.append_trace_event(
                        run_id=session.run_id,
                        task_id=session.task_id,
                        sample_id=session.sample_id,
                        trial_id=trial_id,
                        actor="verifier",
                        event_type="verifier.result",
                        payload={
                            "metric": _verifier_metric_payload(session.judge_outcome.judge_output),
                            "verifier_result": session.judge_outcome.judge_output,
                        },
                        artifact_refs=[verifier_ref],
                    )
                self.trace_emitter.emit_session_end(trace, session, scheduler_result=scheduler_result)
            except FailureEnvelopeError as exc:
                failure = exc.failure
                session.judge_outcome = self.verifier_runner.build_failed_outcome(
                    plan=self.compiled_plan,
                    session=session,
                    sample=sample,
                    failure=failure,
                )
                self.trace_emitter.emit_failure(trace, session, failure=failure)
            except Exception as exc:
                failure = self.failure_mapper.map_exception(
                    exc,
                    failure_domain="trial",
                    failure_stage="execute_trial",
                    component_kind="trial_manager",
                    component_id=f"{session.benchmark_kit_id}.trial.{trial_id}",
                    owner="runtime_trial_core",
                    failure_code="trial.execution_failed",
                    first_bad_step="trial.execute",
                    suspect_files=(
                        "src/gage_eval/agent_runtime/executor.py",
                        "src/gage_eval/agent_runtime/trials.py",
                    ),
                )
                session.judge_outcome = self.verifier_runner.build_failed_outcome(
                    plan=self.compiled_plan,
                    session=session,
                    sample=sample,
                    failure=failure,
                )
                self.trace_emitter.emit_failure(trace, session, failure=failure)
            finally:
                if _should_release_binding(verifier_lease_binding):
                    try:
                        self.resource_manager.release(verifier_lease_binding)
                    except Exception as exc:
                        cleanup_failure = self.failure_mapper.map_exception(
                            exc,
                            failure_domain="persistence",
                            failure_stage="cleanup",
                            component_kind="resource_manager",
                            component_id=f"{session.benchmark_kit_id}.verifier_resource.cleanup",
                            owner="runtime_resource_core",
                            failure_code="persistence.cleanup.resource_manager.release_failed",
                            first_bad_step="environment.cleanup.release_verifier",
                            suspect_files=("src/gage_eval/agent_runtime/resources/manager.py",),
                        )
                        failure = _merge_cleanup_failure(failure, cleanup_failure)
                if _should_release_binding(lease_binding):
                    try:
                        self.resource_manager.release(lease_binding)
                    except Exception as exc:
                        cleanup_failure = self.failure_mapper.map_exception(
                            exc,
                            failure_domain="persistence",
                            failure_stage="cleanup",
                            component_kind="resource_manager",
                            component_id=f"{session.benchmark_kit_id}.resource.cleanup",
                            owner="runtime_resource_core",
                            failure_code="persistence.cleanup.resource_manager.release_failed",
                            first_bad_step="environment.cleanup.release",
                            suspect_files=("src/gage_eval/agent_runtime/resources/manager.py",),
                        )
                        failure = _merge_cleanup_failure(failure, cleanup_failure)

            trace_ref = self.artifact_sink.append_trace_event(
                run_id=session.run_id,
                task_id=session.task_id,
                sample_id=session.sample_id,
                trial_id=trial_id,
                actor="runtime",
                event_type="trial.end",
                payload={
                    "trial_id": trial_id,
                    "status": "failed" if failure is not None else "completed",
                    "failure": failure.to_dict() if failure is not None else None,
                    "duration_ms": (time.perf_counter() - trial_started_at) * 1000.0,
                },
            )
            trial_result = _build_trial_result_from_session(
                trial_id=trial_id,
                scheduler_result=scheduler_result,
                session=session,
                artifact_refs=artifact_refs,
                trace_ref=trace_ref,
                environment_descriptor=environment_descriptor,
                failure=failure,
            )
            record_ref = self.artifact_sink.write_trial_record(
                run_id=session.run_id,
                task_id=session.task_id,
                sample_id=session.sample_id,
                trial_id=trial_id,
                trial_result=trial_result,
            )
            trial_result.artifact_refs.append(record_ref)
            trial_results.append(trial_result)
            _emit_sample_failed_summary(
                trace=trace,
                session=session,
                trial_id=trial_id,
                failure=failure,
                judge_outcome=session.judge_outcome,
            )

            if trial_index == 1:
                primary_session = session
                primary_scheduler_result = scheduler_result

        aggregate = aggregate_trial_results(trial_results, aggregation=trial_policy.aggregation)
        if primary_session is None:
            raise RuntimeError("multi-trial execution produced no primary trial")
        self.artifact_sink.write_trial_aggregate(
            run_id=primary_session.run_id,
            task_id=primary_session.task_id,
            sample_id=primary_session.sample_id,
            aggregate=aggregate,
        )
        if primary_session.judge_outcome is not None:
            _safe_persist_verifier_result(self.artifact_sink, primary_session.judge_outcome)
        _write_sample_infra_artifacts(
            artifact_sink=self.artifact_sink,
            plan=self.compiled_plan,
            session=primary_session,
            sample=sample,
            trial_policy=trial_policy,
            trial_results=trial_results,
            aggregate=aggregate,
            scheduler_result=primary_scheduler_result,
            failure=primary_scheduler_result.failure if primary_scheduler_result is not None else None,
        )
        output = _build_runtime_model_output(primary_session, primary_scheduler_result)
        _attach_agent_eval_output(
            output,
            aggregate=aggregate,
            trial_results=trial_results,
        )
        return to_json_compatible(output)


def _build_scheduler_failure_skipped_outcome(
    *,
    plan: CompiledRuntimePlan,
    session: AgentRuntimeSession,
    sample: dict[str, Any],
    scheduler_result: SchedulerResult,
) -> RuntimeJudgeOutcome | None:
    scheduler_failure = scheduler_result.failure
    if scheduler_failure is None:
        return None
    resources = dict(plan.judge_binding.verifier_resource_refs or {}) if plan.judge_binding is not None else {}
    resources.pop("adapter", None)
    verifier_input = VerifierInput(
        benchmark_kit_id=session.benchmark_kit_id,
        scheduler_type=session.scheduler_type,
        sample_id=session.sample_id,
        sample=sample,
        scheduler_result=scheduler_result.to_dict(),
        runtime_context=dict(session.runtime_context or {}),
        verifier_resources=resources,
    )
    payload = {
        "status": "skipped",
        "resolved": False,
        "score": 0.0,
        "failure_code": "verifier.skipped_due_to_scheduler_failure",
        "failure_reason": scheduler_failure.failure_code,
        "summary": "Verifier skipped because scheduler reported failure",
        "scheduler_failure": scheduler_failure.to_dict(),
    }
    verifier_result = VerifierResult(status="skipped", payload=payload)
    judge_output = _normalize_judge_output(
        verifier_result.payload,
        scheduler_result=scheduler_result,
        judge_source="runtime_verifier",
    )
    return RuntimeJudgeOutcome(
        verifier_input=verifier_input,
        verifier_result=verifier_result,
        judge_output=judge_output,
        persisted_path=session.artifact_layout["verifier_result"],
        failure=scheduler_failure,
    )


def _environment_descriptor_from_binding(lease_binding: Any) -> dict[str, Any]:
    lease = getattr(lease_binding, "resource_lease", None)
    if lease is not None and hasattr(lease, "to_dict"):
        return lease.to_dict()
    if isinstance(lease, dict):
        return dict(lease)
    return {}


def _should_release_binding(lease_binding: Any | None) -> bool:
    if lease_binding is None:
        return False
    return bool(getattr(lease_binding, "owns_environment_lease", True))


def _scheduler_payload(
    *,
    payload: dict[str, Any],
    session: AgentRuntimeSession,
    resource_plan: dict[str, Any],
    lease_binding: Any,
    artifact_sink: RuntimeArtifactSink,
    ) -> dict[str, Any]:
    scheduler_payload = {
        **payload,
        "environment_profile": resource_plan.get("environment_profile"),
        "provider_config": resource_plan.get("provider_config"),
        "artifact_sink": artifact_sink,
    }
    if scheduler_payload.get("environment_lease") is None:
        environment_lease = session.runtime_context.get("environment_lease")
        if environment_lease is None:
            environment_lease = _environment_lease_from_binding(
                lease_binding,
                artifact_sink=artifact_sink,
            )
        if environment_lease is not None:
            scheduler_payload["environment_lease"] = environment_lease
    return scheduler_payload


def _reset_mcp_processes(processes: list[Any], *, trial_id: str) -> None:
    for process in processes:
        reset_for_trial = getattr(process, "reset_for_trial", None)
        if callable(reset_for_trial):
            reset_for_trial(trial_id)
            continue
        bind_trial = getattr(process, "bind_trial", None)
        if callable(bind_trial):
            bind_trial(trial_id)


def _append_environment_handle_projected(
    *,
    artifact_sink: RuntimeArtifactSink,
    session: AgentRuntimeSession,
    trial_id: str,
    environment_lease: Any | None,
    lease_binding: Any,
    requested_fields: Any | None = None,
    proxy_starter: Any | None = None,
) -> None:
    if session.scheduler_type == "acp_client":
        return
    try:
        external_handle = build_external_client_environment_handle(
            session=session,
            environment_lease=environment_lease,
            lease_binding=lease_binding,
            requested_fields=requested_fields,
            proxy_starter=proxy_starter,
        )
        handle_payload = external_handle.model_dump(mode="json")
        trace_handle_payload = _redact_external_handle_for_trace(handle_payload)
        capabilities_payload: dict[str, Any] = {"capabilities": list(external_handle.capabilities)}
        session.runtime_context["external_environment_handle"] = dict(handle_payload)
    except ClientEnvironmentProjectionError as exc:
        if session.scheduler_type in {"installed_client", "acp_client"}:
            raise FailureEnvelopeError(
                projection_failure_envelope(
                    exc,
                    component_id=f"{session.scheduler_type}.environment_projection",
                    first_bad_step=f"{session.scheduler_type}.project_environment_handle",
                    suspect_files=(
                        "src/gage_eval/agent_runtime/clients/contracts.py",
                        "src/gage_eval/agent_runtime/executor.py",
                    ),
                )
            ) from exc
        handle_payload = _environment_handle_payload(environment_lease, lease_binding)
        trace_handle_payload = handle_payload
        capabilities_payload = _environment_capabilities_payload(environment_lease)
    artifact_sink.append_trace_event(
        run_id=session.run_id,
        task_id=session.task_id,
        sample_id=session.sample_id,
        trial_id=trial_id,
        actor="scheduler",
        event_type="client.environment_handle.projected",
        payload={
            "environment_handle": trace_handle_payload,
            "capabilities": capabilities_payload,
        },
    )


def _redact_external_handle_for_trace(handle_payload: dict[str, Any]) -> dict[str, Any]:
    redacted = dict(handle_payload or {})
    env_vars = redacted.get("env_vars")
    if isinstance(env_vars, dict):
        redacted["env_vars"] = {str(key): "<redacted>" for key in env_vars}
    return redacted


def _environment_handle_payload(environment_lease: Any | None, lease_binding: Any) -> dict[str, Any]:
    if environment_lease is not None and hasattr(environment_lease, "to_descriptor"):
        return environment_lease.to_descriptor()
    lease = getattr(lease_binding, "resource_lease", None)
    if lease is not None and hasattr(lease, "to_dict"):
        return lease.to_dict()
    return _environment_descriptor_from_binding(lease_binding)


def _environment_capabilities_payload(environment_lease: Any | None) -> dict[str, Any]:
    environment = getattr(environment_lease, "environment", None)
    capabilities = getattr(environment, "capabilities", None)
    if hasattr(capabilities, "model_dump"):
        return capabilities.model_dump(mode="python")
    if isinstance(capabilities, dict):
        return dict(capabilities)
    return {}


def _prepare_verifier_environment_context(
    *,
    session: AgentRuntimeSession,
    lease_binding: Any,
    artifact_sink: RuntimeArtifactSink,
) -> None:
    environment_lease = _environment_lease_from_binding(
        lease_binding,
        artifact_sink=artifact_sink,
    )
    if environment_lease is not None:
        session.runtime_context["environment_lease"] = environment_lease


async def _carry_scheduler_submission_patch_for_verifier(
    *,
    session: AgentRuntimeSession,
    scheduler_result: SchedulerResult,
) -> None:
    if session.benchmark_kit_id != "swebench":
        return
    output = scheduler_result.agent_output
    if not isinstance(output, dict):
        return
    for key in ("patch_content", "patch", "diff"):
        value = output.get(key)
        if isinstance(value, str) and value.strip():
            return
    answer = output.get("answer")
    if isinstance(answer, str) and "diff --git" in answer:
        return

    environment_lease = session.runtime_context.get("environment_lease")
    if environment_lease is None:
        return
    submission_patch = await _read_scheduler_submission_patch(environment_lease)
    if not submission_patch:
        submission_patch = await _read_scheduler_git_diff(environment_lease)
    if submission_patch:
        output["patch_content"] = submission_patch


async def _read_scheduler_submission_patch(environment_lease: Any) -> str:
    reader = getattr(environment_lease, "read_file", None)
    if not callable(reader):
        return ""
    try:
        payload = await reader("/workspace/submission.patch")
    except Exception:
        return ""
    return _decode_bytes_payload(payload)


async def _read_scheduler_git_diff(environment_lease: Any) -> str:
    executor = getattr(environment_lease, "exec", None)
    if not callable(executor):
        return ""
    for command in (
        "cd /workspace && git diff --binary --no-color",
        "cd /workspace && git diff --binary --no-color HEAD --",
    ):
        try:
            result = await executor(command, timeout_s=10)
        except Exception:
            continue
        if getattr(result, "exit_code", 1) not in (0, None):
            continue
        stdout = _decode_bytes_payload(getattr(result, "stdout", ""))
        if "diff --git" in stdout or stdout.startswith("*** Begin Patch"):
            return stdout
    return ""


def _decode_bytes_payload(payload: Any) -> str:
    if isinstance(payload, (bytes, bytearray)):
        return payload.decode("utf-8", errors="replace")
    if payload is None:
        return ""
    return str(payload)


def _verifier_metric_payload(judge_output: dict[str, Any]) -> dict[str, Any]:
    return {
        "status": judge_output.get("status"),
        "resolved": bool(judge_output.get("resolved")),
        "score": float(judge_output.get("score") or 0.0),
        "failure_reason": judge_output.get("failure_reason"),
    }


def _environment_lease_from_binding(
    lease_binding: Any,
    *,
    artifact_sink: RuntimeArtifactSink,
) -> EnvironmentLease | None:
    if lease_binding is None:
        return None
    environment_lease = getattr(lease_binding, "environment_lease", None)
    if environment_lease is not None:
        environment_lease.artifact_sink = artifact_sink
        return environment_lease
    return None


def _payload_environment_lease(payload: dict[str, Any]) -> EnvironmentLease | None:
    environment_lease = payload.get("environment_lease")
    if environment_lease is not None and hasattr(environment_lease, "to_descriptor"):
        return environment_lease
    return None


def _payload_with_environment_lease(
    payload: dict[str, Any],
    *,
    session: AgentRuntimeSession,
    lease_binding: Any,
    artifact_sink: RuntimeArtifactSink,
) -> dict[str, Any]:
    environment_lease = session.runtime_context.get("environment_lease")
    if environment_lease is None:
        environment_lease = _environment_lease_from_binding(
            lease_binding,
            artifact_sink=artifact_sink,
        )
    if environment_lease is None:
        return dict(payload)
    updated = dict(payload)
    updated["environment_lease"] = environment_lease
    session.runtime_context.setdefault("environment_lease", environment_lease)
    return updated


def _scope_session_to_trial(
    session: AgentRuntimeSession,
    *,
    artifact_sink: RuntimeArtifactSink,
    trial_id: str,
) -> None:
    session.session_id = f"{session.session_id}-{trial_id}"
    session.artifact_sink = artifact_sink
    base_dir = Path(getattr(artifact_sink, "_base_dir"))
    trial_root = (
        base_dir
        / session.run_id
        / "artifacts"
        / session.task_id
        / session.sample_id
        / "trials"
        / trial_id
    )
    session.artifact_layout = {
        "sample_root": str(trial_root),
        "artifacts_dir": str(trial_root),
        "verifier_result": str(trial_root / "verifier" / "verifier_result.json"),
        "runtime_metadata": str(trial_root / "infra" / "trial_result.json"),
        "legacy_runtime_metadata": str(trial_root / "infra" / "runtime_metadata.json"),
        "raw_error": str(trial_root / "infra" / "raw_error.json"),
    }


def _verifier_resource_plan(plan: CompiledRuntimePlan) -> dict[str, Any]:
    resource_plan = dict(plan.resource_plan or {})
    if plan.verifier_environment_profile_id:
        profile = _verifier_environment_profile(
            plan,
            profile_id=plan.verifier_environment_profile_id,
            provider=str(resource_plan.get("resource_kind") or plan.environment_provider or ""),
        )
        resource_plan["environment_profile"] = profile
        resource_plan["provider_config"] = dict(profile.get("config") or {})
        resource_plan["resources"] = dict(profile.get("resources") or {})
        resource_plan["startup_env"] = dict(profile.get("startup_env") or {})
        resource_plan["lifecycle"] = str(profile.get("lifecycle") or plan.lifecycle or "per_sample")
        resource_plan["resource_kind"] = str(profile.get("provider") or resource_plan.get("resource_kind") or "")
    return resource_plan


def _verifier_acquire_sample(
    plan: CompiledRuntimePlan,
    sample: dict[str, Any],
) -> dict[str, Any]:
    del plan
    verifier_sample = dict(sample or {})
    verifier_sample.pop("sandbox", None)
    return verifier_sample


def _verifier_environment_profile(
    plan: CompiledRuntimePlan,
    *,
    profile_id: str,
    provider: str,
) -> dict[str, Any]:
    profile = _kit_environment_profile(plan, profile_id)
    if not profile:
        profile = dict(plan.environment_profile or {})
    profile["profile_id"] = profile_id
    if provider:
        profile["provider"] = provider
    else:
        profile.setdefault("provider", plan.environment_provider)
    profile.setdefault("config", {})
    profile.setdefault("lifecycle", plan.lifecycle or "per_sample")
    return profile


def _kit_environment_profile(plan: CompiledRuntimePlan, profile_id: str) -> dict[str, Any]:
    profiles = getattr(getattr(plan, "kit_entry", None), "environment_profiles", None)
    if not isinstance(profiles, dict):
        return {}
    profile = profiles.get(profile_id)
    if hasattr(profile, "model_dump"):
        dumped = profile.model_dump(mode="python")
        return dict(dumped) if isinstance(dumped, dict) else {}
    if isinstance(profile, dict):
        return dict(profile)
    return {}


def _merge_cleanup_failure(
    failure: FailureEnvelope | None,
    cleanup_failure: FailureEnvelope,
) -> FailureEnvelope:
    if failure is None:
        return cleanup_failure
    details = dict(failure.details or {})
    cleanup_failures = list(details.get("cleanup_failures") or [])
    cleanup_failures.append(cleanup_failure.to_dict())
    details["cleanup_failures"] = cleanup_failures
    failure.details = details
    return failure


def _build_trial_result_from_session(
    *,
    trial_id: str,
    scheduler_result: SchedulerResult | None,
    session: AgentRuntimeSession,
    artifact_refs: list[ArtifactRef],
    trace_ref: ArtifactRef,
    environment_descriptor: dict[str, Any],
    failure: FailureEnvelope | None,
) -> TrialResult:
    scheduler_payload = scheduler_result.to_dict() if scheduler_result is not None else {}
    verifier_payload = session.judge_outcome.judge_output if session.judge_outcome is not None else {}
    failure_payload = failure.to_dict() if failure is not None else None
    status = _trial_result_status(
        scheduler_result=scheduler_result,
        verifier_payload=verifier_payload,
        failure=failure,
    )
    return TrialResult(
        trial_id=trial_id,
        status=status,
        scheduler_result=to_json_compatible(scheduler_payload),
        verifier_result=to_json_compatible(verifier_payload),
        environment_descriptor=to_json_compatible(environment_descriptor),
        artifact_refs=list(artifact_refs),
        trace_ref=trace_ref,
        failure=to_json_compatible(failure_payload) if failure_payload is not None else None,
    )


def _trial_result_status(
    *,
    scheduler_result: SchedulerResult | None,
    verifier_payload: dict[str, Any],
    failure: FailureEnvelope | None,
) -> str:
    if failure is not None:
        return "failed"
    status = verifier_payload.get("status")
    if status in {"completed", "failed", "aborted"}:
        return str(status)
    if scheduler_result is not None:
        return scheduler_result.status
    return "failed"


def _write_sample_infra_artifacts(
    *,
    artifact_sink: RuntimeArtifactSink,
    plan: CompiledRuntimePlan,
    session: AgentRuntimeSession,
    sample: dict[str, Any],
    trial_policy: TrialPolicy,
    trial_results: list[TrialResult],
    aggregate: Any,
    scheduler_result: SchedulerResult | None,
    failure: FailureEnvelope | None,
) -> list[ArtifactRef]:
    effective_config, source_layers = _effective_config_payload(
        plan=plan,
        trial_policy=trial_policy,
    )
    effective_config_ref = artifact_sink.write_effective_config(
        run_id=session.run_id,
        task_id=session.task_id,
        sample_id=session.sample_id,
        final_config=effective_config,
        source_layers=source_layers,
    )
    aggregate_ref = artifact_sink.write_trial_aggregate(
        run_id=session.run_id,
        task_id=session.task_id,
        sample_id=session.sample_id,
        aggregate=aggregate,
    )
    artifact_refs: list[ArtifactRef] = [effective_config_ref, aggregate_ref]
    for trial_result in trial_results:
        artifact_refs.extend(trial_result.artifact_refs)
        artifact_refs.append(trial_result.trace_ref)

    primary_trial = trial_results[0] if trial_results else None
    failure_payload = _sample_failure_payload(
        failure=failure,
        trial_results=trial_results,
        scheduler_result=scheduler_result,
    )
    status = _sample_record_status(trial_results=trial_results, failure_payload=failure_payload)
    sample_record = {
        "run_id": session.run_id,
        "task_id": session.task_id,
        "sample_id": session.sample_id,
        "dut_id": str(plan.dut_id or "dut"),
        "input_ref": _input_ref(sample=sample, session=session),
        "trial_policy": trial_policy.to_dict(),
        "trial_results": [result.model_dump(mode="python") for result in trial_results],
        "aggregate_result": aggregate.to_dict() if hasattr(aggregate, "to_dict") else to_json_compatible(aggregate),
        "scheduler_result": scheduler_result.to_dict() if scheduler_result is not None else {},
        "verifier_result": primary_trial.verifier_result if primary_trial is not None else {},
        "environment_descriptor": primary_trial.environment_descriptor if primary_trial is not None else {},
        "effective_config_ref": effective_config_ref.model_dump(mode="python"),
        "artifacts": [ref.model_dump(mode="python") for ref in artifact_refs],
        "status": status,
        "failure": failure_payload,
    }
    sample_ref = artifact_sink.write_artifact(
        run_id=session.run_id,
        task_id=session.task_id,
        sample_id=session.sample_id,
        owner="infra",
        name="sample_record.json",
        content=sample_record,
        mime_type="application/json",
    )
    if isinstance(sample_ref, ArtifactRef):
        artifact_refs.append(sample_ref)
    return artifact_refs


def _effective_config_payload(
    *,
    plan: CompiledRuntimePlan,
    trial_policy: TrialPolicy,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    final_config = {
        "run_id": plan.run_id,
        "dut_id": plan.dut_id,
        "agent_id": plan.agent_id,
        "env_id": plan.env_id,
        "benchmark_id": plan.benchmark_id,
        "kit_id": plan.kit_id,
        "kit_config": dict(plan.kit_config or {}),
        "agent_config": dict(plan.agent_config or {}),
        "scheduler_type": plan.scheduler_type,
        "scheduler_config": dict(plan.scheduler_config or {}),
        "environment_provider": plan.environment_provider,
        "environment_profile_id": plan.environment_profile_id,
        "environment_profile": to_json_compatible(plan.environment_profile or {}),
        "provider_config": dict(plan.provider_config or {}),
        "startup_env": dict(plan.startup_env or {}),
        "resources": dict(plan.resources or {}),
        "verifier_environment_policy": plan.verifier_environment_policy,
        "verifier_environment_profile_id": plan.verifier_environment_profile_id,
        "trial_policy": trial_policy.to_dict(),
        "plan_id": plan.plan_id,
    }
    return final_config, [{"name": "compiled_runtime_plan", "values": final_config}]


def _input_ref(*, sample: dict[str, Any], session: AgentRuntimeSession) -> dict[str, Any]:
    ref = {
        "run_id": session.run_id,
        "task_id": session.task_id,
        "sample_id": session.sample_id,
    }
    for key in ("id", "sample_id", "instance_id", "task_id", "dataset", "split"):
        value = sample.get(key)
        if value not in (None, ""):
            ref[key] = to_json_compatible(value)
    return ref


def _sample_failure_payload(
    *,
    failure: FailureEnvelope | None,
    trial_results: list[TrialResult],
    scheduler_result: SchedulerResult | None,
) -> dict[str, Any] | None:
    if failure is not None:
        return failure.to_dict()
    if scheduler_result is not None and scheduler_result.failure is not None:
        return scheduler_result.failure.to_dict()
    for result in trial_results:
        if result.failure is not None:
            return to_json_compatible(result.failure)
        failure_code = result.verifier_result.get("failure_code")
        if failure_code:
            return {
                "failure_domain": "verifier",
                "failure_stage": "run_verifier",
                "failure_code": str(failure_code),
                "summary": str(result.verifier_result.get("failure_reason") or failure_code),
                "trial_id": result.trial_id,
            }
    return None


def _sample_record_status(
    *,
    trial_results: list[TrialResult],
    failure_payload: dict[str, Any] | None,
) -> str:
    if failure_payload is not None:
        return "failed"
    statuses = {result.status for result in trial_results}
    if "aborted" in statuses:
        return "aborted"
    if "failed" in statuses:
        return "failed"
    return "completed"


def _emit_sample_failed_summary(
    *,
    trace: ObservabilityTrace | None,
    session: AgentRuntimeSession,
    trial_id: str,
    failure: FailureEnvelope | None,
    judge_outcome: RuntimeJudgeOutcome | None,
) -> None:
    if trace is None:
        return
    payload: dict[str, Any] = {
        "run_id": session.run_id,
        "task_id": session.task_id,
        "sample_id": session.sample_id,
        "trial_id": trial_id,
        "scheduler_type": session.scheduler_type,
        "benchmark_kit_id": session.benchmark_kit_id,
    }
    if failure is not None:
        payload["failure_code"] = failure.failure_code
        payload["failure"] = failure.to_dict()
    judge_output = judge_outcome.judge_output if judge_outcome is not None else {}
    verifier_failure_code = judge_output.get("failure_code")
    if verifier_failure_code:
        payload["verifier_failure_code"] = verifier_failure_code
    if failure is None and not verifier_failure_code:
        return
    trace.emit("sample.failed", to_json_compatible(payload), sample_id=session.sample_id)


def _scheduler_failure_from_result(
    *,
    session: AgentRuntimeSession,
    scheduler_result: SchedulerResult | None,
) -> FailureEnvelope | None:
    if scheduler_result is None:
        return None
    if scheduler_result.failure is not None:
        return scheduler_result.failure
    if scheduler_result.status not in {"failed", "aborted"}:
        return None
    agent_output = dict(scheduler_result.agent_output or {})
    failure_code = str(
        agent_output.get("failure_code")
        or f"client_execution.run_scheduler.{session.scheduler_type}.scheduler_failed"
    )
    return FailureEnvelope(
        failure_domain="client_execution",
        failure_stage="run_scheduler",
        failure_code=failure_code,
        component_kind="scheduler",
        component_id=f"{session.scheduler_type}.scheduler",
        owner="runtime_scheduler_core",
        retryable=False,
        summary=str(agent_output.get("failure_reason") or f"{session.scheduler_type} scheduler failed"),
        first_bad_step=f"{session.scheduler_type}.scheduler.run",
        suspect_files=(f"src/gage_eval/agent_runtime/schedulers/{session.scheduler_type}.py",),
        details={
            "scheduler_status": scheduler_result.status,
            "benchmark_kit_id": session.benchmark_kit_id,
            "sample_id": session.sample_id,
        },
    )


def _normalize_judge_output(
    payload: dict[str, Any],
    *,
    scheduler_result: SchedulerResult | None,
    judge_source: str,
) -> dict[str, Any]:
    status = str(payload.get("status") or ("failed" if payload.get("failure_reason") else "completed"))
    inferred_resolved, inferred_score, inferred_reason = _infer_benchmark_judge_fields(
        payload,
        status=status,
    )
    resolved = bool(payload.get("resolved", inferred_resolved))
    score = payload.get("score")
    if score is None:
        score = inferred_score
    normalized = {
        "status": status,
        "resolved": resolved,
        "failure_reason": payload.get("failure_reason") or inferred_reason,
        "failure_domain": payload.get("failure_domain"),
        "score": score,
        "summary": payload.get("summary") or payload.get("message") or "",
        "artifact_paths": dict(payload.get("artifact_paths") or (scheduler_result.artifact_paths if scheduler_result is not None else {})),
        "runtime_handle": dict((scheduler_result.runtime_state if scheduler_result is not None else {})),
        "judge_source": payload.get("judge_source") or judge_source,
    }
    diagnostic_reason, diagnostic_details = _build_diagnostic_fields(
        payload,
        scheduler_result=scheduler_result,
        normalized=normalized,
    )
    if diagnostic_reason:
        normalized["diagnostic_reason"] = diagnostic_reason
    if diagnostic_details:
        normalized["diagnostic_details"] = diagnostic_details
    for key, value in payload.items():
        normalized.setdefault(key, value)
    return to_json_compatible(normalized)


def _update_swebench_diagnostics_post_verifier(
    *,
    session: AgentRuntimeSession,
    verifier_output: dict[str, Any],
) -> None:
    try:
        from gage_eval.agent_eval_kits.swebench.artifacts import (
            update_swebench_diagnostics_post_verifier,
        )

        update_swebench_diagnostics_post_verifier(
            session=session,
            verifier_output=verifier_output,
        )
    except Exception:
        return


def _infer_benchmark_judge_fields(
    payload: dict[str, Any],
    *,
    status: str,
) -> tuple[bool, float, str | None]:
    """Infer canonical judge fields from benchmark-native verifier payloads."""

    if "appworld" in payload and isinstance(payload.get("appworld"), dict):
        return _infer_appworld_judge_fields(payload, status=status)
    if "tau2" in payload and isinstance(payload.get("tau2"), dict):
        return _infer_tau2_judge_fields(payload, status=status)
    resolved = status == "completed"
    return resolved, (1.0 if resolved else 0.0), payload.get("failure_reason")


def _infer_appworld_judge_fields(
    payload: dict[str, Any],
    *,
    status: str,
) -> tuple[bool, float, str | None]:
    """Infer AppWorld success from TGC/tests instead of completion status."""

    appworld_payload = payload.get("appworld")
    if not isinstance(appworld_payload, dict):
        resolved = status == "completed"
        return resolved, (1.0 if resolved else 0.0), payload.get("failure_reason")

    appworld_status = str(appworld_payload.get("status") or "")
    appworld_failure_reason = appworld_payload.get("failure_reason")
    if appworld_status == "error" or appworld_failure_reason:
        return False, 0.0, str(appworld_failure_reason or "appworld_error")

    tgc = _coerce_float(appworld_payload.get("tgc"))
    tests = appworld_payload.get("tests") if isinstance(appworld_payload.get("tests"), dict) else {}
    fail_count = _coerce_collection_size(tests.get("fails"))
    pass_count = _coerce_collection_size(tests.get("passes"))

    if tgc is not None:
        resolved = tgc >= (1.0 - 1e-6)
        failure_reason = None if resolved else "task_incomplete"
        return resolved, float(tgc), failure_reason

    if fail_count is not None or pass_count is not None:
        resolved = (fail_count or 0) == 0 and (pass_count or 0) > 0
        failure_reason = None if resolved else "assertion_failed"
        return resolved, (1.0 if resolved else 0.0), failure_reason

    return False, 0.0, payload.get("failure_reason") or "missing_appworld_success_signal"


def _infer_tau2_judge_fields(
    payload: dict[str, Any],
    *,
    status: str,
) -> tuple[bool, float, str | None]:
    """Infer Tau2 success from reward instead of completion status."""

    tau2_payload = payload.get("tau2")
    if not isinstance(tau2_payload, dict):
        resolved = status == "completed"
        return resolved, (1.0 if resolved else 0.0), payload.get("failure_reason")

    reward = _coerce_float(tau2_payload.get("reward"))
    termination_reason = tau2_payload.get("termination_reason")
    if reward is None:
        return False, 0.0, payload.get("failure_reason") or "missing_reward"

    resolved = abs(reward - 1.0) <= 1e-6
    failure_reason = None if resolved else str(termination_reason or payload.get("failure_reason") or "reward_below_threshold")
    return resolved, float(reward), failure_reason


def _coerce_float(value: Any) -> float | None:
    """Coerce a numeric payload field to float when possible."""

    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _coerce_collection_size(value: Any) -> int | None:
    """Resolve the size of a list-like verifier field."""

    if value is None:
        return None
    if isinstance(value, (list, tuple, set, dict)):
        return len(value)
    return None


def _build_diagnostic_fields(
    payload: dict[str, Any],
    *,
    scheduler_result: SchedulerResult | None,
    normalized: dict[str, Any],
) -> tuple[str | None, dict[str, Any] | None]:
    """Build human-auditable diagnostics from verifier payloads and agent trace."""

    failure_reason = normalized.get("failure_reason")
    agent_output = scheduler_result.agent_output if scheduler_result is not None else {}
    agent_trace_summary = _summarize_agent_trace(agent_output.get("agent_trace"))
    raw_details = payload.get("diagnostic_details")
    details: dict[str, Any] = dict(raw_details) if isinstance(raw_details, dict) else {}
    if agent_trace_summary:
        details["agent_trace_summary"] = agent_trace_summary
    diagnostic_reason = payload.get("diagnostic_reason")
    if diagnostic_reason not in (None, ""):
        return str(diagnostic_reason), details or None
    if failure_reason:
        return str(failure_reason), details or None
    return None, details or None


def _summarize_agent_trace(agent_trace: Any) -> dict[str, Any] | None:
    if not isinstance(agent_trace, list) or not agent_trace:
        return None
    summary_steps: list[dict[str, Any]] = []
    for step in agent_trace[-3:]:
        if not isinstance(step, dict):
            continue
        summary_steps.append(
            {
                "trace_step": step.get("trace_step"),
                "trace_role": step.get("trace_role"),
                "name": step.get("name") or step.get("tool"),
                "response_return_status": step.get("status"),
                "input_excerpt": _truncate_text(_render_trace_value(step.get("input"))),
                "output_excerpt": _truncate_text(_render_trace_value(step.get("output"))),
            }
        )
    return {
        "step_count": len(agent_trace),
        "last_steps": summary_steps,
    }


def _render_trace_value(value: Any) -> str:
    if value in (None, "", [], {}):
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, dict):
        for key in ("answer", "text", "content", "stdout", "stderr", "error"):
            nested = value.get(key)
            if isinstance(nested, str) and nested.strip():
                return nested
        return str(to_json_compatible(value))
    return str(value)


def _truncate_text(value: Any, *, limit: int = 240) -> str:
    text = str(value or "").strip()
    if not text:
        return ""
    if len(text) <= limit:
        return text
    return f"{text[:limit]}..."


def _build_runtime_model_output(
    session: AgentRuntimeSession,
    scheduler_result: SchedulerResult | None,
) -> dict[str, Any]:
    agent_output = dict(scheduler_result.agent_output if scheduler_result is not None else {})
    if session.prompt_context:
        agent_output.setdefault("prompt_context", to_json_compatible(session.prompt_context))
    if session.runtime_context:
        agent_output.setdefault("runtime_context", to_json_compatible(session.runtime_context))
    if session.judge_outcome is not None:
        agent_output["runtime_judge_outcome"] = session.judge_outcome.to_dict()
    agent_output["runtime_session"] = {
        "session_id": session.session_id,
        "runtime_metadata_path": session.artifact_layout["runtime_metadata"],
        "verifier_result_path": session.artifact_layout["verifier_result"],
        "sample_root": session.artifact_layout["sample_root"],
        "scheduler_type": session.scheduler_type,
        "benchmark_kit_id": session.benchmark_kit_id,
    }
    return to_json_compatible(agent_output)


def _build_failed_runtime_model_output(
    session: AgentRuntimeSession,
    failure: FailureEnvelope,
) -> dict[str, Any]:
    return to_json_compatible(
        {
        "answer": "",
        "agent_trace": [],
        "runtime_failure": failure.to_dict(),
        "runtime_judge_outcome": session.judge_outcome.to_dict() if session.judge_outcome is not None else None,
        "runtime_session": {
            "session_id": session.session_id,
            "runtime_metadata_path": session.artifact_layout["runtime_metadata"],
            "verifier_result_path": session.artifact_layout["verifier_result"],
            "sample_root": session.artifact_layout["sample_root"],
            "scheduler_type": session.scheduler_type,
            "benchmark_kit_id": session.benchmark_kit_id,
        },
        }
    )


def _attach_agent_eval_output(
    output: dict[str, Any],
    *,
    aggregate: Any,
    trial_results: list[TrialResult],
) -> dict[str, Any]:
    """Attach trial aggregate projection to the runtime model output."""

    agent_eval = output.setdefault("agent_eval", {})
    if not isinstance(agent_eval, dict):
        agent_eval = {}
        output["agent_eval"] = agent_eval
    agent_eval["trial_aggregate"] = (
        aggregate.to_dict() if hasattr(aggregate, "to_dict") else to_json_compatible(aggregate)
    )
    agent_eval["trial_results"] = [
        result.model_dump(mode="python") for result in trial_results
    ]
    return output


def _safe_persist_raw_error(
    *,
    artifact_sink: RuntimeArtifactSink,
    session: AgentRuntimeSession,
    error: BaseException,
) -> str | None:
    """Persist the raw error payload on a best-effort basis."""

    try:
        return artifact_sink.persist_raw_error(session=session, error=error)
    except Exception:
        return None


def _safe_persist_verifier_result(
    artifact_sink: RuntimeArtifactSink,
    outcome: RuntimeJudgeOutcome,
) -> None:
    """Persist verifier output on a best-effort basis."""

    try:
        artifact_sink.persist_verifier_result(outcome)
    except Exception:
        return None


def _safe_persist_runtime_metadata(
    *,
    artifact_sink: RuntimeArtifactSink,
    session: AgentRuntimeSession,
    scheduler_result: SchedulerResult | None,
    failure: FailureEnvelope | None,
) -> None:
    """Persist runtime metadata on a best-effort basis."""

    try:
        artifact_sink.persist_runtime_metadata(
            session=session,
            scheduler_result=scheduler_result,
            failure=failure,
        )
    except Exception:
        return None
