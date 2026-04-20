from __future__ import annotations

import hashlib
from typing import Any

from gage_eval.agent_runtime.artifacts import RuntimeArtifactSink, RuntimeTraceEmitter
from gage_eval.agent_runtime.compiled_plan import CompiledRuntimePlan
from gage_eval.agent_runtime.contracts.failure import FailureEnvelope, FailureEnvelopeError
from gage_eval.agent_runtime.contracts.scheduler import SchedulerResult
from gage_eval.agent_runtime.failure_mapper import FailureMapper
from gage_eval.agent_runtime.serialization import to_json_compatible
from gage_eval.agent_runtime.session import AgentRuntimeSession
from gage_eval.agent_runtime.verifier.adapters import build_failure_result
from gage_eval.agent_runtime.verifier.contracts import RuntimeJudgeOutcome, VerifierInput
from gage_eval.observability.trace import ObservabilityTrace
from gage_eval.sandbox.provider import SandboxProvider


class DefaultVerifierRunner:
    """Runs the runtime-owned verifier adapter bound in the compiled plan."""

    def run(
        self,
        *,
        plan: CompiledRuntimePlan,
        session: AgentRuntimeSession,
        sample: dict[str, Any],
        scheduler_result: SchedulerResult,
        sandbox_provider,
    ) -> RuntimeJudgeOutcome:
        resources = dict(plan.judge_binding.verifier_resource_refs or {})
        adapter = resources.pop("adapter", None)
        if adapter is None:
            raise RuntimeError("runtime verifier adapter is not bound")
        verifier_input = VerifierInput(
            benchmark_kit_id=session.benchmark_kit_id,
            scheduler_type=session.scheduler_type,
            sample_id=session.sample_id,
            sample=sample,
            scheduler_result=scheduler_result.to_dict(),
            runtime_context={
                **dict(session.runtime_context or {}),
                "runtime_handle": session.resource_lease.handle_ref if session.resource_lease is not None else {},
                "sandbox_provider": sandbox_provider,
            },
            verifier_resources=resources,
        )
        verifier_result = adapter.run(verifier_input)
        judge_output = _normalize_judge_output(
            verifier_result.payload,
            scheduler_result=scheduler_result,
            judge_source=getattr(adapter, "judge_source", "runtime_verifier"),
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
            benchmark_config=dict(plan.benchmark_config or {}),
            artifact_layout=layout,
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
    ) -> None:
        self.compiled_plan = compiled_plan
        self.resource_manager = resource_manager
        self.session_factory = session_factory
        self.verifier_runner = verifier_runner
        self.artifact_sink = artifact_sink
        self.trace_emitter = trace_emitter
        self.failure_mapper = failure_mapper

    async def aexecute(
        self,
        *,
        sample: dict[str, Any],
        payload: dict[str, Any],
        trace: ObservabilityTrace | None = None,
    ) -> dict[str, Any]:
        """Execute one runtime-owned sample flow and return model output."""

        # STEP 1: Materialize the runtime session and sample-scoped trace context.
        session = self.session_factory.create(
            plan=self.compiled_plan,
            sample=sample,
            payload=payload,
        )
        self.trace_emitter.emit_session_start(trace, session)

        # STEP 2: Acquire the resource lease and bootstrap the benchmark runtime.
        lease_binding = None
        scheduler_result: SchedulerResult | None = None
        final_output: dict[str, Any] | None = None
        final_failure: FailureEnvelope | None = None
        try:
            try:
                bound_provider = payload.get("sandbox_provider")
                if isinstance(bound_provider, SandboxProvider):
                    lease_binding = self.resource_manager.bind_existing(
                        session,
                        resource_plan=self.compiled_plan.resource_plan,
                        sandbox_provider=bound_provider,
                    )
                else:
                    lease_binding = self.resource_manager.acquire(
                        session,
                        resource_plan=self.compiled_plan.resource_plan,
                        trace=trace,
                        sample=sample,
                    )
                session.resource_lease = lease_binding.resource_lease
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

            sandbox_provider = lease_binding.sandbox_provider if lease_binding is not None else None
            try:
                runtime_context = self.compiled_plan.kit_runtime_ref.bootstrap(
                    session=session,
                    sample=sample,
                    payload=payload,
                    sandbox_provider=sandbox_provider,
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
                scheduler_result = await self.compiled_plan.scheduler_handle.arun(
                    session=session,
                    sample=sample,
                    payload={**payload, "sandbox_config": self.compiled_plan.resource_plan.get("sandbox_config")},
                    workflow_bundle=self.compiled_plan.workflow_bundle,
                    sandbox_provider=sandbox_provider,
                )
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
                    outcome = self.verifier_runner.run(
                        plan=self.compiled_plan,
                        session=session,
                        sample=sample,
                        scheduler_result=scheduler_result,
                        sandbox_provider=sandbox_provider,
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
            try:
                if session.judge_outcome is not None:
                    self.artifact_sink.persist_verifier_result(session.judge_outcome)
                self.artifact_sink.persist_runtime_metadata(
                    session=session,
                    scheduler_result=scheduler_result,
                    failure=scheduler_result.failure if scheduler_result is not None else None,
                )
            except Exception as exc:
                failure = self.failure_mapper.map_exception(
                    exc,
                    failure_domain="persistence",
                    failure_stage="persist_outputs",
                    component_kind="artifact_sink",
                    component_id="runtime.artifact_sink.persist_outputs",
                    owner="runtime_artifact_core",
                    failure_code="persistence.persist_outputs.runtime.artifact_sink.persist_failed",
                    first_bad_step="runtime.artifact_sink.persist_outputs",
                    suspect_files=(
                        "src/gage_eval/agent_runtime/artifacts.py",
                        "src/gage_eval/agent_runtime/executor.py",
                    ),
                )
                raise FailureEnvelopeError(failure) from exc
            final_output = _build_runtime_model_output(session, scheduler_result)
        except FailureEnvelopeError as exc:
            failure = exc.failure
            final_failure = failure
            sandbox_provider = lease_binding.sandbox_provider if lease_binding is not None else None
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
            _safe_persist_runtime_metadata(
                artifact_sink=self.artifact_sink,
                session=session,
                scheduler_result=scheduler_result,
                failure=failure,
            )
            self.trace_emitter.emit_failure(trace, session, failure=failure)
            final_output = _build_failed_runtime_model_output(session, failure)

        cleanup_error: BaseException | None = None
        if lease_binding is not None:
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
            _safe_persist_runtime_metadata(
                artifact_sink=self.artifact_sink,
                session=session,
                scheduler_result=scheduler_result,
                failure=failure,
            )
            self.trace_emitter.emit_failure(trace, session, failure=failure)
            final_failure = failure
            final_output = _build_failed_runtime_model_output(session, failure)

        if final_failure is None:
            self.trace_emitter.emit_session_end(
                trace,
                session,
                scheduler_result=scheduler_result,
            )
        if final_output is None:
            return _build_runtime_model_output(session, scheduler_result)
        return final_output


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
