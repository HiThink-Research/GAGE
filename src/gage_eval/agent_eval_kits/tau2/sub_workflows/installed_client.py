from __future__ import annotations

from gage_eval.agent_eval_kits.tau2.artifacts import persist_tau2_artifacts
from gage_eval.agent_runtime.compiled_plan import SchedulerWorkflowBundle


def build_workflow_bundle() -> SchedulerWorkflowBundle:
    """Build the installed-client Tau2 workflow."""

    return SchedulerWorkflowBundle(
        bundle_id="tau2.installed_client",
        benchmark_kit_id="tau2",
        scheduler_type="installed_client",
        prepare_inputs=_prepare_inputs,
        prepare_environment=_prepare_environment,
        capture_environment_artifacts=_capture_environment_artifacts,
        finalize_result=_finalize_result,
        failure_normalizer=lambda **_: {},
    )


def _prepare_inputs(*, session, sample, payload, sandbox_provider=None):
    initialize_result = dict(session.benchmark_state.get("initialize_result") or {})
    return {
        "messages": list(sample.get("messages") or []),
        "policy": session.prompt_context.get("policy"),
        "domain": session.prompt_context.get("domain"),
        "tools_schema": list(session.prompt_context.get("tools_schema") or []),
        "user_simulator_state": initialize_result.get("user_simulator_state"),
        "metadata": dict(sample.get("metadata") or {}),
    }


def _prepare_environment(*, session, sample, sandbox_provider=None):
    return {"resource_kind": "local_process"}


def _capture_environment_artifacts(*, session, sample, scheduler_output, sandbox_provider=None):
    environment_lease = (getattr(session, "runtime_context", {}) or {}).get("environment_lease")
    _record_tau2_agent_usage(runtime_source=environment_lease or sandbox_provider, scheduler_output=scheduler_output)
    artifact_paths = dict(scheduler_output.get("artifact_paths") or {})
    artifact_paths.update(
        persist_tau2_artifacts(
            session=session,
            scheduler_output=scheduler_output,
            environment_lease=environment_lease,
            sandbox_provider=sandbox_provider,
        )
    )
    return artifact_paths


def _finalize_result(*, session, sample, scheduler_output, artifact_paths):
    output = dict(scheduler_output or {})
    output.setdefault("artifact_paths", dict(artifact_paths or {}))
    return output


def _record_tau2_agent_usage(*, runtime_source, scheduler_output) -> None:
    runtime = getattr(runtime_source, "environment", None)
    if runtime is None and hasattr(runtime_source, "get_handle"):
        handle = runtime_source.get_handle()
        runtime = getattr(handle, "sandbox", None) if handle is not None else None
    if runtime is None:
        runtime = runtime_source
    recorder = getattr(runtime, "record_agent_usage", None)
    if not callable(recorder):
        return
    recorder((scheduler_output or {}).get("usage"))
