from __future__ import annotations

from typing import Any

from gage_eval.agent_eval_kits.tau2.artifacts import persist_tau2_artifacts
from gage_eval.agent_runtime.compiled_plan import SchedulerWorkflowBundle
from gage_eval.agent_eval_kits.tau2.units import build_tau2_messages, build_tau2_tools


def build_workflow_bundle() -> SchedulerWorkflowBundle:
    """Build the framework-loop Tau2 workflow."""

    return SchedulerWorkflowBundle(
        bundle_id="tau2.framework_loop",
        benchmark_kit_id="tau2",
        scheduler_type="framework_loop",
        build_loop_inputs=_build_loop_inputs,
        inject_prompt_context=_inject_prompt_context,
        inject_tool_schemas=_inject_tool_schemas,
        finalize_loop_result=_finalize_loop_result,
        failure_normalizer=lambda **_: {},
    )


def _build_loop_inputs(*, session, sample, payload):
    return {"messages": build_tau2_messages(sample)}


def _inject_prompt_context(*, session, sample, payload):
    return dict(session.prompt_context or {})


def _inject_tool_schemas(*, session, sample, payload):
    initialize_result = session.benchmark_state.get("initialize_result") or {}
    return build_tau2_tools(sample, initialize_result)


def _finalize_loop_result(*, session, sample, scheduler_output, sandbox_provider=None):
    output = dict(scheduler_output or {})
    _mark_loop_termination(
        sandbox_provider=sandbox_provider,
        loop_exit_reason=output.get("loop_exit_reason"),
    )
    _record_tau2_agent_usage(sandbox_provider=sandbox_provider, scheduler_output=output)
    artifact_paths = dict(output.get("artifact_paths") or {})
    artifact_paths.update(
        persist_tau2_artifacts(
            session=session,
            scheduler_output=output,
            sandbox_provider=sandbox_provider,
        )
    )
    output["artifact_paths"] = artifact_paths
    return output


def _mark_loop_termination(*, sandbox_provider, loop_exit_reason: Any | None) -> None:
    if loop_exit_reason not in {"tool_call_retry_budget", "max_turns"}:
        return
    if sandbox_provider is None:
        return
    handle = sandbox_provider.get_handle()
    runtime = handle.sandbox if handle is not None else None
    mark_agent_exhausted = getattr(runtime, "mark_agent_exhausted", None)
    if not callable(mark_agent_exhausted):
        return

    detail = (
        "no_tool_call_from_agent"
        if loop_exit_reason == "tool_call_retry_budget"
        else "agent_loop_max_turns"
    )
    mark_agent_exhausted(detail)


def _record_tau2_agent_usage(*, sandbox_provider, scheduler_output) -> None:
    if sandbox_provider is None:
        return
    handle = sandbox_provider.get_handle()
    runtime = handle.sandbox if handle is not None else None
    recorder = getattr(runtime, "record_agent_usage", None)
    if not callable(recorder):
        return
    recorder((scheduler_output or {}).get("usage"))
