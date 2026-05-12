from __future__ import annotations

from typing import Any

from gage_eval.agent_eval_kits.tau2.artifacts import persist_tau2_artifacts
from gage_eval.agent_eval_kits.tau2.tools import build_tau2_messages, build_tau2_tools
from gage_eval.agent_runtime.compiled_plan import SchedulerWorkflowBundle


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
    required_tool = None if (payload or {}).get("tool_choice") == "none" else "respond"
    return {
        "messages": build_tau2_messages(sample),
        "required_tool": required_tool,
        "plain_text_response_tool": "respond",
        "plain_text_response_argument": "message",
        "refresh_tool_schemas": True,
        "tool_text_parser": "tau2",
        "tool_result_user_message_field": "user_message",
        "benchmark_config": _benchmark_config_from_context(session=session, payload=payload),
    }


def _inject_prompt_context(*, session, sample, payload):
    return dict(session.prompt_context or {})


def _benchmark_config_from_context(*, session, payload) -> dict[str, Any]:
    benchmark_config = (payload or {}).get("benchmark_config")
    if isinstance(benchmark_config, dict):
        return dict(benchmark_config)
    runtime_context = getattr(session, "runtime_context", {}) or {}
    benchmark_config = runtime_context.get("benchmark_config")
    return dict(benchmark_config or {}) if isinstance(benchmark_config, dict) else {}


def _inject_tool_schemas(*, session, sample, payload):
    initialize_result = session.benchmark_state.get("initialize_result") or {}
    return build_tau2_tools(sample, initialize_result)


def _finalize_loop_result(*, session, sample, scheduler_output, sandbox_provider=None):
    output = dict(scheduler_output or {})
    environment_lease = _resolve_environment_lease(session)
    _mark_loop_termination(
        runtime_source=environment_lease or sandbox_provider,
        loop_exit_reason=output.get("loop_exit_reason"),
    )
    _record_tau2_agent_usage(runtime_source=environment_lease or sandbox_provider, scheduler_output=output)
    runtime_state = _capture_tau2_state(environment_lease or sandbox_provider)
    if runtime_state:
        output["runtime_state"] = runtime_state
    artifact_paths = dict(output.get("artifact_paths") or {})
    artifact_paths.update(
        persist_tau2_artifacts(
            session=session,
            scheduler_output=output,
            environment_lease=environment_lease,
            sandbox_provider=sandbox_provider,
        )
    )
    output["artifact_paths"] = artifact_paths
    return output


def _capture_tau2_state(runtime_source) -> dict[str, Any]:
    runtime = _resolve_runtime(runtime_source)
    if runtime is None:
        return {}
    getter = getattr(runtime, "get_state", None)
    if not callable(getter):
        return {}
    state = getter()
    return dict(state) if isinstance(state, dict) else {}


def _mark_loop_termination(*, runtime_source, loop_exit_reason: Any | None) -> None:
    if loop_exit_reason not in {"tool_call_retry_budget", "max_turns"}:
        return
    runtime = _resolve_runtime(runtime_source)
    mark_agent_exhausted = getattr(runtime, "mark_agent_exhausted", None)
    if not callable(mark_agent_exhausted):
        return
    detail = (
        "no_tool_call_from_agent"
        if loop_exit_reason == "tool_call_retry_budget"
        else "agent_loop_max_turns"
    )
    mark_agent_exhausted(detail)


def _record_tau2_agent_usage(*, runtime_source, scheduler_output) -> None:
    runtime = _resolve_runtime(runtime_source)
    recorder = getattr(runtime, "record_agent_usage", None)
    if not callable(recorder):
        return
    recorder((scheduler_output or {}).get("usage"))


def _resolve_environment_lease(session) -> Any | None:
    runtime_context = getattr(session, "runtime_context", {}) or {}
    return runtime_context.get("environment_lease")


def _resolve_runtime(runtime_source) -> Any | None:
    if runtime_source is None:
        return None
    runtime = getattr(runtime_source, "environment", None)
    if runtime is not None:
        return runtime
    if hasattr(runtime_source, "get_handle"):
        handle = runtime_source.get_handle()
        runtime = getattr(handle, "sandbox", None) if handle is not None else None
        if runtime is not None:
            return runtime
    return runtime_source
