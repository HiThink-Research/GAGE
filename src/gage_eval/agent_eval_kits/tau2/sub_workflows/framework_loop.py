from __future__ import annotations

import json

from gage_eval.agent_eval_kits.common import resolve_sample_artifact_target
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
    artifact_paths = dict(output.get("artifact_paths") or {})
    state = _capture_tau2_state(sandbox_provider)
    if state:
        artifact_paths["tau2_state"] = _persist_tau2_state_artifact(session, state)
    output["artifact_paths"] = artifact_paths
    return output


def _capture_tau2_state(sandbox_provider) -> dict:
    """Captures the Tau2 runtime state when the sandbox exposes it."""

    if sandbox_provider is None:
        return {}
    handle = sandbox_provider.get_handle()
    runtime = handle.sandbox if handle is not None else None
    getter = getattr(runtime, "get_state", None)
    if not callable(getter):
        return {}
    state = getter()
    return dict(state) if isinstance(state, dict) else {}


def _persist_tau2_state_artifact(session, state: dict) -> str:
    """Persists the Tau2 runtime state under the sample root."""

    target, relative_path = resolve_sample_artifact_target(session, "tau2_state.json")
    target.write_text(json.dumps(state, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    return relative_path
