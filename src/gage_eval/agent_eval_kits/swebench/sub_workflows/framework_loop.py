from __future__ import annotations

from gage_eval.agent_runtime.compiled_plan import SchedulerWorkflowBundle
from gage_eval.agent_eval_kits.swebench.artifacts import (
    persist_swebench_artifacts,
    resolve_swebench_failure_category,
)
from gage_eval.agent_eval_kits.swebench.units import build_swebench_messages, build_swebench_tools


def build_workflow_bundle() -> SchedulerWorkflowBundle:
    """Build the framework-loop SWE-bench workflow."""

    return SchedulerWorkflowBundle(
        bundle_id="swebench.framework_loop",
        benchmark_kit_id="swebench",
        scheduler_type="framework_loop",
        build_loop_inputs=_build_loop_inputs,
        inject_prompt_context=_inject_prompt_context,
        inject_tool_schemas=_inject_tool_schemas,
        finalize_loop_result=_finalize_loop_result,
        failure_normalizer=lambda **_: {},
    )


def _build_loop_inputs(*, session, sample, payload):
    return {"messages": build_swebench_messages(sample)}


def _inject_prompt_context(*, session, sample, payload):
    return dict(session.prompt_context or {})


def _inject_tool_schemas(*, session, sample, payload):
    return build_swebench_tools(sample)


def _finalize_loop_result(*, session, sample, scheduler_output, sandbox_provider=None):
    output = dict(scheduler_output or {})
    output["artifact_paths"] = persist_swebench_artifacts(
        session=session,
        scheduler_output=output,
        sandbox_provider=sandbox_provider,
    )
    resolved_failure_category = resolve_swebench_failure_category(
        output=output,
        agent_trace=output.get("agent_trace"),
        materialized_artifact_paths=output["artifact_paths"],
    )
    if resolved_failure_category:
        output.setdefault("failure_category", resolved_failure_category)
    return output
