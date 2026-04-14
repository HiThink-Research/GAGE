from __future__ import annotations

from gage_eval.agent_eval_kits.terminal_bench.artifacts import (
    normalize_terminal_answer,
    persist_terminal_artifacts,
)
from gage_eval.agent_runtime.compiled_plan import SchedulerWorkflowBundle
from gage_eval.agent_eval_kits.terminal_bench.units import build_terminal_messages, build_terminal_tools


def build_workflow_bundle() -> SchedulerWorkflowBundle:
    """Build the framework-loop workflow for terminal benchmark."""

    return SchedulerWorkflowBundle(
        bundle_id="terminal_bench.framework_loop",
        benchmark_kit_id="terminal_bench",
        scheduler_type="framework_loop",
        build_loop_inputs=_build_loop_inputs,
        inject_prompt_context=_inject_prompt_context,
        inject_tool_schemas=_inject_tool_schemas,
        finalize_loop_result=_finalize_loop_result,
        failure_normalizer=lambda **_: {},
    )


def _build_loop_inputs(*, session, sample, payload):
    return {"messages": build_terminal_messages(sample)}


def _inject_prompt_context(*, session, sample, payload):
    return dict(session.prompt_context or {})


def _inject_tool_schemas(*, session, sample, payload):
    return build_terminal_tools(sample)


def _finalize_loop_result(*, session, sample, scheduler_output, sandbox_provider=None):
    output = dict(scheduler_output or {})
    artifact_paths = dict(output.get("artifact_paths") or {})
    artifact_paths.update(
        persist_terminal_artifacts(
            session=session,
            scheduler_output=output,
            sandbox_provider=sandbox_provider,
        )
    )
    normalized_answer = normalize_terminal_answer(
        session=session,
        sample=sample,
        scheduler_output=output,
        sandbox_provider=sandbox_provider,
    )
    if normalized_answer is not None:
        output["answer"] = normalized_answer
    output["artifact_paths"] = artifact_paths
    return output
