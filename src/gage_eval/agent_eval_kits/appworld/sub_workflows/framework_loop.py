from __future__ import annotations

from gage_eval.agent_eval_kits.appworld.artifacts import persist_appworld_artifacts
from gage_eval.agent_runtime.compiled_plan import SchedulerWorkflowBundle
from gage_eval.agent_eval_kits.appworld.units import (
    build_appworld_instruction,
    build_appworld_messages,
    build_appworld_tools,
)


def build_workflow_bundle(runtime_entry) -> SchedulerWorkflowBundle:
    """Build the framework-loop AppWorld workflow."""

    return SchedulerWorkflowBundle(
        bundle_id="appworld.framework_loop",
        benchmark_kit_id="appworld",
        scheduler_type="framework_loop",
        build_loop_inputs=_build_loop_inputs,
        inject_prompt_context=_inject_prompt_context,
        inject_tool_schemas=_inject_tool_schemas,
        finalize_loop_result=lambda **kwargs: _finalize_loop_result(runtime_entry=runtime_entry, **kwargs),
        failure_normalizer=lambda **_: {},
    )


def _build_loop_inputs(*, session, sample, payload):
    return {
        "messages": build_appworld_messages(
            sample,
            instruction_override=str(session.prompt_context.get("instruction") or ""),
        )
    }


def _inject_prompt_context(*, session, sample, payload):
    prompt_context = dict(session.prompt_context or {})
    prompt_context["instruction"] = build_appworld_instruction(
        sample,
        instruction_override=str(prompt_context.get("instruction") or ""),
    )
    return prompt_context


def _inject_tool_schemas(*, session, sample, payload):
    return build_appworld_tools(sample)


def _finalize_loop_result(*, runtime_entry, session, sample, scheduler_output, sandbox_provider=None):
    output = dict(scheduler_output or {})
    saved = runtime_entry.save(sample=sample, sandbox_provider=sandbox_provider)
    artifact_paths = dict(output.get("artifact_paths") or {})
    artifact_paths.update(
        persist_appworld_artifacts(
            session=session,
            scheduler_output=output,
            saved_payload=saved,
        )
    )
    output["artifact_paths"] = artifact_paths
    return output
