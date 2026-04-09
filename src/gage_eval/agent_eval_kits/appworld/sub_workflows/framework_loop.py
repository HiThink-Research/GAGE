from __future__ import annotations

import json

from gage_eval.agent_eval_kits.common import resolve_sample_artifact_target
from gage_eval.agent_runtime.compiled_plan import SchedulerWorkflowBundle
from gage_eval.agent_eval_kits.appworld.units import build_appworld_messages, build_appworld_tools


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
    return {"messages": build_appworld_messages(sample)}


def _inject_prompt_context(*, session, sample, payload):
    return dict(session.prompt_context or {})


def _inject_tool_schemas(*, session, sample, payload):
    return build_appworld_tools(sample)


def _finalize_loop_result(*, runtime_entry, session, sample, scheduler_output, sandbox_provider=None):
    output = dict(scheduler_output or {})
    saved = runtime_entry.save(sample=sample, sandbox_provider=sandbox_provider)
    artifact_paths = dict(output.get("artifact_paths") or {})
    if saved:
        artifact_paths["appworld_save"] = _persist_appworld_save_artifact(session, saved)
    output["artifact_paths"] = artifact_paths
    return output


def _persist_appworld_save_artifact(session, saved: dict) -> str:
    """Persists the AppWorld save payload under the sample root."""

    target, relative_path = resolve_sample_artifact_target(session, "appworld_save.json")
    target.write_text(json.dumps(saved, ensure_ascii=False, indent=2), encoding="utf-8")
    return relative_path
