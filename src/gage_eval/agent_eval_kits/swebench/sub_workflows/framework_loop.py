from __future__ import annotations

from pathlib import Path

from gage_eval.agent_runtime.compiled_plan import SchedulerWorkflowBundle
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
    artifact_paths = dict(output.get("artifact_paths") or {})
    artifact_paths["submission_patch"] = _normalize_submission_patch_path(
        artifact_paths.get("submission_patch")
    )
    output["artifact_paths"] = artifact_paths
    return output


def _normalize_submission_patch_path(path: object) -> str:
    """Normalizes submission.patch evidence to a sample-root relative path."""

    if isinstance(path, str) and path.strip():
        return Path(path).name
    return "submission.patch"
