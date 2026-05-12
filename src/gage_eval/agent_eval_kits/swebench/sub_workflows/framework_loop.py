from __future__ import annotations

from gage_eval.agent_runtime.compiled_plan import SchedulerWorkflowBundle
from gage_eval.agent_eval_kits.swebench.artifacts import persist_swebench_artifacts
from gage_eval.agent_eval_kits.swebench.tools import build_swebench_messages, build_swebench_tools


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
    return {
        "messages": build_swebench_messages(
            sample,
            working_dir=_working_dir_from_payload(payload),
        ),
        "required_tool": "submit_patch_tool",
    }


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
    return output


def _working_dir_from_payload(payload) -> str:
    lease = (payload or {}).get("environment_lease") if isinstance(payload, dict) else None
    metadata = getattr(lease, "metadata", None)
    if isinstance(metadata, dict):
        for key in ("exec_workdir", "workdir"):
            value = metadata.get(key)
            if isinstance(value, str) and value.startswith("/"):
                return value
    provider_config = (payload or {}).get("provider_config") if isinstance(payload, dict) else None
    if isinstance(provider_config, dict):
        for key in ("exec_workdir", "workdir"):
            value = provider_config.get(key)
            if isinstance(value, str) and value.startswith("/"):
                return value
    return "/app"
