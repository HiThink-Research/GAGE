from __future__ import annotations

from gage_eval.agent_eval_kits.appworld.artifacts import persist_appworld_artifacts
from gage_eval.agent_runtime.compiled_plan import SchedulerWorkflowBundle
from gage_eval.agent_eval_kits.appworld.units import (
    build_appworld_instruction,
    build_appworld_tools,
    fetch_mcp_tool_schemas,
)
from gage_eval.evaluation.support_artifacts import resolve_support_field


def build_workflow_bundle(runtime_entry) -> SchedulerWorkflowBundle:
    """Build the installed-client AppWorld workflow."""

    return SchedulerWorkflowBundle(
        bundle_id="appworld.installed_client",
        benchmark_kit_id="appworld",
        scheduler_type="installed_client",
        prepare_inputs=_prepare_inputs,
        prepare_environment=_prepare_environment,
        capture_environment_artifacts=lambda **kwargs: _capture_environment_artifacts(runtime_entry=runtime_entry, **kwargs),
        finalize_result=_finalize_result,
        failure_normalizer=lambda **_: {},
    )


def _prepare_inputs(*, session, sample, payload, sandbox_provider=None):
    instruction = build_appworld_instruction(
        sample,
        instruction_override=str(session.prompt_context.get("instruction") or ""),
    )
    mcp_endpoint = session.prompt_context.get("mcp_endpoint")
    allowed_apps = list(session.prompt_context.get("allowed_apps") or [])
    mcp_client_id = resolve_support_field(sample, "mcp_client_id")
    if mcp_endpoint:
        try:
            tools_schema = fetch_mcp_tool_schemas(
                mcp_endpoint,
                mcp_client_id,
                allowed_apps=allowed_apps or None,
            )
        except Exception:
            tools_schema = build_appworld_tools(sample)
    else:
        tools_schema = build_appworld_tools(sample)
    return {
        "instruction": instruction,
        "allowed_apps": allowed_apps,
        "ground_truth_mode": session.prompt_context.get("ground_truth_mode"),
        "mcp_endpoint": mcp_endpoint,
        "env_endpoint": session.prompt_context.get("env_endpoint"),
        "tool_use_contract": "use_appworld_tools",
        "tools_schema": tools_schema,
        "metadata": dict(sample.get("metadata") or {}),
    }


def _prepare_environment(*, session, sample, sandbox_provider=None):
    return {"mcp_endpoint": session.prompt_context.get("mcp_endpoint")}


def _capture_environment_artifacts(*, runtime_entry, session, sample, scheduler_output, sandbox_provider=None):
    saved = runtime_entry.save(sample=sample, sandbox_provider=sandbox_provider)
    artifact_paths = dict(scheduler_output.get("artifact_paths") or {})
    artifact_paths.update(
        persist_appworld_artifacts(
            session=session,
            scheduler_output=scheduler_output,
            saved_payload=saved,
        )
    )
    return artifact_paths


def _finalize_result(*, session, sample, scheduler_output, artifact_paths):
    output = dict(scheduler_output or {})
    output.setdefault("artifact_paths", dict(artifact_paths or {}))
    return output
