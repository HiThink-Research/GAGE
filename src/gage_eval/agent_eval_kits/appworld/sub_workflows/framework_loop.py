from __future__ import annotations

from gage_eval.agent_eval_kits.appworld.artifacts import persist_appworld_artifacts
from gage_eval.agent_runtime.compiled_plan import SchedulerWorkflowBundle
from gage_eval.evaluation.support_artifacts import resolve_support_field
from gage_eval.agent_eval_kits.appworld.units import (
    build_appworld_instruction,
    build_appworld_messages,
    build_appworld_tools,
    fetch_mcp_tool_schemas,
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
    import traceback
    mcp_endpoint = session.prompt_context.get("mcp_endpoint")
    if not mcp_endpoint:
        return build_appworld_tools(sample)
    mcp_client_id = resolve_support_field(sample, "mcp_client_id")
    allowed_apps = list(session.prompt_context.get("allowed_apps") or [])
    try:
        schemas = fetch_mcp_tool_schemas(
            mcp_endpoint,
            mcp_client_id,
            allowed_apps=allowed_apps or None,
        )
        import pathlib, datetime
        pathlib.Path("/tmp/gage_inject_tool_schemas_ok.txt").write_text(
            f"{datetime.datetime.now(datetime.timezone.utc).isoformat()}\n"
            f"mcp_endpoint={mcp_endpoint!r}\n"
            f"mcp_client_id={mcp_client_id!r}\n"
            f"schemas_count={len(schemas)}\n"
            f"first_tool={schemas[0].get('function', {}).get('name') if schemas else 'NONE'}\n"
        )
        return schemas
    except Exception:
        import pathlib, datetime
        log_path = pathlib.Path("/tmp/gage_inject_tool_schemas_error.txt")
        log_path.write_text(
            f"{datetime.datetime.now(datetime.timezone.utc).isoformat()}\n"
            f"mcp_endpoint={mcp_endpoint!r}\n"
            f"mcp_client_id={mcp_client_id!r}\n"
            f"allowed_apps={allowed_apps!r}\n"
            f"{traceback.format_exc()}"
        )
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
