from __future__ import annotations

import json

from gage_eval.agent_eval_kits.common import resolve_sample_artifact_target
from gage_eval.agent_runtime.compiled_plan import SchedulerWorkflowBundle


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
    return {
        "instruction": session.prompt_context.get("instruction"),
        "allowed_apps": list(session.prompt_context.get("allowed_apps") or []),
        "ground_truth_mode": session.prompt_context.get("ground_truth_mode"),
        "mcp_endpoint": session.prompt_context.get("mcp_endpoint"),
        "env_endpoint": session.prompt_context.get("env_endpoint"),
        "metadata": dict(sample.get("metadata") or {}),
    }


def _prepare_environment(*, session, sample, sandbox_provider=None):
    return {"mcp_endpoint": session.prompt_context.get("mcp_endpoint")}


def _capture_environment_artifacts(*, runtime_entry, session, sample, scheduler_output, sandbox_provider=None):
    saved = runtime_entry.save(sample=sample, sandbox_provider=sandbox_provider)
    artifact_paths = dict(scheduler_output.get("artifact_paths") or {})
    if saved:
        artifact_paths["appworld_save"] = _persist_appworld_save_artifact(session, saved)
    return artifact_paths


def _finalize_result(*, session, sample, scheduler_output, artifact_paths):
    output = dict(scheduler_output or {})
    output.setdefault("artifact_paths", dict(artifact_paths or {}))
    return output


def _persist_appworld_save_artifact(session, saved: dict) -> str:
    """Persists the AppWorld save payload under the sample root."""

    target, relative_path = resolve_sample_artifact_target(session, "appworld_save.json")
    target.write_text(json.dumps(saved, ensure_ascii=False, indent=2), encoding="utf-8")
    return relative_path
