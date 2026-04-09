from __future__ import annotations

import json

from gage_eval.agent_eval_kits.common import resolve_sample_artifact_target
from gage_eval.agent_runtime.compiled_plan import SchedulerWorkflowBundle


def build_workflow_bundle() -> SchedulerWorkflowBundle:
    """Build the installed-client Tau2 workflow."""

    return SchedulerWorkflowBundle(
        bundle_id="tau2.installed_client",
        benchmark_kit_id="tau2",
        scheduler_type="installed_client",
        prepare_inputs=_prepare_inputs,
        prepare_environment=_prepare_environment,
        capture_environment_artifacts=_capture_environment_artifacts,
        finalize_result=_finalize_result,
        failure_normalizer=lambda **_: {},
    )


def _prepare_inputs(*, session, sample, payload, sandbox_provider=None):
    return {
        "messages": list(sample.get("messages") or []),
        "policy": session.prompt_context.get("policy"),
        "domain": session.prompt_context.get("domain"),
        "tools_schema": list(session.prompt_context.get("tools_schema") or []),
        "metadata": dict(sample.get("metadata") or {}),
    }


def _prepare_environment(*, session, sample, sandbox_provider=None):
    return {"resource_kind": "local_process"}


def _capture_environment_artifacts(*, session, sample, scheduler_output, sandbox_provider=None):
    artifact_paths = dict(scheduler_output.get("artifact_paths") or {})
    state = _capture_tau2_state(sandbox_provider)
    if state:
        artifact_paths["tau2_state"] = _persist_tau2_state_artifact(session, state)
    return artifact_paths


def _finalize_result(*, session, sample, scheduler_output, artifact_paths):
    output = dict(scheduler_output or {})
    output.setdefault("artifact_paths", dict(artifact_paths or {}))
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
