from __future__ import annotations

from gage_eval.agent_eval_kits.terminal_bench.artifacts import persist_terminal_artifacts
from gage_eval.agent_runtime.compiled_plan import SchedulerWorkflowBundle
from gage_eval.agent_eval_kits.common import extract_instruction


def build_workflow_bundle() -> SchedulerWorkflowBundle:
    """Build the installed-client workflow for terminal benchmark."""

    return SchedulerWorkflowBundle(
        bundle_id="terminal_bench.installed_client",
        benchmark_kit_id="terminal_bench",
        scheduler_type="installed_client",
        prepare_inputs=_prepare_inputs,
        prepare_environment=_prepare_environment,
        capture_environment_artifacts=_capture_environment_artifacts,
        finalize_result=_finalize_result,
        failure_normalizer=lambda **_: {},
    )


def _prepare_inputs(*, session, sample, payload, sandbox_provider=None):
    return {
        "instruction": extract_instruction(sample),
        "cwd": session.runtime_context.get("cwd") or "/workspace",
        "env": dict(session.runtime_context.get("env") or {}),
        "metadata": dict(sample.get("metadata") or {}),
    }


def _prepare_environment(*, session, sample, sandbox_provider=None):
    return {"cwd": session.runtime_context.get("cwd") or "/workspace"}


def _capture_environment_artifacts(*, session, sample, scheduler_output, sandbox_provider=None):
    return persist_terminal_artifacts(
        session=session,
        scheduler_output=scheduler_output,
        sandbox_provider=sandbox_provider,
    )


def _finalize_result(*, session, sample, scheduler_output, artifact_paths):
    output = dict(scheduler_output or {})
    output.setdefault("artifact_paths", dict(artifact_paths or {}))
    output.setdefault("answer", output.get("answer") or "")
    return output
