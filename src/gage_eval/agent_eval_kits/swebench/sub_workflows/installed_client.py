from __future__ import annotations

from gage_eval.agent_runtime.compiled_plan import SchedulerWorkflowBundle
from gage_eval.agent_eval_kits.swebench.artifacts import persist_swebench_artifacts
from gage_eval.agent_eval_kits.swebench.units import build_swebench_instruction


def build_workflow_bundle() -> SchedulerWorkflowBundle:
    """Build the installed-client SWE-bench workflow."""

    return SchedulerWorkflowBundle(
        bundle_id="swebench.installed_client",
        benchmark_kit_id="swebench",
        scheduler_type="installed_client",
        prepare_inputs=_prepare_inputs,
        prepare_environment=_prepare_environment,
        capture_environment_artifacts=_capture_environment_artifacts,
        finalize_result=_finalize_result,
        failure_normalizer=lambda **_: {},
    )


def _prepare_inputs(*, session, sample, payload, sandbox_provider=None):
    return {
        "instruction": build_swebench_instruction(sample),
        "submission_contract": "submission.patch",
        "required_artifacts": ["submission.patch"],
        "patch_submission_instructions": (
            "Persist the final repository diff to /workspace/submission.patch before returning."
        ),
        "repo": session.runtime_context.get("repo"),
        "base_commit": session.runtime_context.get("base_commit"),
        "test_command": session.runtime_context.get("test_command"),
        "metadata": dict(sample.get("metadata") or {}),
    }


def _prepare_environment(*, session, sample, sandbox_provider=None):
    return {"workspace": "/workspace"}


def _capture_environment_artifacts(*, session, sample, scheduler_output, sandbox_provider=None):
    return persist_swebench_artifacts(
        session=session,
        scheduler_output=scheduler_output,
        sandbox_provider=sandbox_provider,
    )


def _finalize_result(*, session, sample, scheduler_output, artifact_paths):
    output = dict(scheduler_output or {})
    output["artifact_paths"] = dict(artifact_paths or {})
    return output
