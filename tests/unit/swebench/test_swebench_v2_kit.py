from __future__ import annotations

import asyncio
import inspect

import pytest

from gage_eval.agent_eval_kits.common import validate_benchmark_kit_entry
from gage_eval.agent_eval_kits.swebench.kit import load_kit
from gage_eval.agent_eval_kits.swebench.sub_workflows.framework_loop import build_workflow_bundle
from gage_eval.agent_eval_kits.swebench.tools import (
    build_swebench_instruction,
    build_swebench_messages,
    build_swebench_tools,
)
from gage_eval.agent_runtime.compiled_plan import SchedulerWorkflowBundle
from gage_eval.agent_runtime.contracts.failure import FailureEnvelopeError
from gage_eval.agent_runtime.schedulers.framework_loop import FrameworkLoopScheduler
from gage_eval.agent_runtime.schedulers.installed_client import InstalledClientScheduler
from gage_eval.agent_runtime.session import AgentRuntimeSession
from gage_eval.agent_runtime.tooling.registry import RuntimeToolRegistry
from gage_eval.agent_runtime.tooling.router import ToolRouter


def test_swebench_kit_declares_fresh_verifier_policy() -> None:
    kit = load_kit()

    assert kit.verifier_environment_policy == "fresh_from_profile"
    assert kit.verifier_environment_profile_id == "swebench_runtime"
    assert callable(kit.provider_config_resolver)
    assert validate_benchmark_kit_entry(kit) is kit


def test_swebench_provider_config_resolver_injects_sample_image_uri() -> None:
    resolver = load_kit().provider_config_resolver
    assert resolver is not None

    provider_config = resolver(
        sample={"metadata": {"environment_overrides": {"image_uri": "jefzda/sweap-images:sample-1"}}},
        base_provider_config={"workdir": "/workspace"},
        provider="docker",
        profile_id="swebench_runtime",
    )

    assert provider_config == {
        "workdir": "/workspace",
        "image": "jefzda/sweap-images:sample-1",
    }


def test_swebench_provider_config_resolver_rejects_legacy_metadata_image_uri() -> None:
    resolver = load_kit().provider_config_resolver
    assert resolver is not None

    with pytest.raises(ValueError, match="config.swebench.image_uri.missing"):
        resolver(
            sample={"metadata": {"image_uri": "legacy/image:should-not-be-used"}},
            base_provider_config={"workdir": "/workspace"},
            provider="docker",
            profile_id="swebench_runtime",
        )


def test_swebench_provider_config_resolver_rejects_missing_docker_image_uri() -> None:
    resolver = load_kit().provider_config_resolver
    assert resolver is not None

    with pytest.raises(ValueError, match="config.swebench.image_uri.missing"):
        resolver(
            sample={"metadata": {}},
            base_provider_config={"workdir": "/workspace"},
            provider="docker",
            profile_id="swebench_runtime",
        )


def test_swebench_verifier_resources_use_kit_owned_adapter() -> None:
    adapter = load_kit().resolve_verifier_resources()["adapter"]

    assert adapter.__class__.__module__ == "gage_eval.agent_eval_kits.swebench.judge.adapters"
    assert "role.judge.swebench_docker" not in inspect.getsource(adapter.__class__)


def test_swebench_prompt_and_tool_surface_match_official_workflow_contract() -> None:
    messages = build_swebench_messages({"instruction": "Fix the issue."})
    tools = build_swebench_tools({})
    tool_names = {tool["function"]["name"] for tool in tools}

    assert "WORKFLOW" in messages[0]["content"]
    assert "DO NOT modify test files" in messages[0]["content"]
    assert "MUST call submit_patch_tool before terminating" in messages[0]["content"]
    assert "DO NOT finish with a prose explanation" in messages[0]["content"]
    assert "/app/" in messages[0]["content"]
    assert "submit_patch_tool" in messages[0]["content"]
    assert {"run_shell", "str_replace_editor", "view_file_window", "find_in_repo", "find_file", "submit_patch_tool"}.issubset(tool_names)
    editor_schema = next(tool["function"]["parameters"] for tool in tools if tool["function"]["name"] == "str_replace_editor")
    assert "undo_edit" in editor_schema["properties"]["command"]["enum"]


def test_swebench_prompt_uses_runtime_working_dir() -> None:
    instruction = build_swebench_instruction({"instruction": "Fix the issue."}, working_dir="/repo")
    messages = build_swebench_messages({"instruction": "Fix the issue."}, working_dir="/repo")

    assert "starting with /repo/" in instruction
    assert "starting with /repo/" in messages[0]["content"]
    assert "starting with /app/" not in messages[0]["content"]


def test_swebench_framework_loop_loop_inputs_declares_required_tool() -> None:
    workflow = build_workflow_bundle()

    class _Lease:
        metadata = {"exec_workdir": "/repo"}

    loop_inputs = workflow.build_loop_inputs(
        session=_session("framework_loop"),
        sample={"instruction": "Fix the issue."},
        payload={"environment_lease": _Lease()},
    )

    assert loop_inputs["required_tool"] == "submit_patch_tool"
    assert "starting with /repo/" in loop_inputs["messages"][0]["content"]


def test_swebench_framework_loop_prepare_failure_uses_required_failure_code() -> None:
    session = _session("framework_loop")
    scheduler = FrameworkLoopScheduler(
        backend=object(),
        tool_router=ToolRouter(RuntimeToolRegistry()),
        tool_registry=RuntimeToolRegistry(),
    )
    workflow = SchedulerWorkflowBundle(
        bundle_id="swebench.framework_loop",
        benchmark_kit_id="swebench",
        scheduler_type="framework_loop",
        build_loop_inputs=lambda **_: (_ for _ in ()).throw(RuntimeError("prepare failed")),
    )

    with pytest.raises(FailureEnvelopeError) as excinfo:
        asyncio.run(
            scheduler.arun(
                session=session,
                sample={"id": "sample-1"},
                payload={},
                workflow_bundle=workflow,
                sandbox_provider=None,
            )
        )

    assert excinfo.value.failure.failure_code == "input_projection.workflow.prepare_failed"


def test_swebench_installed_client_prepare_failure_uses_required_failure_code() -> None:
    session = _session("installed_client")
    scheduler = InstalledClientScheduler(client=_NoopInstalledClient())
    workflow = SchedulerWorkflowBundle(
        bundle_id="swebench.installed_client",
        benchmark_kit_id="swebench",
        scheduler_type="installed_client",
        prepare_inputs=lambda **_: (_ for _ in ()).throw(RuntimeError("prepare failed")),
    )

    with pytest.raises(FailureEnvelopeError) as excinfo:
        asyncio.run(
            scheduler.arun(
                session=session,
                sample={"id": "sample-1"},
                payload={},
                workflow_bundle=workflow,
                sandbox_provider=None,
            )
        )

    assert excinfo.value.failure.failure_code == "input_projection.workflow.prepare_failed"


def _session(scheduler_type: str) -> AgentRuntimeSession:
    return AgentRuntimeSession(
        session_id="session-1",
        run_id="run-1",
        task_id="task-1",
        sample_id="sample-1",
        benchmark_kit_id="swebench",
        scheduler_type=scheduler_type,
        artifact_layout={"sample_root": "sample", "artifacts_dir": "sample/artifacts"},
    )


class _NoopInstalledClient:
    def invoke(self, payload):
        del payload
        return {}
