from __future__ import annotations

import asyncio
import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from gage_eval.agent_eval_kits.appworld.sub_workflows.framework_loop import (
    build_workflow_bundle as build_appworld_framework_loop_bundle,
)
from gage_eval.agent_eval_kits.appworld.sub_workflows.installed_client import (
    build_workflow_bundle as build_appworld_installed_client_bundle,
)
from gage_eval.agent_eval_kits.appworld.runtime import AppWorldRuntime
from gage_eval.agent_runtime import build_compiled_runtime_executor, compile_agent_runtime_plan
from gage_eval.agent_runtime.resources.contracts import ResourceLease
from gage_eval.agent_runtime.resources.manager import RuntimeLeaseBinding
from gage_eval.agent_runtime.verifier.contracts import RuntimeJudgeOutcome, VerifierInput, VerifierResult
from gage_eval.role.agent.backends.demo_agent import RuntimeSmokeAgent


class _InstalledClientStub:
    def __init__(self, *, answer: str = "done", patch_content: str | None = None) -> None:
        self._answer = answer
        self._patch_content = patch_content

    def invoke(self, payload: dict[str, Any]) -> dict[str, Any]:
        result = {
            "answer": self._answer,
            "agent_trace": [],
            "artifact_paths": {"stdout": "stdout.txt"},
        }
        if self._patch_content is not None:
            result["patch_content"] = self._patch_content
        return result


class _SwebenchFrameworkAgent:
    def run(self, payload: dict[str, Any]) -> dict[str, Any]:
        return {
            "answer": "diff --git a/foo b/foo\n",
            "patch_content": "diff --git a/foo b/foo\n",
            "agent_trace": [
                {
                    "trace_step": 1,
                    "trace_role": "tool",
                    "name": "submit_patch_tool",
                    "output": {"stdout": "diff --git a/foo b/foo\n"},
                }
            ],
        }

    def invoke(self, payload: dict[str, Any]) -> dict[str, Any]:
        return self.run(payload)


class _PassingVerifierRunner:
    def run(
        self,
        *,
        plan,
        session,
        sample,
        scheduler_result,
        sandbox_provider,
    ) -> RuntimeJudgeOutcome:
        verifier_input = VerifierInput(
            benchmark_kit_id=session.benchmark_kit_id,
            scheduler_type=session.scheduler_type,
            sample_id=session.sample_id,
            sample=sample,
            scheduler_result=scheduler_result.to_dict(),
            runtime_context=dict(session.runtime_context or {}),
            verifier_resources={"adapter_type": "test_stub"},
        )
        verifier_result = VerifierResult(
            status="completed",
            payload={
                "status": "completed",
                "resolved": True,
                "score": 1.0,
                "summary": "runtime smoke completed",
                "judge_source": "runtime_smoke_stub",
            },
        )
        judge_output = {
            "status": "completed",
            "resolved": True,
            "score": 1.0,
            "summary": "runtime smoke completed",
            "artifact_paths": dict(scheduler_result.artifact_paths or {}),
            "runtime_handle": dict(scheduler_result.runtime_state or {}),
            "judge_source": "runtime_smoke_stub",
        }
        return RuntimeJudgeOutcome(
            verifier_input=verifier_input,
            verifier_result=verifier_result,
            judge_output=judge_output,
            persisted_path=session.artifact_layout["verifier_result"],
        )

    def build_failed_outcome(self, *, plan, session, sample, failure):
        raise AssertionError(f"Unexpected failure path during runtime smoke: {failure}")


class _StaticProvider:
    def __init__(self, *, runtime_handle: dict[str, Any] | None = None, sandbox: Any = None) -> None:
        self._handle = SimpleNamespace(
            runtime_handle=dict(runtime_handle or {}),
            sandbox=sandbox,
        )

    def get_handle(self):
        return self._handle


class _StaticResourceManager:
    def __init__(
        self,
        *,
        resource_kind: str,
        profile_id: str,
        sandbox_provider: _StaticProvider | None = None,
    ) -> None:
        self._resource_kind = resource_kind
        self._profile_id = profile_id
        self._sandbox_provider = sandbox_provider

    def acquire(self, session, *, resource_plan, trace=None, sample=None):
        handle = self._sandbox_provider.get_handle() if self._sandbox_provider is not None else None
        runtime_handle = dict(getattr(handle, "runtime_handle", {}) or {})
        resource_lease = ResourceLease(
            lease_id=f"lease-{session.sample_id}",
            resource_kind=self._resource_kind,  # type: ignore[arg-type]
            profile_id=self._profile_id,
            lifecycle="per_sample",
            endpoints={key: str(value) for key, value in runtime_handle.items() if "endpoint" in key},
            handle_ref=runtime_handle,
        )
        return RuntimeLeaseBinding(
            resource_lease=resource_lease,
            sandbox_provider=self._sandbox_provider,
            sandbox_handle=handle,
        )

    def release(self, binding) -> None:
        return None


class _FakeTau2Runtime:
    def __init__(self) -> None:
        self._agent_cost = 0.1

    def initialize_task(self, sample: dict[str, Any]) -> dict[str, Any]:
        tau2_meta = sample.setdefault("metadata", {}).setdefault("tau2", {})
        tau2_meta.setdefault("policy", "policy")
        tau2_meta.setdefault("agent_instruction", "Help the user.")
        tau2_meta.setdefault("gage_instruction", "Use the provided tools.")
        tools_schema = [
            {
                "type": "function",
                "function": {
                    "name": "respond",
                    "description": "Reply to the user.",
                    "parameters": {"type": "object", "properties": {"message": {"type": "string"}}},
                },
            }
        ]
        sample["messages"] = [{"role": "user", "content": "Need support"}]
        sample["tools"] = tools_schema
        return {
            "messages": sample["messages"],
            "tools_schema": tools_schema,
            "metadata": sample["metadata"],
        }

    def record_agent_usage(self, usage: dict[str, Any] | None) -> None:
        if not isinstance(usage, dict):
            return
        total_tokens = usage.get("total_tokens")
        if total_tokens is not None:
            self._agent_cost = float(total_tokens)

    def get_state(self) -> dict[str, Any]:
        return {
            "domain": "telecom",
            "messages": [{"role": "assistant", "content": "done"}],
            "termination_reason": "agent_stop",
            "agent_cost": self._agent_cost,
            "user_cost": 0.0,
        }


def _build_sample(agent_runtime_id: str) -> dict[str, Any]:
    sample_id = agent_runtime_id.replace("_", "-")
    if agent_runtime_id.startswith("terminal_bench"):
        return {
            "id": sample_id,
            "instruction": "say done",
            "expected_answer": "done",
            "messages": [{"role": "user", "content": "say done"}],
        }
    if agent_runtime_id.startswith("swebench"):
        return {
            "id": sample_id,
            "instruction": "fix the failing test",
            "messages": [{"role": "user", "content": "fix the failing test"}],
            "metadata": {
                "repo": "example/repo",
                "base_commit": "abc123",
                "test_command": "pytest -q",
            },
        }
    if agent_runtime_id.startswith("appworld"):
        return {
            "id": sample_id,
            "instruction": "create a note",
            "messages": [{"role": "user", "content": "create a note"}],
            "metadata": {
                "appworld": {
                    "task_id": "task-1",
                    "allowed_apps": ["notes"],
                    "ground_truth_mode": "minimal",
                }
            },
        }
    if agent_runtime_id.startswith("tau2"):
        return {
            "id": sample_id,
            "instruction": "help the telecom user",
            "metadata": {
                "tau2": {
                    "domain": "telecom",
                    "trial": 0,
                    "seed": 1,
                }
            },
        }
    raise AssertionError(f"Unhandled runtime id: {agent_runtime_id}")


def _build_executor(agent_runtime_id: str):
    plan = compile_agent_runtime_plan(agent_runtime_id=agent_runtime_id)
    sample = _build_sample(agent_runtime_id)
    lifecycle_calls: list[str] = []
    resource_kind = "local_process" if agent_runtime_id.startswith("tau2") else "docker"
    sandbox_provider: _StaticProvider | None = None

    if agent_runtime_id.startswith("appworld"):
        def requester(endpoint: str, method: str, payload: dict[str, Any], timeout_s: int) -> dict[str, Any]:
            lifecycle_calls.append(method)
            return {"output": {"method": method, "task_id": payload.get("task_id")}}

        runtime_entry = AppWorldRuntime(requester=requester)
        workflow_bundle = (
            build_appworld_installed_client_bundle(runtime_entry)
            if "installed_client" in agent_runtime_id
            else build_appworld_framework_loop_bundle(runtime_entry)
        )
        plan = replace(
            plan,
            kit_runtime_ref=runtime_entry,
            workflow_bundle=workflow_bundle,
        )
        sandbox_provider = _StaticProvider(
            runtime_handle={
                "env_endpoint": "http://env",
                "apis_endpoint": "http://apis",
                "mcp_endpoint": "http://mcp",
            }
        )
    elif agent_runtime_id.startswith("tau2"):
        sandbox_provider = _StaticProvider(runtime_handle={}, sandbox=_FakeTau2Runtime())

    plan = replace(
        plan,
        resource_plan={
            "resource_kind": resource_kind,
            "sandbox_config": {} if sandbox_provider is not None else None,
        },
    )
    if agent_runtime_id == "swebench_framework_loop":
        backend = _SwebenchFrameworkAgent()
    elif agent_runtime_id == "swebench_installed_client":
        backend = _InstalledClientStub(
            answer="diff --git a/foo b/foo\n",
            patch_content="diff --git a/foo b/foo\n",
        )
    elif "installed_client" in agent_runtime_id:
        backend = _InstalledClientStub()
    else:
        backend = RuntimeSmokeAgent()
    executor = build_compiled_runtime_executor(
        compiled_plan=plan,
        agent_backend=backend,
        installed_client_override=backend if "installed_client" in agent_runtime_id else None,
        max_turns=4,
    )
    executor.resource_manager = _StaticResourceManager(
        resource_kind=resource_kind,
        profile_id=plan.runtime_spec.sandbox_profile_id or plan.runtime_spec.benchmark_kit_id,
        sandbox_provider=sandbox_provider,
    )
    executor.verifier_runner = _PassingVerifierRunner()
    return executor, sample, lifecycle_calls


@pytest.mark.fast
@pytest.mark.parametrize(
    ("agent_runtime_id", "expected_artifact"),
    [
        ("terminal_bench_installed_client", "artifacts/tool_trace.json"),
        ("terminal_bench_framework_loop", "artifacts/tool_trace.json"),
        ("swebench_installed_client", None),
        ("swebench_framework_loop", None),
        ("appworld_installed_client", "artifacts/appworld_save.json"),
        ("appworld_framework_loop", "artifacts/appworld_save.json"),
        ("tau2_installed_client", "artifacts/tau2_state.json"),
        ("tau2_framework_loop", "artifacts/tau2_state.json"),
    ],
)
def test_phase1_runtime_flow_matrix_smoke(
    tmp_path: Path,
    monkeypatch,
    agent_runtime_id: str,
    expected_artifact: str | None,
) -> None:
    monkeypatch.setenv("GAGE_EVAL_SAVE_DIR", str(tmp_path))
    executor, sample, lifecycle_calls = _build_executor(agent_runtime_id)

    result = asyncio.run(
        executor.aexecute(
            sample=sample,
            payload={
                "sample": sample,
                "execution_context": {
                    "run_id": "runtime-flow-matrix",
                    "task_id": agent_runtime_id,
                    "sample_id": sample["id"],
                },
            },
        )
    )

    if agent_runtime_id.startswith("swebench"):
        assert "diff --git" in result["answer"]
    else:
        assert result["answer"] == "done"
    assert result["runtime_judge_outcome"]["judge_output"]["resolved"] is True

    runtime_session = result["runtime_session"]
    runtime_metadata = json.loads(Path(runtime_session["runtime_metadata_path"]).read_text(encoding="utf-8"))
    verifier_payload = json.loads(Path(runtime_session["verifier_result_path"]).read_text(encoding="utf-8"))

    assert runtime_metadata["session_id"] == runtime_session["session_id"]
    assert runtime_metadata["sample_id"] == sample["id"]
    assert runtime_metadata["resource_lease"]["resource_kind"] in {"docker", "local_process"}
    assert verifier_payload["verifier_input"]["sample_id"] == sample["id"]
    assert verifier_payload["judge_output"]["resolved"] is True

    if expected_artifact is not None:
        assert (Path(runtime_session["sample_root"]) / expected_artifact).exists()
    if agent_runtime_id.startswith("terminal_bench"):
        artifact_paths = verifier_payload["judge_output"]["artifact_paths"]
        assert artifact_paths["tool_trace"] == "artifacts/tool_trace.json"
        assert artifact_paths["stdout"] == "artifacts/stdout.log"
        assert artifact_paths["stderr"] == "artifacts/stderr.log"
        assert artifact_paths["workspace_diff"] == "artifacts/workspace_diff.json"
        if agent_runtime_id.startswith("swebench"):
            assert verifier_payload["judge_output"]["artifact_paths"]["submission_patch"] == "artifacts/submission.patch"
    if agent_runtime_id.startswith("appworld"):
        assert lifecycle_calls == ["initialize", "save"]
