from __future__ import annotations

import hashlib
import importlib
import json
from dataclasses import replace
from typing import Any

from gage_eval.agent_eval_kits import load_benchmark_kit
from gage_eval.agent_runtime.artifacts import RuntimeArtifactSink, RuntimeTraceEmitter
from gage_eval.agent_runtime.clients import resolve_installed_client
from gage_eval.agent_runtime.compiled_plan import CompiledRuntimePlan, RuntimeCompileError
from gage_eval.agent_runtime.executor import (
    AgentRuntimeSessionFactory,
    CompiledRuntimeExecutor,
    DefaultVerifierRunner,
)
from gage_eval.agent_runtime.failure_mapper import FailureMapper
from gage_eval.agent_runtime.resources.manager import RuntimeResourceManager
from gage_eval.agent_runtime.schedulers.framework_loop import FrameworkLoopScheduler
from gage_eval.agent_runtime.schedulers.installed_client import InstalledClientScheduler
from gage_eval.agent_runtime.spec import AgentRuntimeSpec
from gage_eval.agent_runtime.verifier.binding import JudgeBinding
from gage_eval.role.agent.human_gateway import build_default_human_gateway
from gage_eval.role.agent.tool_router import ToolRouter
from gage_eval.sandbox.manager import SandboxManager


_BUILTIN_RUNTIME_SPECS: dict[str, AgentRuntimeSpec] = {
    "terminal_bench_installed_client": AgentRuntimeSpec(
        agent_runtime_id="terminal_bench_installed_client",
        benchmark_kit_id="terminal_bench",
        scheduler_type="installed_client",
        client_id="codex",
        sandbox_profile_id="terminal_bench_runtime",
        resource_policy={"resource_kind": "docker", "lifecycle": "per_sample"},
        verifier_binding_id="terminal_bench_native",
    ),
    "terminal_bench_framework_loop": AgentRuntimeSpec(
        agent_runtime_id="terminal_bench_framework_loop",
        benchmark_kit_id="terminal_bench",
        scheduler_type="framework_loop",
        sandbox_profile_id="terminal_bench_runtime",
        resource_policy={"resource_kind": "docker", "lifecycle": "per_sample"},
        verifier_binding_id="terminal_bench_native",
    ),
    "swebench_installed_client": AgentRuntimeSpec(
        agent_runtime_id="swebench_installed_client",
        benchmark_kit_id="swebench",
        scheduler_type="installed_client",
        client_id="codex",
        sandbox_profile_id="swebench_runtime",
        resource_policy={"resource_kind": "docker", "lifecycle": "per_sample"},
        verifier_binding_id="swebench_verifier",
    ),
    "swebench_framework_loop": AgentRuntimeSpec(
        agent_runtime_id="swebench_framework_loop",
        benchmark_kit_id="swebench",
        scheduler_type="framework_loop",
        sandbox_profile_id="swebench_runtime",
        resource_policy={"resource_kind": "docker", "lifecycle": "per_sample"},
        verifier_binding_id="swebench_verifier",
    ),
    "skillsbench_installed_client": AgentRuntimeSpec(
        agent_runtime_id="skillsbench_installed_client",
        benchmark_kit_id="skillsbench",
        scheduler_type="installed_client",
        client_id="codex",
        sandbox_profile_id="swebench_runtime",
        resource_policy={"resource_kind": "docker", "lifecycle": "per_sample"},
        verifier_binding_id="swebench_verifier",
    ),
    "skillsbench_framework_loop": AgentRuntimeSpec(
        agent_runtime_id="skillsbench_framework_loop",
        benchmark_kit_id="skillsbench",
        scheduler_type="framework_loop",
        sandbox_profile_id="swebench_runtime",
        resource_policy={"resource_kind": "docker", "lifecycle": "per_sample"},
        verifier_binding_id="swebench_verifier",
    ),
    "appworld_installed_client": AgentRuntimeSpec(
        agent_runtime_id="appworld_installed_client",
        benchmark_kit_id="appworld",
        scheduler_type="installed_client",
        client_id="codex",
        sandbox_profile_id="appworld_local",
        resource_policy={"resource_kind": "docker", "lifecycle": "per_sample"},
        verifier_binding_id="appworld_verifier",
    ),
    "appworld_framework_loop": AgentRuntimeSpec(
        agent_runtime_id="appworld_framework_loop",
        benchmark_kit_id="appworld",
        scheduler_type="framework_loop",
        sandbox_profile_id="appworld_local",
        resource_policy={"resource_kind": "docker", "lifecycle": "per_sample"},
        verifier_binding_id="appworld_verifier",
    ),
    "tau2_installed_client": AgentRuntimeSpec(
        agent_runtime_id="tau2_installed_client",
        benchmark_kit_id="tau2",
        scheduler_type="installed_client",
        client_id="codex",
        sandbox_profile_id="tau2_local",
        resource_policy={"resource_kind": "local_process", "lifecycle": "per_sample"},
        verifier_binding_id="tau2_verifier",
    ),
    "tau2_framework_loop": AgentRuntimeSpec(
        agent_runtime_id="tau2_framework_loop",
        benchmark_kit_id="tau2",
        scheduler_type="framework_loop",
        sandbox_profile_id="tau2_local",
        resource_policy={"resource_kind": "local_process", "lifecycle": "per_sample"},
        verifier_binding_id="tau2_verifier",
    ),
}


def resolve_agent_runtime_spec(
    agent_runtime_id: str,
    runtime_specs: dict[str, AgentRuntimeSpec] | None = None,
) -> AgentRuntimeSpec:
    """Resolve the declared runtime spec from the builtin runtime catalog."""

    if runtime_specs and agent_runtime_id in runtime_specs:
        return runtime_specs[agent_runtime_id]
    if agent_runtime_id not in _BUILTIN_RUNTIME_SPECS:
        raise KeyError(f"Unknown agent runtime '{agent_runtime_id}'")
    spec = _BUILTIN_RUNTIME_SPECS[agent_runtime_id]
    if spec.benchmark_kit_id == "skillsbench":
        return replace(spec, benchmark_kit_id="swebench")
    return spec


def compile_agent_runtime_plan(
    *,
    agent_runtime_id: str,
    sandbox_config: dict[str, Any] | None = None,
    runtime_specs: dict[str, AgentRuntimeSpec] | None = None,
    benchmark_config: dict[str, Any] | None = None,
) -> CompiledRuntimePlan:
    """Compile a runtime plan from the builtin runtime catalog and benchmark kit."""

    # STEP 1: Resolve the runtime spec and benchmark kit.
    runtime_spec = resolve_agent_runtime_spec(agent_runtime_id, runtime_specs=runtime_specs)
    benchmark_kit = load_benchmark_kit(runtime_spec.benchmark_kit_id)
    diagnostics: list[dict[str, Any]] = []
    if runtime_spec.scheduler_type == "installed_client" and not runtime_spec.client_id:
        diagnostics.append(
            {
                "severity": "error",
                "code": "installed_client_missing_client_id",
                "agent_runtime_id": runtime_spec.agent_runtime_id,
            }
        )
        raise RuntimeCompileError(
            f"Installed-client runtime '{runtime_spec.agent_runtime_id}' requires client_id",
            diagnostics=diagnostics,
        )

    # STEP 2: Resolve the scheduler-local workflow and verifier resources.
    workflow_bundle = benchmark_kit.resolve_workflow_bundle(runtime_spec.scheduler_type)
    verifier_resources = benchmark_kit.resolve_verifier_resources()
    if not verifier_resources.get("adapter"):
        diagnostics.append(
            {
                "severity": "error",
                "code": "verifier_resources_missing",
                "benchmark_kit_id": runtime_spec.benchmark_kit_id,
            }
        )
        raise RuntimeCompileError(
            f"Runtime verifier adapter is missing for benchmark kit "
            f"'{runtime_spec.benchmark_kit_id}'",
            diagnostics=diagnostics,
        )
    judge_binding = JudgeBinding(
        judge_mode="runtime_verifier",
        benchmark_kit_id=runtime_spec.benchmark_kit_id,
        verifier_kind=benchmark_kit.verifier_kind,  # type: ignore[arg-type]
        verifier_resource_refs=verifier_resources,
        failure_policy="bind_failure",
    )

    # STEP 3: Resolve resource plan and artifact policy.
    kit_module = importlib.import_module(
        f"gage_eval.agent_eval_kits.{runtime_spec.benchmark_kit_id}.kit"
    )
    resource_plan = kit_module.build_resource_plan(runtime_spec, sandbox_config)
    artifact_policy = {
        "write_runtime_metadata": True,
        "write_verifier_result": True,
        "sample_root_namespace": "runtime",
    }

    # STEP 4: Freeze the cold-path payload into a stable cache key.
    cache_key_payload = {
        "agent_runtime_id": runtime_spec.agent_runtime_id,
        "benchmark_kit_id": runtime_spec.benchmark_kit_id,
        "scheduler_type": runtime_spec.scheduler_type,
        "resource_plan": resource_plan,
        "artifact_policy": artifact_policy,
        "benchmark_config": benchmark_config or {},
    }
    cache_key = hashlib.md5(
        json.dumps(cache_key_payload, sort_keys=True, default=str).encode("utf-8")
    ).hexdigest()
    return CompiledRuntimePlan(
        plan_id=f"compiled-{cache_key[:12]}",
        runtime_spec=runtime_spec,
        scheduler_handle=runtime_spec.scheduler_type,
        workflow_bundle=workflow_bundle,
        kit_runtime_ref=benchmark_kit.runtime_entry,
        judge_binding=judge_binding,
        resource_plan=resource_plan,
        artifact_policy=artifact_policy,
        cache_key=cache_key,
        benchmark_config=dict(benchmark_config or {}),
        compile_diagnostics=diagnostics,
    )


def build_compiled_runtime_executor(
    *,
    compiled_plan: CompiledRuntimePlan,
    agent_backend: Any,
    installed_client_override: Any = None,
    prompt_renderer=None,
    max_turns: int = 8,
    tool_call_retry_budget: int = 3,
    max_total_invalid_tool_calls: int = 20,
    pre_hooks=None,
    post_hooks=None,
    mcp_clients: dict[str, Any] | None = None,
    sandbox_manager: SandboxManager | None = None,
) -> CompiledRuntimeExecutor:
    """Build the hot-path executor for one compiled runtime plan."""

    sandbox_manager = sandbox_manager or SandboxManager()
    if compiled_plan.runtime_spec.scheduler_type == "installed_client":
        scheduler_handle = InstalledClientScheduler(
            resolve_installed_client(
                client_id=compiled_plan.runtime_spec.client_id,
                client_override=installed_client_override,
            )
        )
    elif compiled_plan.runtime_spec.scheduler_type == "framework_loop":
        scheduler_handle = FrameworkLoopScheduler(
            backend=agent_backend,
            tool_router=ToolRouter(
                mcp_clients=mcp_clients,
                human_gateway=build_default_human_gateway(),
            ),
            prompt_renderer=prompt_renderer,
            max_turns=max_turns,
            tool_call_retry_budget=tool_call_retry_budget,
            max_total_invalid_tool_calls=max_total_invalid_tool_calls,
            pre_hooks=pre_hooks,
            post_hooks=post_hooks,
        )
    else:
        raise KeyError(f"Unsupported scheduler '{compiled_plan.runtime_spec.scheduler_type}'")

    resolved_plan = replace(compiled_plan, scheduler_handle=scheduler_handle)
    artifact_sink = RuntimeArtifactSink()
    return CompiledRuntimeExecutor(
        compiled_plan=resolved_plan,
        resource_manager=RuntimeResourceManager(sandbox_manager),
        session_factory=AgentRuntimeSessionFactory(artifact_sink),
        verifier_runner=DefaultVerifierRunner(),
        artifact_sink=artifact_sink,
        trace_emitter=RuntimeTraceEmitter(),
        failure_mapper=FailureMapper(),
    )
