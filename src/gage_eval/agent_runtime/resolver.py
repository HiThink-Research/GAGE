from __future__ import annotations

import hashlib
import importlib
import json
from dataclasses import replace
from typing import Any

from gage_eval.agent_eval_kits import load_benchmark_kit
from gage_eval.agent_runtime.artifacts import RuntimeArtifactSink, RuntimeTraceEmitter
from gage_eval.agent_runtime.clients import resolve_installed_client
from gage_eval.agent_runtime.compiled_plan import (
    CompiledRuntimePlan,
    RuntimeCompileError,
    SchedulerWorkflowBundle,
)
from gage_eval.agent_runtime.executor import (
    AgentRuntimeSessionFactory,
    CompiledRuntimeExecutor,
    DefaultVerifierRunner,
)
from gage_eval.agent_runtime.failure_mapper import FailureMapper
from gage_eval.agent_runtime.resources.manager import RuntimeResourceManager
from gage_eval.agent_runtime.schedulers.acp_client import AcpClientScheduler
from gage_eval.agent_runtime.schedulers.framework_loop import FrameworkLoopScheduler, StaticModelBackendAdapter
from gage_eval.agent_runtime.schedulers.installed_client import InstalledClientScheduler
from gage_eval.agent_runtime.spec import AgentRuntimeSpec
from gage_eval.agent_runtime.tooling.mcp.client import McpServerProcess
from gage_eval.agent_runtime.tooling.mcp.discovery import discover_mcp_tools
from gage_eval.agent_runtime.tooling.registry import RuntimeToolRegistry
from gage_eval.agent_runtime.tooling.router import ToolRouter
from gage_eval.agent_runtime.tooling.skills.policy import SkillPolicy
from gage_eval.agent_runtime.tooling.skills.resolver import SkillManifestResolver
from gage_eval.agent_runtime.verifier.adapters import NativeVerifierAdapter
from gage_eval.agent_runtime.verifier.binding import JudgeBinding


_BUILTIN_RUNTIME_SPECS: dict[str, AgentRuntimeSpec] = {
    "terminal_bench_installed_client": AgentRuntimeSpec(
        agent_runtime_id="terminal_bench_installed_client",
        benchmark_kit_id="terminal_bench",
        scheduler_type="installed_client",
        client_id="codex",
        environment_profile_id="terminal_bench_runtime",
        resource_policy={"resource_kind": "docker", "lifecycle": "per_sample"},
        verifier_binding_id="terminal_bench_native",
    ),
    "terminal_bench_framework_loop": AgentRuntimeSpec(
        agent_runtime_id="terminal_bench_framework_loop",
        benchmark_kit_id="terminal_bench",
        scheduler_type="framework_loop",
        environment_profile_id="terminal_bench_runtime",
        resource_policy={"resource_kind": "docker", "lifecycle": "per_sample"},
        verifier_binding_id="terminal_bench_native",
    ),
    "terminal_bench_acp_client": AgentRuntimeSpec(
        agent_runtime_id="terminal_bench_acp_client",
        benchmark_kit_id="terminal_bench",
        scheduler_type="acp_client",
        client_id="acp",
        environment_profile_id="terminal_bench_runtime",
        resource_policy={"resource_kind": "docker", "lifecycle": "per_sample"},
        verifier_binding_id="terminal_bench_native",
    ),
    "swebench_installed_client": AgentRuntimeSpec(
        agent_runtime_id="swebench_installed_client",
        benchmark_kit_id="swebench",
        scheduler_type="installed_client",
        client_id="codex",
        environment_profile_id="swebench_runtime",
        resource_policy={"resource_kind": "docker", "lifecycle": "per_sample"},
        verifier_binding_id="swebench_verifier",
    ),
    "swebench_framework_loop": AgentRuntimeSpec(
        agent_runtime_id="swebench_framework_loop",
        benchmark_kit_id="swebench",
        scheduler_type="framework_loop",
        environment_profile_id="swebench_runtime",
        resource_policy={"resource_kind": "docker", "lifecycle": "per_sample"},
        verifier_binding_id="swebench_verifier",
    ),
    "swebench_acp_client": AgentRuntimeSpec(
        agent_runtime_id="swebench_acp_client",
        benchmark_kit_id="swebench",
        scheduler_type="acp_client",
        client_id="acp",
        environment_profile_id="swebench_runtime",
        resource_policy={"resource_kind": "docker", "lifecycle": "per_sample"},
        verifier_binding_id="swebench_verifier",
    ),
    "skillsbench_installed_client": AgentRuntimeSpec(
        agent_runtime_id="skillsbench_installed_client",
        benchmark_kit_id="skillsbench",
        scheduler_type="installed_client",
        client_id="codex",
        environment_profile_id="swebench_runtime",
        resource_policy={"resource_kind": "docker", "lifecycle": "per_sample"},
        verifier_binding_id="swebench_verifier",
    ),
    "skillsbench_framework_loop": AgentRuntimeSpec(
        agent_runtime_id="skillsbench_framework_loop",
        benchmark_kit_id="skillsbench",
        scheduler_type="framework_loop",
        environment_profile_id="swebench_runtime",
        resource_policy={"resource_kind": "docker", "lifecycle": "per_sample"},
        verifier_binding_id="swebench_verifier",
    ),
    "appworld_installed_client": AgentRuntimeSpec(
        agent_runtime_id="appworld_installed_client",
        benchmark_kit_id="appworld",
        scheduler_type="installed_client",
        client_id="codex",
        environment_profile_id="appworld_local",
        resource_policy={"resource_kind": "docker", "lifecycle": "per_sample"},
        verifier_binding_id="appworld_verifier",
    ),
    "appworld_framework_loop": AgentRuntimeSpec(
        agent_runtime_id="appworld_framework_loop",
        benchmark_kit_id="appworld",
        scheduler_type="framework_loop",
        environment_profile_id="appworld_local",
        resource_policy={"resource_kind": "docker", "lifecycle": "per_sample"},
        verifier_binding_id="appworld_verifier",
    ),
    "appworld_acp_client": AgentRuntimeSpec(
        agent_runtime_id="appworld_acp_client",
        benchmark_kit_id="appworld",
        scheduler_type="acp_client",
        client_id="acp",
        environment_profile_id="appworld_local",
        resource_policy={"resource_kind": "docker", "lifecycle": "per_sample"},
        verifier_binding_id="appworld_verifier",
    ),
    "tau2_installed_client": AgentRuntimeSpec(
        agent_runtime_id="tau2_installed_client",
        benchmark_kit_id="tau2",
        scheduler_type="installed_client",
        client_id="codex",
        environment_profile_id="tau2-local-process",
        resource_policy={"resource_kind": "local_process", "lifecycle": "per_sample"},
        verifier_binding_id="tau2_verifier",
    ),
    "tau2_framework_loop": AgentRuntimeSpec(
        agent_runtime_id="tau2_framework_loop",
        benchmark_kit_id="tau2",
        scheduler_type="framework_loop",
        environment_profile_id="tau2-local-process",
        resource_policy={"resource_kind": "local_process", "lifecycle": "per_sample"},
        verifier_binding_id="tau2_verifier",
    ),
    "tau2_acp_client": AgentRuntimeSpec(
        agent_runtime_id="tau2_acp_client",
        benchmark_kit_id="tau2",
        scheduler_type="acp_client",
        client_id="acp",
        environment_profile_id="tau2-local-process",
        resource_policy={"resource_kind": "local_process", "lifecycle": "per_sample"},
        verifier_binding_id="tau2_verifier",
    ),
}


def resolve_agent_runtime_spec(agent_runtime_id: str) -> AgentRuntimeSpec:
    """Resolve the declared runtime spec from the builtin runtime catalog."""

    if agent_runtime_id not in _BUILTIN_RUNTIME_SPECS:
        raise KeyError(f"Unknown agent runtime '{agent_runtime_id}'")
    spec = _BUILTIN_RUNTIME_SPECS[agent_runtime_id]
    if spec.benchmark_kit_id == "skillsbench":
        return replace(spec, benchmark_kit_id="swebench")
    return spec


def compile_agent_runtime_plan(
    *,
    agent_runtime_id: str,
    environment_profile: dict[str, Any] | None = None,
    provider_config: dict[str, Any] | None = None,
    resources: dict[str, Any] | None = None,
    startup_env: dict[str, Any] | None = None,
    lifecycle: str | None = None,
) -> CompiledRuntimePlan:
    """Compile a runtime plan from the builtin runtime catalog and benchmark kit."""

    # STEP 1: Resolve the runtime spec and benchmark kit.
    runtime_spec = resolve_agent_runtime_spec(agent_runtime_id)
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
    workflow_bundle = (
        _build_acp_workflow_bundle(runtime_spec)
        if runtime_spec.scheduler_type == "acp_client"
        else benchmark_kit.resolve_workflow_bundle(runtime_spec.scheduler_type)
    )
    verifier_adapter = benchmark_kit.build_verifier_adapter()
    verifier_resources = {"adapter": verifier_adapter} if verifier_adapter is not None else {}
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
        verifier_kind=_verifier_kind_for_adapter(verifier_adapter),
        verifier_resource_refs=verifier_resources,
        failure_policy="bind_failure",
    )

    # STEP 3: Resolve resource plan and artifact policy.
    kit_module = importlib.import_module(
        f"gage_eval.agent_eval_kits.{runtime_spec.benchmark_kit_id}.kit"
    )
    resource_plan = kit_module.build_resource_plan(
        runtime_spec,
        environment_profile=environment_profile,
        provider_config=provider_config,
        resources=resources,
        startup_env=startup_env,
        lifecycle=lifecycle,
    )
    resolved_profile = dict(resource_plan.get("environment_profile") or {})
    resolved_provider_config = dict(resource_plan.get("provider_config") or {})
    resolved_resources = dict(resource_plan.get("resources") or {})
    resolved_startup_env = dict(resource_plan.get("startup_env") or {})
    resolved_provider = str(
        resource_plan.get("resource_kind")
        or resolved_profile.get("provider")
        or runtime_spec.resource_policy.get("resource_kind")
        or ""
    )
    resolved_profile_id = str(
        resolved_profile.get("profile_id")
        or runtime_spec.environment_profile_id
        or runtime_spec.benchmark_kit_id
    )
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
        "resource_plan": _cache_key_compatible(resource_plan),
        "artifact_policy": artifact_policy,
    }
    cache_key = hashlib.md5(
        json.dumps(cache_key_payload, sort_keys=True, default=str).encode("utf-8")
    ).hexdigest()
    return CompiledRuntimePlan(
        run_id=f"compiled-{cache_key[:12]}",
        dut_id=runtime_spec.agent_runtime_id,
        agent_id=runtime_spec.agent_runtime_id,
        env_id=resolved_profile_id,
        benchmark_id=runtime_spec.benchmark_kit_id,
        trial_policy={"trials": 1},
        kit_id=runtime_spec.benchmark_kit_id,
        kit_entry=benchmark_kit,
        kit_config={},
        agent_config={},
        scheduler_type=runtime_spec.scheduler_type,
        scheduler_config={},
        environment_provider=resolved_provider,
        environment_profile_id=resolved_profile_id,
        environment_profile=resolved_profile,
        lifecycle=str(resource_plan.get("lifecycle") or runtime_spec.resource_policy.get("lifecycle") or "per_sample"),
        provider_config=resolved_provider_config,
        startup_env=resolved_startup_env,
        resources=resolved_resources,
        verifier_environment_policy=benchmark_kit.verifier_environment_policy,
        verifier_environment_profile_id=benchmark_kit.verifier_environment_profile_id,
        workflow_bundle=workflow_bundle,
        tool_registry=_ensure_runtime_tool_registry(benchmark_kit.build_tool_registry()),
        tool_provider_adapter=None,
        verifier_adapter=verifier_adapter,
        artifact_sink=None,
        plan_id=f"compiled-{cache_key[:12]}",
        runtime_spec=runtime_spec,
        scheduler_handle=runtime_spec.scheduler_type,
        kit_runtime_ref=benchmark_kit.runtime_entry,
        judge_binding=judge_binding,
        resource_plan=resource_plan,
        artifact_policy=artifact_policy,
        cache_key=cache_key,
        compile_diagnostics=diagnostics,
    )


def _cache_key_compatible(value: Any) -> Any:
    if callable(value):
        return {
            "callable": f"{getattr(value, '__module__', '')}.{getattr(value, '__qualname__', repr(value))}"
        }
    if isinstance(value, dict):
        return {str(key): _cache_key_compatible(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set, frozenset)):
        return [_cache_key_compatible(item) for item in value]
    return value


def build_compiled_runtime_executor(
    *,
    compiled_plan: CompiledRuntimePlan,
    agent_backend: Any = None,
    static_model_backend: Any = None,
    installed_client_override: Any = None,
    prompt_renderer=None,
    max_turns: int = 150,
    pre_hooks=None,
    post_hooks=None,
    mcp_clients: dict[str, Any] | None = None,
    sandbox_manager: Any | None = None,
    environment_manager: Any | None = None,
) -> CompiledRuntimeExecutor:
    """Build the hot-path executor for one compiled runtime plan."""

    del sandbox_manager
    if compiled_plan.runtime_spec.scheduler_type == "installed_client":
        scheduler_handle = InstalledClientScheduler(
            resolve_installed_client(
                client_id=compiled_plan.runtime_spec.client_id,
                client_override=installed_client_override,
                client_config=compiled_plan.scheduler_config,
            )
        )
    elif compiled_plan.runtime_spec.scheduler_type == "acp_client":
        scheduler_handle = AcpClientScheduler()
    elif compiled_plan.runtime_spec.scheduler_type == "framework_loop":
        runtime_tool_registry = _ensure_runtime_tool_registry(compiled_plan.tool_registry)
        mcp_processes = _register_agent_tooling_contributions(
            runtime_tool_registry,
            agent_config=compiled_plan.agent_config,
            mcp_clients=mcp_clients,
        )
        resolved_backend = (
            StaticModelBackendAdapter(static_model_backend)
            if static_model_backend is not None
            else agent_backend
        )
        scheduler_handle = FrameworkLoopScheduler(
            backend=resolved_backend,
            tool_router=ToolRouter(runtime_tool_registry),
            tool_registry=runtime_tool_registry,
            prompt_renderer=prompt_renderer,
            max_turns=max_turns,
            pre_hooks=pre_hooks,
            post_hooks=post_hooks,
            mcp_clients=mcp_clients,
        )
    else:
        raise KeyError(f"Unsupported scheduler '{compiled_plan.runtime_spec.scheduler_type}'")

    resolved_plan = replace(
        compiled_plan,
        scheduler_handle=scheduler_handle,
        tool_registry=runtime_tool_registry if compiled_plan.runtime_spec.scheduler_type == "framework_loop" else compiled_plan.tool_registry,
    )
    artifact_sink = RuntimeArtifactSink()
    return CompiledRuntimeExecutor(
        compiled_plan=resolved_plan,
        resource_manager=RuntimeResourceManager(environment_manager=environment_manager),
        session_factory=AgentRuntimeSessionFactory(artifact_sink),
        verifier_runner=DefaultVerifierRunner(),
        artifact_sink=artifact_sink,
        trace_emitter=RuntimeTraceEmitter(),
        failure_mapper=FailureMapper(),
        mcp_processes=mcp_processes if compiled_plan.runtime_spec.scheduler_type == "framework_loop" else [],
    )


def _ensure_runtime_tool_registry(value: Any) -> RuntimeToolRegistry:
    """Normalize cold-path kit registry output into the runtime ToolRegistry contract."""

    if isinstance(value, RuntimeToolRegistry):
        return value
    registry = RuntimeToolRegistry()
    if value is None:
        return registry
    if isinstance(value, dict) and not value:
        return registry
    if isinstance(value, list):
        for raw_schema in value:
            if isinstance(raw_schema, dict):
                registry.register_provider_schema(raw_schema, provider="kit")
        return registry
    if isinstance(value, dict):
        for raw_schema in value.values():
            if isinstance(raw_schema, dict):
                registry.register_provider_schema(raw_schema, provider="kit")
        return registry
    return registry


def _verifier_kind_for_adapter(adapter: Any) -> str:
    return "native" if isinstance(adapter, NativeVerifierAdapter) else "judge_adapter"


def _build_acp_workflow_bundle(runtime_spec: AgentRuntimeSpec) -> SchedulerWorkflowBundle:
    return SchedulerWorkflowBundle(
        bundle_id=f"{runtime_spec.benchmark_kit_id}.acp_client",
        benchmark_kit_id=runtime_spec.benchmark_kit_id,
        scheduler_type="acp_client",
        failure_normalizer=lambda **_: {},
    )


def _register_agent_tooling_contributions(
    registry: RuntimeToolRegistry,
    *,
    agent_config: dict[str, Any] | None,
    mcp_clients: dict[str, Any] | None,
) -> list[McpServerProcess]:
    mcp_processes: list[McpServerProcess] = []
    tooling = (agent_config or {}).get("tooling") if isinstance(agent_config, dict) else None
    if not isinstance(tooling, dict):
        tooling = {}

    skill_manifests = tooling.get("skill_manifests")
    if isinstance(skill_manifests, dict):
        resolver = SkillManifestResolver(
            {
                str(skill_id): manifest
                for skill_id, manifest in skill_manifests.items()
                if isinstance(manifest, dict)
            },
            policy=_skill_policy_from_tooling(tooling),
        )
        for skill_id in _string_list(tooling.get("skill_ids")):
            for schema in resolver.resolve(skill_id):
                registry.register_schema(
                    schema,
                    provider_kind="local_function",
                    metadata={"skill_id": skill_id},
                )

    if not mcp_clients:
        return mcp_processes
    selected_servers = _string_list(tooling.get("mcp_servers"))
    for server_id in selected_servers:
        client_ref = mcp_clients.get(server_id)
        if client_ref is None:
            continue
        client, process = _resolve_mcp_client(client_ref)
        if process is not None:
            mcp_processes.append(process)
        for schema in discover_mcp_tools(client, server_id=server_id):
            registry.register_mcp_tool(schema, client, metadata={"server_id": server_id})
    return mcp_processes


def _resolve_mcp_client(client_ref: Any) -> tuple[Any, McpServerProcess | None]:
    if isinstance(client_ref, McpServerProcess):
        return client_ref.client, client_ref
    return client_ref, None


def _skill_policy_from_tooling(tooling: dict[str, Any]) -> SkillPolicy:
    raw_policy = tooling.get("skill_policy")
    if not isinstance(raw_policy, dict):
        return SkillPolicy.allow_all()
    allowed = raw_policy.get("allowed_skill_ids")
    if allowed is None:
        return SkillPolicy.allow_all()
    return SkillPolicy.from_iterable(_string_list(allowed))


def _string_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(item) for item in value if isinstance(item, str) and item]
