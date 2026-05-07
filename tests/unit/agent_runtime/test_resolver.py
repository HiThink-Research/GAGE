from __future__ import annotations

import asyncio

import pytest

import gage_eval.agent_runtime.resolver as resolver_module
from gage_eval.agent_runtime import RuntimeCompileError, compile_agent_runtime_plan, resolve_agent_runtime_spec
from gage_eval.agent_runtime.compiled_plan import SchedulerWorkflowBundle
from gage_eval.agent_runtime.resolver import build_compiled_runtime_executor
from gage_eval.agent_runtime.schedulers.framework_loop import StaticModelBackendAdapter
from gage_eval.agent_runtime.tooling.registry import RuntimeToolRegistry


class _StaticBackend:
    def __init__(self, result=None) -> None:
        self.calls = []
        self.result = result or {"answer": "static answer", "usage": {"input_tokens": 1}}

    async def ainvoke(self, payload):
        self.calls.append(payload)
        return self.result


class _McpClient:
    def __init__(self) -> None:
        self.called = []
        self.list_tools_calls = 0

    def list_tools(self):
        self.list_tools_calls += 1
        return [{"name": "mcp_lookup", "inputSchema": {"type": "object"}}]

    async def call_tool(self, name, arguments):
        self.called.append((name, arguments))
        return {"ok": True}


@pytest.mark.parametrize(
    ("agent_runtime_id", "benchmark_kit_id", "scheduler_type", "resource_kind"),
    [
        ("terminal_bench_installed_client", "terminal_bench", "installed_client", "docker"),
        ("terminal_bench_framework_loop", "terminal_bench", "framework_loop", "docker"),
        ("swebench_installed_client", "swebench", "installed_client", "docker"),
        ("swebench_framework_loop", "swebench", "framework_loop", "docker"),
        ("appworld_installed_client", "appworld", "installed_client", "docker"),
        ("appworld_framework_loop", "appworld", "framework_loop", "docker"),
        ("tau2_installed_client", "tau2", "installed_client", "local_process"),
        ("tau2_framework_loop", "tau2", "framework_loop", "local_process"),
    ],
)
def test_phase1_runtime_spec_matrix(
    agent_runtime_id: str,
    benchmark_kit_id: str,
    scheduler_type: str,
    resource_kind: str,
) -> None:
    spec = resolve_agent_runtime_spec(agent_runtime_id)

    assert spec.agent_runtime_id == agent_runtime_id
    assert spec.benchmark_kit_id == benchmark_kit_id
    assert spec.scheduler_type == scheduler_type
    assert spec.resource_policy["resource_kind"] == resource_kind
    if scheduler_type == "installed_client":
        assert spec.client_id == "codex"


@pytest.mark.parametrize(
    ("agent_runtime_id", "benchmark_kit_id", "scheduler_type", "resource_kind"),
    [
        ("terminal_bench_installed_client", "terminal_bench", "installed_client", "docker"),
        ("terminal_bench_framework_loop", "terminal_bench", "framework_loop", "docker"),
        ("swebench_installed_client", "swebench", "installed_client", "docker"),
        ("swebench_framework_loop", "swebench", "framework_loop", "docker"),
        ("appworld_installed_client", "appworld", "installed_client", "docker"),
        ("appworld_framework_loop", "appworld", "framework_loop", "docker"),
        ("tau2_installed_client", "tau2", "installed_client", "local_process"),
        ("tau2_framework_loop", "tau2", "framework_loop", "local_process"),
    ],
)
def test_compile_phase1_runtime_matrix(
    agent_runtime_id: str,
    benchmark_kit_id: str,
    scheduler_type: str,
    resource_kind: str,
) -> None:
    plan = compile_agent_runtime_plan(agent_runtime_id=agent_runtime_id)

    assert plan.runtime_spec.benchmark_kit_id == benchmark_kit_id
    assert plan.workflow_bundle.bundle_id == f"{benchmark_kit_id}.{scheduler_type}"
    assert plan.judge_binding.judge_mode == "runtime_verifier"
    assert "adapter" in (plan.judge_binding.verifier_resource_refs or {})
    assert plan.resource_plan["resource_kind"] == resource_kind
    assert plan.cache_key


def test_swebench_runtime_plan_carries_provider_config_resolver_with_stable_cache_payload() -> None:
    plan = compile_agent_runtime_plan(agent_runtime_id="swebench_framework_loop")

    resolver = plan.resource_plan.get("provider_config_resolver")
    assert callable(resolver)
    assert resolver_module._cache_key_compatible({"resolver": resolver}) == {
        "resolver": {
            "callable": "gage_eval.agent_eval_kits.swebench.kit.resolve_provider_config"
        }
    }


def test_compile_runtime_plan_exposes_diagnostics_when_verifier_binding_is_missing(monkeypatch) -> None:
    class _KitStub:
        runtime_entry = object()

        def resolve_workflow_bundle(self, scheduler_type: str):
            return SchedulerWorkflowBundle(
                bundle_id=f"terminal_bench.{scheduler_type}",
                benchmark_kit_id="terminal_bench",
                scheduler_type=scheduler_type,
                failure_normalizer=lambda **_: {},
            )

        def build_verifier_adapter(self):
            return None

    monkeypatch.setattr(resolver_module, "load_benchmark_kit", lambda _benchmark_kit_id: _KitStub())

    with pytest.raises(RuntimeCompileError) as exc_info:
        compile_agent_runtime_plan(agent_runtime_id="terminal_bench_framework_loop")

    assert exc_info.value.diagnostics
    assert exc_info.value.diagnostics[0]["code"] == "verifier_resources_missing"
    assert exc_info.value.diagnostics[0]["benchmark_kit_id"] == "terminal_bench"


def test_static_model_backend_adapter_forwards_agent_loop_payload_to_static_backend() -> None:
    backend = _StaticBackend(
        {
            "choices": [
                {
                    "message": {
                        "content": "provider answer",
                        "tool_calls": [
                            {
                                "id": "call-1",
                                "type": "function",
                                "function": {"name": "lookup", "arguments": "{}"},
                            }
                        ],
                    }
                }
            ],
            "usage": {"prompt_tokens": 4, "completion_tokens": 2},
        }
    )
    adapter = StaticModelBackendAdapter(backend)

    result = asyncio.run(
        adapter.ainvoke(
            {
                "messages": [{"role": "user", "content": "hi"}],
                "tools": [{"type": "function", "function": {"name": "lookup"}}],
                "tool_choice": "auto",
                "sample": {"id": "sample-1"},
                "sampling_params": {"temperature": 0},
            }
        )
    )

    assert backend.calls == [
        {
            "messages": [{"role": "user", "content": "hi"}],
            "tools": [{"type": "function", "function": {"name": "lookup"}}],
            "tool_choice": "auto",
            "sample": {"id": "sample-1"},
            "sampling_params": {"temperature": 0},
        }
    ]
    assert result["answer"] == "provider answer"
    assert result["tool_calls"] == [
        {
            "id": "call-1",
            "type": "function",
            "function": {"name": "lookup", "arguments": "{}"},
        }
    ]
    assert result["usage"] == {"prompt_tokens": 4, "completion_tokens": 2}
    assert result["agent_trace"] == []


def test_build_framework_loop_executor_wraps_static_model_backend() -> None:
    plan = compile_agent_runtime_plan(agent_runtime_id="tau2_framework_loop")
    backend = _StaticBackend()

    executor = build_compiled_runtime_executor(
        compiled_plan=plan,
        agent_backend=None,
        static_model_backend=backend,
    )

    scheduler = executor.compiled_plan.scheduler_handle
    assert isinstance(scheduler._backend, StaticModelBackendAdapter)
    assert scheduler._backend.static_backend is backend


def test_build_framework_loop_executor_registers_mcp_and_skill_tooling_contributions() -> None:
    plan = compile_agent_runtime_plan(agent_runtime_id="tau2_framework_loop")
    plan = resolver_module.replace(
        plan,
        agent_config={
            "tooling": {
                "mcp_servers": ["local_mcp"],
                "skill_ids": ["local_skill"],
                "skill_manifests": {
                    "local_skill": {
                        "tools": [{"name": "skill_lookup", "inputSchema": {"type": "object"}}]
                    }
                },
            }
        },
    )
    mcp_client = _McpClient()

    executor = build_compiled_runtime_executor(
        compiled_plan=plan,
        static_model_backend=_StaticBackend(),
        mcp_clients={"local_mcp": mcp_client},
    )

    registry = executor.compiled_plan.tool_registry
    assert isinstance(registry, RuntimeToolRegistry)
    assert registry.get("mcp_lookup").provider_kind == "mcp"
    assert registry.get("mcp_lookup").executor is mcp_client
    assert registry.get("skill_lookup").schema.metadata["provider"] == "skill:local_skill"
    projected_names = {
        tool["function"]["name"]
        for tool in executor.compiled_plan.scheduler_handle._tool_registry.project_tool_schemas()
    }
    assert {"mcp_lookup", "skill_lookup"}.issubset(projected_names)


def test_build_framework_loop_executor_applies_skill_policy() -> None:
    plan = compile_agent_runtime_plan(agent_runtime_id="tau2_framework_loop")
    plan = resolver_module.replace(
        plan,
        agent_config={
            "tooling": {
                "skill_ids": ["local_skill"],
                "skill_policy": {"allowed_skill_ids": ["different_skill"]},
                "skill_manifests": {
                    "local_skill": {
                        "tools": [{"name": "skill_lookup", "inputSchema": {"type": "object"}}]
                    }
                },
            }
        },
    )

    with pytest.raises(Exception, match="skill is denied by kit policy"):
        build_compiled_runtime_executor(
            compiled_plan=plan,
            static_model_backend=_StaticBackend(),
        )


def test_framework_loop_executor_does_not_cold_discover_unrequested_mcp_clients() -> None:
    plan = compile_agent_runtime_plan(agent_runtime_id="appworld_framework_loop")
    plan = resolver_module.replace(plan, agent_config={"tooling": {}})
    mcp_client = _McpClient()

    executor = build_compiled_runtime_executor(
        compiled_plan=plan,
        static_model_backend=_StaticBackend(),
        mcp_clients={"appworld_env": mcp_client},
    )

    registry = executor.compiled_plan.tool_registry
    assert isinstance(registry, RuntimeToolRegistry)
    assert registry.get("mcp_lookup") is None
    assert mcp_client.list_tools_calls == 0
