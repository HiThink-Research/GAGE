from __future__ import annotations

import pytest

import gage_eval.agent_runtime.resolver as resolver_module
from gage_eval.agent_runtime import RuntimeCompileError, compile_agent_runtime_plan, resolve_agent_runtime_spec
from gage_eval.agent_runtime.compiled_plan import SchedulerWorkflowBundle


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


def test_compile_runtime_plan_exposes_diagnostics_when_verifier_binding_is_missing(monkeypatch) -> None:
    class _KitStub:
        verifier_kind = "native"
        runtime_entry = object()

        def resolve_workflow_bundle(self, scheduler_type: str):
            return SchedulerWorkflowBundle(
                bundle_id=f"terminal_bench.{scheduler_type}",
                benchmark_kit_id="terminal_bench",
                scheduler_type=scheduler_type,
                failure_normalizer=lambda **_: {},
            )

        def resolve_verifier_resources(self) -> dict[str, object]:
            return {}

    monkeypatch.setattr(resolver_module, "load_benchmark_kit", lambda _benchmark_kit_id: _KitStub())

    with pytest.raises(RuntimeCompileError) as exc_info:
        compile_agent_runtime_plan(agent_runtime_id="terminal_bench_framework_loop")

    assert exc_info.value.diagnostics
    assert exc_info.value.diagnostics[0]["code"] == "verifier_resources_missing"
    assert exc_info.value.diagnostics[0]["benchmark_kit_id"] == "terminal_bench"
