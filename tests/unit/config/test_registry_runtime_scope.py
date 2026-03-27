from __future__ import annotations

import importlib
import sys
from types import ModuleType
from uuid import uuid4

import pytest

from gage_eval.config.pipeline_config import PipelineConfig, RoleAdapterSpec
from gage_eval.config.registry import ConfigRegistry, _collect_runtime_registry_packages
from gage_eval.evaluation.runtime_builder import build_runtime
from gage_eval.observability.trace import ObservabilityTrace
from gage_eval.registry import RegistryBootstrapCoordinator, RegistryManager, RegistryRuntimeMutationError, registry
from gage_eval.role.adapters.arena import ArenaRoleAdapter
from gage_eval.role.model.backends.builder import build_backend
from gage_eval.role.resource_profile import NodeResource, ResourceProfile


def _build_minimal_config(**extra):
    payload = {
        "datasets": [{"dataset_id": "d1", "loader": "dummy"}],
        "role_adapters": [{"adapter_id": "dut", "role_type": "dut_model"}],
        "custom": {"steps": [{"step": "inference", "adapter_id": "dut"}]},
    }
    payload.update(extra)
    return PipelineConfig.from_dict(payload)


def _build_isolated_config_registry() -> ConfigRegistry:
    isolated_registry = RegistryManager()
    for kind in registry.kinds():
        isolated_registry.declare_kind(kind, desc=registry.describe_kind(kind))
    return ConfigRegistry(
        bootstrap_coordinator=RegistryBootstrapCoordinator(isolated_registry),
    )


def _detach_modules(*module_names: str) -> dict[str, ModuleType | None]:
    previous_modules: dict[str, ModuleType | None] = {}
    for name in module_names:
        previous_modules[name] = sys.modules.pop(name, None)
        parent_name, _, attr_name = name.rpartition(".")
        if parent_name and attr_name:
            parent_module = sys.modules.get(parent_name)
            if parent_module is not None and hasattr(parent_module, attr_name):
                delattr(parent_module, attr_name)
    return previous_modules


def _restore_modules(previous_modules: dict[str, ModuleType | None]) -> None:
    for name in sorted(previous_modules, key=lambda item: item.count(".")):
        module = previous_modules[name]
        if module is None:
            continue
        sys.modules[name] = module
        parent_name, _, attr_name = name.rpartition(".")
        if parent_name and attr_name:
            parent_module = sys.modules.get(parent_name)
            if parent_module is not None:
                setattr(parent_module, attr_name, module)


@pytest.mark.fast
def test_prepare_runtime_registry_context_keeps_inline_prompts_run_local() -> None:
    prompt_id = f"runtime_prompt_{uuid4().hex}"
    config = _build_minimal_config(
        prompts=[{"prompt_id": prompt_id, "renderer": "jinja2", "template": "hello"}],
    )
    config_registry = ConfigRegistry()

    context = config_registry.prepare_runtime_registry_context(config, run_id=f"run-{uuid4().hex}")
    scoped_registry = config_registry.with_runtime_registry_context(context)
    prompts = scoped_registry.materialize_prompts(config)

    assert prompt_id in prompts
    assert scoped_registry.registry_view.get("prompts", prompt_id) is prompts[prompt_id]
    with pytest.raises(KeyError):
        registry.get("prompts", prompt_id)

    context.close()
    with pytest.raises(KeyError):
        registry.get("prompts", prompt_id)


@pytest.mark.fast
def test_runtime_guard_blocks_global_mutation_until_lease_released() -> None:
    config = _build_minimal_config()
    config_registry = ConfigRegistry()
    context = config_registry.prepare_runtime_registry_context(config, run_id=f"run-{uuid4().hex}")

    with pytest.raises(RegistryRuntimeMutationError):
        registry.register(
            "prompts",
            f"late_{uuid4().hex}",
            object(),
            desc="late prompt",
        )

    context.close()

    registry.register(
        "prompts",
        f"released_{uuid4().hex}",
        object(),
        desc="released prompt",
    )


@pytest.mark.fast
def test_registry_view_lease_release_clears_scoped_cache() -> None:
    config = _build_minimal_config()
    config_registry = ConfigRegistry()
    context = config_registry.prepare_runtime_registry_context(config, run_id=f"run-{uuid4().hex}")

    cache = context.view.get_scoped_cache("step_contract_catalog")
    cache["catalog"] = object()
    assert context.view.get_scoped_cache("step_contract_catalog")

    context.close()

    assert context.view.get_scoped_cache("step_contract_catalog") == {}


@pytest.mark.fast
def test_runtime_package_selection_preloads_step_contracts_and_report_generators() -> None:
    config = _build_minimal_config()

    packages = _collect_runtime_registry_packages(config)

    assert packages["pipeline_steps"] == ("gage_eval.pipeline.steps",)
    assert packages["summary_generators"] == ("gage_eval.reporting.summary_generators",)


@pytest.mark.fast
def test_runtime_package_selection_preloads_gamearena_runtime_packages_for_arena_adapters() -> None:
    config = _build_minimal_config(
        role_adapters=[
            {
                "adapter_id": "arena",
                "role_type": "arena",
                "params": {},
            }
        ],
        custom={"steps": [{"step": "arena", "adapter_id": "arena"}]},
    )

    packages = _collect_runtime_registry_packages(config)

    assert packages["game_kits"] == ("gage_eval.game_kits.registry",)
    assert packages["scheduler_bindings"] == ("gage_eval.role.arena.schedulers.specs",)
    assert packages["support_workflows"] == ("gage_eval.role.arena.support.specs",)


@pytest.mark.fast
def test_prepare_runtime_registry_context_primes_metric_assets_before_runtime_freeze() -> None:
    metric_name = "global_piqa_accuracy_local"
    config = _build_minimal_config(
        metrics=[{"metric_id": metric_name, "implementation": metric_name}],
    )
    config_registry = ConfigRegistry()

    context = config_registry.prepare_runtime_registry_context(config, run_id=f"run-{uuid4().hex}")
    try:
        entry = context.view.entry("metrics", metric_name)
        assert entry.name == metric_name
        metrics = config_registry.with_runtime_registry_context(context).materialize_metrics(config)
        assert metric_name in metrics
    finally:
        context.close()


@pytest.mark.fast
def test_prepare_runtime_registry_context_primes_appworld_roles_without_circular_imports() -> None:
    config = _build_minimal_config(
        role_adapters=[
            {
                "adapter_id": "toolchain_main",
                "role_type": "toolchain",
            },
            {
                "adapter_id": "dut_agent_main",
                "role_type": "dut_agent",
            },
        ],
        custom={"steps": [{"step": "support", "adapter_id": "toolchain_main"}]},
    )
    config_registry = ConfigRegistry()
    previous_modules = _detach_modules(
        "gage_eval.role.adapters",
        "gage_eval.role.adapters.base",
        "gage_eval.role.toolchain",
        "gage_eval.role.toolchain.toolchain",
    )

    context = config_registry.prepare_runtime_registry_context(config, run_id=f"run-{uuid4().hex}")
    try:
        assert context.view.entry("roles", "toolchain").name == "toolchain"
        assert context.view.entry("roles", "dut_agent").name == "dut_agent"
        assert not any(
            issue.kind == "roles" and issue.name == "toolchain" for issue in context.discovery_report.issues
        )
    finally:
        context.close()
        _detach_modules(
            "gage_eval.role.adapters",
            "gage_eval.role.adapters.base",
            "gage_eval.role.toolchain",
            "gage_eval.role.toolchain.toolchain",
        )
        _restore_modules(previous_modules)


@pytest.mark.fast
def test_prepare_runtime_registry_context_primes_arena_assets_before_runtime_freeze() -> None:
    config = _build_minimal_config(
        role_adapters=[
            {
                "adapter_id": "arena",
                "role_type": "arena",
                "params": {
                    "environment": {"impl": "gomoku_local_v1"},
                    "parser": {"impl": "grid_parser_v1"},
                    "visualizer": {
                        "enabled": True,
                        "renderer": {"impl": "gomoku_board_v1"},
                    },
                },
            }
        ],
        custom={"steps": [{"step": "arena", "adapter_id": "arena"}]},
    )
    config_registry = ConfigRegistry()

    context = config_registry.prepare_runtime_registry_context(config, run_id=f"run-{uuid4().hex}")
    try:
        assert context.view.entry("arena_impls", "gomoku_local_v1").name == "gomoku_local_v1"
        assert context.view.entry("parser_impls", "grid_parser_v1").name == "grid_parser_v1"
        assert context.view.entry("renderer_impls", "gomoku_board_v1").name == "gomoku_board_v1"
    finally:
        context.close()


@pytest.mark.fast
def test_prepare_runtime_registry_context_preloads_gamearena_runtime_packages(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime_module = importlib.import_module("gage_eval.registry.runtime")
    imported_modules: list[str] = []
    real_import_module = runtime_module.importlib.import_module

    def _tracking_import(name: str, package: str | None = None):
        imported_modules.append(name)
        return real_import_module(name, package)

    monkeypatch.setattr(runtime_module.importlib, "import_module", _tracking_import)

    config = _build_minimal_config(
        role_adapters=[
            {
                "adapter_id": "arena",
                "role_type": "arena",
                "class_path": "gage_eval.role.adapters.arena.ArenaRoleAdapter",
                "params": {},
            }
        ],
        custom={"steps": [{"step": "arena", "adapter_id": "arena"}]},
    )
    config_registry = _build_isolated_config_registry()

    context = config_registry.prepare_runtime_registry_context(config, run_id=f"run-{uuid4().hex}")
    try:
        assert "gage_eval.game_kits.registry" in imported_modules
        assert "gage_eval.role.arena.schedulers.specs" in imported_modules
        assert "gage_eval.role.arena.support.specs" in imported_modules
        assert context.view.entry("game_kits", "tictactoe").name == "tictactoe"
        assert context.view.entry("scheduler_bindings", "turn/default").name == "turn/default"
        assert context.view.entry("support_workflows", "arena/default").name == "arena/default"
    finally:
        context.close()


@pytest.mark.fast
def test_prepare_runtime_registry_context_preloads_gamearena_runtime_packages_across_sequential_contexts() -> None:
    config = _build_minimal_config(
        role_adapters=[
            {
                "adapter_id": "arena",
                "role_type": "arena",
                "class_path": "gage_eval.role.adapters.arena.ArenaRoleAdapter",
                "params": {},
            }
        ],
        custom={"steps": [{"step": "arena", "adapter_id": "arena"}]},
    )
    config_registry = _build_isolated_config_registry()

    first_context = config_registry.prepare_runtime_registry_context(config, run_id=f"run-{uuid4().hex}")
    try:
        assert first_context.view.entry("game_kits", "tictactoe").name == "tictactoe"
        assert first_context.view.entry("scheduler_bindings", "turn/default").name == "turn/default"
        assert first_context.view.entry("support_workflows", "arena/default").name == "arena/default"
    finally:
        first_context.close()

    second_context = config_registry.prepare_runtime_registry_context(config, run_id=f"run-{uuid4().hex}")
    try:
        assert second_context.view.entry("game_kits", "tictactoe").name == "tictactoe"
        assert second_context.view.entry("scheduler_bindings", "turn/default").name == "turn/default"
        assert second_context.view.entry("support_workflows", "arena/default").name == "arena/default"
    finally:
        second_context.close()


@pytest.mark.fast
def test_prepare_runtime_registry_context_replays_gamearena_support_registrations_for_cached_modules() -> None:
    importlib.import_module("gage_eval.role.arena.support.specs")

    config = _build_minimal_config(
        role_adapters=[
            {
                "adapter_id": "arena",
                "role_type": "arena",
                "class_path": "gage_eval.role.adapters.arena.ArenaRoleAdapter",
                "params": {
                    "game_kit": "tictactoe",
                    "env": "tictactoe_standard",
                },
            }
        ],
        custom={"steps": [{"step": "arena", "adapter_id": "arena"}]},
    )
    config_registry = _build_isolated_config_registry()

    context = config_registry.prepare_runtime_registry_context(config, run_id=f"run-{uuid4().hex}")
    try:
        assert context.view.entry("support_workflows", "arena/default").name == "arena/default"
        assert context.view.entry("support_units", "arena/default").name == "arena/default"
        assert context.view.entry("observation_workflows", "arena/default").name == "arena/default"
    finally:
        context.close()


@pytest.mark.fast
def test_prepare_runtime_registry_context_primes_provider_default_renderer_before_runtime_freeze() -> None:
    config = _build_minimal_config(
        role_adapters=[
            {
                "adapter_id": "arena",
                "role_type": "arena",
                "params": {
                    "environment": {"impl": "tictactoe_v1"},
                    "parser": {"impl": "grid_parser_v1"},
                    "visualizer": {
                        "enabled": True,
                    },
                },
            }
        ],
        custom={"steps": [{"step": "arena", "adapter_id": "arena"}]},
    )
    config_registry = ConfigRegistry()

    context = config_registry.prepare_runtime_registry_context(config, run_id=f"run-{uuid4().hex}")
    try:
        assert context.view.entry("renderer_impls", "tictactoe_board_v1").name == "tictactoe_board_v1"
    finally:
        context.close()


@pytest.mark.fast
def test_metrics_builtin_package_import_is_safe_with_runtime_guard() -> None:
    config = _build_minimal_config()
    config_registry = ConfigRegistry()
    context = config_registry.prepare_runtime_registry_context(config, run_id=f"run-{uuid4().hex}")
    previous_modules = _detach_modules("gage_eval.metrics.builtin", "gage_eval.metrics.builtin.gomoku")

    try:
        module = importlib.import_module("gage_eval.metrics.builtin")
        assert module.__name__ == "gage_eval.metrics.builtin"
    finally:
        context.close()
        _detach_modules("gage_eval.metrics.builtin", "gage_eval.metrics.builtin.gomoku")
        _restore_modules(previous_modules)


@pytest.mark.fast
def test_build_runtime_with_empty_metrics_task_config_is_runtime_guard_safe() -> None:
    config = PipelineConfig.from_dict(
        {
            "datasets": [
                {
                    "dataset_id": "demo_echo_dataset",
                    "loader": "jsonl",
                    "params": {
                        "path": "config/builtin_templates/demo_echo/data/demo_echo.jsonl",
                        "streaming": False,
                    },
                }
            ],
            "backends": [
                {
                    "backend_id": "demo_echo_dummy",
                    "type": "dummy",
                    "config": {"responses": [], "echo_prompt": True, "cycle": True},
                }
            ],
            "role_adapters": [
                {
                    "adapter_id": "demo_echo_dut",
                    "role_type": "dut_model",
                    "backend_id": "demo_echo_dummy",
                    "capabilities": ["chat_completion"],
                }
            ],
            "tasks": [
                {
                    "task_id": "demo_echo_task",
                    "dataset_id": "demo_echo_dataset",
                    "steps": [{"step": "inference", "adapter_id": "demo_echo_dut"}],
                    "max_samples": 1,
                }
            ],
        }
    )
    trace = ObservabilityTrace(run_id=f"run-{uuid4().hex}")
    runtime = build_runtime(
        config=config,
        registry=ConfigRegistry(),
        resource_profile=ResourceProfile(
            nodes=[NodeResource(node_id="local", gpus=0, cpus=1)]
        ),
        trace=trace,
    )

    try:
        assert runtime is not None
    finally:
        runtime.shutdown()
        trace.close(cache_store=None)


@pytest.mark.fast
def test_build_runtime_enables_inline_sample_execution_for_single_worker_tasks() -> None:
    config = PipelineConfig.from_dict(
        {
            "datasets": [
                {
                    "dataset_id": "demo_echo_dataset",
                    "loader": "jsonl",
                    "params": {
                        "path": "config/builtin_templates/demo_echo/data/demo_echo.jsonl",
                        "streaming": False,
                    },
                }
            ],
            "backends": [
                {
                    "backend_id": "demo_echo_dummy",
                    "type": "dummy",
                    "config": {"responses": [], "echo_prompt": True, "cycle": True},
                }
            ],
            "role_adapters": [
                {
                    "adapter_id": "demo_echo_dut",
                    "role_type": "dut_model",
                    "backend_id": "demo_echo_dummy",
                    "capabilities": ["chat_completion"],
                }
            ],
            "tasks": [
                {
                    "task_id": "t1",
                    "dataset_id": "demo_echo_dataset",
                    "steps": [{"step": "inference", "adapter_id": "demo_echo_dut"}],
                    "max_samples": 1,
                    "concurrency": 1,
                }
            ],
        }
    )
    trace = ObservabilityTrace(run_id=f"run-{uuid4().hex}")
    runtime = build_runtime(
        config=config,
        registry=ConfigRegistry(),
        resource_profile=ResourceProfile(
            nodes=[NodeResource(node_id="local", gpus=0, cpus=1)]
        ),
        trace=trace,
    )

    try:
        entry = runtime._tasks[0]
        controller = entry.sample_loop._execution_controller
        assert controller is not None
        assert controller.sample_workers == 1
        assert getattr(controller, "_inline_sample_execution") is True
    finally:
        runtime.shutdown()
        trace.close(cache_store=None)


@pytest.mark.fast
def test_build_backend_with_runtime_registry_view_skips_manifest_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    clone = registry.clone()
    view = clone.freeze(view_id=f"backend-view-{uuid4().hex}")
    backend_type = f"missing_backend_{uuid4().hex}"

    def _unexpected_import(*args, **kwargs):
        raise AssertionError("runtime registry views must not trigger manifest fallback")

    monkeypatch.setattr(
        "gage_eval.role.model.backends.builder._import_backend_asset_module",
        _unexpected_import,
    )

    with pytest.raises(KeyError, match=f"Backend '{backend_type}' is not registered"):
        build_backend({"type": backend_type, "config": {}}, registry_view=view)


@pytest.mark.fast
def test_resolve_arena_role_adapter_injects_runtime_registry_view() -> None:
    clone = registry.clone()
    view = clone.freeze(view_id=f"arena-view-{uuid4().hex}")
    config_registry = ConfigRegistry(registry_view=view)
    spec = RoleAdapterSpec(
        adapter_id="arena",
        role_type="arena",
        params={"environment": {"impl": "gomoku_local_v1"}},
    )

    adapter = config_registry.resolve_role_adapter(spec)

    assert adapter.__class__.__name__ == ArenaRoleAdapter.__name__
    assert adapter._registry_view is view
