from __future__ import annotations

import importlib
import sys
from uuid import uuid4

import pytest

from gage_eval.config.pipeline_config import PipelineConfig
from gage_eval.config.registry import ConfigRegistry, _collect_runtime_registry_packages
from gage_eval.evaluation.runtime_builder import build_runtime
from gage_eval.observability.trace import ObservabilityTrace
from gage_eval.registry import RegistryRuntimeMutationError, registry
from gage_eval.role.resource_profile import NodeResource, ResourceProfile


def _build_minimal_config(**extra):
    payload = {
        "datasets": [{"dataset_id": "d1", "loader": "dummy"}],
        "role_adapters": [{"adapter_id": "dut", "role_type": "dut_model"}],
        "custom": {"steps": [{"step": "inference", "adapter_id": "dut"}]},
    }
    payload.update(extra)
    return PipelineConfig.from_dict(payload)


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
def test_prepare_runtime_registry_context_primes_metric_assets_before_runtime_freeze() -> None:
    metric_name = "global_piqa_accuracy_local"
    config = _build_minimal_config(
        metrics=[{"metric_id": metric_name, "implementation": metric_name}],
    )
    config_registry = ConfigRegistry()

    context = config_registry.prepare_runtime_registry_context(config, run_id=f"run-{uuid4().hex}")
    try:
        entry = registry.entry("metrics", metric_name)
        assert entry.name == metric_name
        metrics = config_registry.with_runtime_registry_context(context).materialize_metrics(config)
        assert metric_name in metrics
    finally:
        context.close()


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
        assert registry.entry("arena_impls", "gomoku_local_v1").name == "gomoku_local_v1"
        assert registry.entry("parser_impls", "grid_parser_v1").name == "grid_parser_v1"
        assert registry.entry("renderer_impls", "gomoku_board_v1").name == "gomoku_board_v1"
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
        assert registry.entry("renderer_impls", "tictactoe_board_v1").name == "tictactoe_board_v1"
    finally:
        context.close()


@pytest.mark.fast
def test_metrics_builtin_package_import_is_safe_with_runtime_guard() -> None:
    config = _build_minimal_config()
    config_registry = ConfigRegistry()
    context = config_registry.prepare_runtime_registry_context(config, run_id=f"run-{uuid4().hex}")
    previous_modules = {
        name: sys.modules.pop(name, None)
        for name in ("gage_eval.metrics.builtin", "gage_eval.metrics.builtin.gomoku")
    }

    try:
        module = importlib.import_module("gage_eval.metrics.builtin")
        assert module.__name__ == "gage_eval.metrics.builtin"
    finally:
        context.close()
        for name in ("gage_eval.metrics.builtin", "gage_eval.metrics.builtin.gomoku"):
            sys.modules.pop(name, None)
        for name, module in previous_modules.items():
            if module is not None:
                sys.modules[name] = module


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
