from __future__ import annotations

from uuid import uuid4

import pytest

from gage_eval.config.pipeline_config import PipelineConfig
from gage_eval.config.registry import ConfigRegistry, _collect_runtime_registry_packages
from gage_eval.registry import RegistryRuntimeMutationError, registry


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
