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
