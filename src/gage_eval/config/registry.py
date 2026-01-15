"""Configuration registry responsible for wiring plugins at runtime."""

from __future__ import annotations

import importlib
from typing import Any, Dict, Optional

from loguru import logger

from gage_eval.config.pipeline_config import (
    BackendSpec,
    DatasetSpec,
    MetricSpec,
    ModelSpec,
    PipelineConfig,
    RoleAdapterSpec,
)
from gage_eval.assets.prompts.assets import PromptTemplateAsset
from gage_eval.assets.prompts.defaults import resolve_prompt_id_for_adapter
from gage_eval.metrics.registry import MetricRegistry
from gage_eval.registry import registry


class ConfigRegistry:
    """Central registry that resolves datasets, role adapters and metrics.

    The class is intentionally lightweight but extensible: new
    registration surfaces can be added without touching callers. The
    implementation is reminiscent of OpenAI Evals' Registry yet adopts
    some pragmatic shortcuts from llm-eval's YAML-driven loaders.
    """

    def __init__(self) -> None:
        """ConfigRegistry no longer stores per-kind builders; registry handles extensibility."""

    # ------------------------------------------------------------------
    # Resolution API
    # ------------------------------------------------------------------
    def resolve_dataset(self, spec: DatasetSpec, *, trace=None) -> Any:
        loader_name = spec.loader or spec.params.get("loader")
        if not loader_name:
            raise ValueError(f"Dataset '{spec.dataset_id}' missing 'loader'")
        hub_name = spec.hub or spec.params.get("hub")
        if not hub_name:
            if loader_name in {"hf_hub", "modelscope"}:
                raise ValueError(f"Dataset '{spec.dataset_id}' using loader '{loader_name}' requires explicit 'hub'")
            hub_name = "inline"
        hub_params = dict(spec.hub_params or spec.params.get("hub_params", {}))
        hub_cls = registry.get("dataset_hubs", hub_name)
        hub = hub_cls(spec, hub_args=hub_params)
        hub_handle = hub.resolve()
        loader_cls = registry.get("dataset_loaders", loader_name)
        loader = loader_cls(spec)
        return loader.load(hub_handle, trace=trace)

    def resolve_role_adapter(
        self,
        spec: RoleAdapterSpec,
        backends: Optional[Dict[str, Any]] = None,
        prompts: Optional[Dict[str, PromptTemplateAsset]] = None,
    ) -> Any:
        adapter_cls = self._resolve_role_class(spec)
        adapter_kwargs = dict(spec.params)
        backend_obj: Any = None
        # NOTE: Inline backend takes precedence: when an inline backend is declared,
        # ignore the `backend_id` reference.
        if spec.backend is not None:
            if spec.backend_id:
                logger.warning(
                    "RoleAdapter '{}' declares both backend_id='{}' and inline backend; "
                    "inline backend configuration will override the referenced backend.",
                    spec.adapter_id,
                    spec.backend_id,
                )
            backend_obj = self._build_inline_backend(spec.backend)
        elif spec.backend_id:
            if not backends or spec.backend_id not in backends:
                raise KeyError(
                    f"Backend '{spec.backend_id}' referenced by adapter '{spec.adapter_id}' is not defined"
                )
            backend_obj = backends[spec.backend_id]
        if backend_obj is not None:
            adapter_kwargs.setdefault("backend", backend_obj)
        prompt_id = spec.prompt_id or resolve_prompt_id_for_adapter(
            spec.adapter_id,
            spec.role_type,
            spec.params,
        )
        if prompt_id:
            prompt_asset = None
            if prompts and prompt_id in prompts:
                prompt_asset = prompts[prompt_id]
            else:
                try:
                    prompt_asset = registry.get("prompts", prompt_id)
                except KeyError as exc:
                    raise KeyError(
                        f"Prompt '{prompt_id}' referenced by adapter '{spec.adapter_id}' is not defined"
                    ) from exc
            prompt_renderer = prompt_asset.instantiate(spec.prompt_params)
            adapter_kwargs.setdefault("prompt_renderer", prompt_renderer)
        adapter = adapter_cls(
            adapter_id=spec.adapter_id,
            role_type=spec.role_type,
            capabilities=spec.capabilities,
            resource_requirement=spec.resource_requirement,
            sandbox_config=spec.sandbox,
            **adapter_kwargs,
        )
        return adapter

    def _resolve_role_class(self, spec: RoleAdapterSpec):
        if spec.class_path:
            module_name, class_name = spec.class_path.rsplit(".", 1)
            module = importlib.import_module(module_name)
            return getattr(module, class_name)
        try:
            return registry.get("roles", spec.role_type)
        except KeyError as exc:
            raise KeyError(
                f"RoleAdapter '{spec.adapter_id}' must declare class_path or reference a registered role_type"
            ) from exc

    # ------------------------------------------------------------------
    # Bulk helpers
    # ------------------------------------------------------------------
    def materialize_role_adapters(
        self,
        config: PipelineConfig,
        backends: Optional[Dict[str, Any]] = None,
        prompts: Optional[Dict[str, PromptTemplateAsset]] = None,
    ) -> Dict[str, Any]:
        """Instantiate all role adapters declared in the pipeline config."""

        instances: Dict[str, Any] = {}
        for spec in config.role_adapters:
            instances[spec.adapter_id] = self.resolve_role_adapter(spec, backends=backends, prompts=prompts)
        return instances

    def materialize_backends(self, config: PipelineConfig) -> Dict[str, Any]:
        """Instantiate backend objects declared at the pipeline level."""

        from gage_eval.role.model.backends.builder import build_backend  # delayed import to avoid heavy deps

        instances: Dict[str, Any] = {}
        for spec in config.backends:
            instances[spec.backend_id] = build_backend({"type": spec.type, "config": spec.config})
        return instances

    def materialize_prompts(self, config: PipelineConfig) -> Dict[str, PromptTemplateAsset]:
        prompts: Dict[str, PromptTemplateAsset] = {}
        for spec in config.prompts:
            asset = PromptTemplateAsset(
                prompt_id=spec.prompt_id,
                renderer_type=spec.renderer,
                template=spec.template,
                default_args=spec.params,
            )
            prompts[spec.prompt_id] = asset
            registry.register(
                "prompts",
                spec.prompt_id,
                asset,
                desc=f"Prompt asset '{spec.prompt_id}' ({spec.renderer})",
                renderer=spec.renderer,
                has_template=bool(spec.template),
            )
        return prompts

    def materialize_datasets(self, config: PipelineConfig, *, trace=None) -> Dict[str, Any]:
        datasets: Dict[str, Any] = {}
        for spec in config.datasets:
            datasets[spec.dataset_id] = self.resolve_dataset(spec, trace=trace)
        return datasets

    def materialize_models(self, config: PipelineConfig) -> Dict[str, Any]:
        from gage_eval.assets.models import clear_model_store
        from gage_eval.assets.models.manager import resolve_model

        clear_model_store()
        handles: Dict[str, Any] = {}
        for spec in config.models:
            handle = resolve_model(spec)
            handles[spec.model_id] = handle
        return handles

    def materialize_metrics(self, config: PipelineConfig) -> Dict[str, Any]:
        metric_registry = MetricRegistry()
        metrics: Dict[str, Any] = {}
        for spec in config.metrics:
            metrics[spec.metric_id] = metric_registry.build_metric(spec)
        return metrics

    def _build_inline_backend(self, spec: Dict[str, Any]) -> Any:
        """Instantiate a backend declared inline within a RoleAdapter entry."""

        from gage_eval.role.model.backends.builder import build_backend  # local import to avoid heavy deps

        backend_payload: Dict[str, Any] = dict(spec)
        backend_type = backend_payload.get("type")
        if not backend_type:
            raise ValueError("Inline backend must declare field 'type'")
        backend_payload.setdefault("config", backend_payload.get("config", {}) or {})
        return build_backend(backend_payload)

    # ------------------------------------------------------------------
    # Convenience utilities
    # ------------------------------------------------------------------
