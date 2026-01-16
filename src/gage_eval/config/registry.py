"""Configuration registry responsible for wiring plugins at runtime."""

from __future__ import annotations

import importlib
from typing import Any, Dict, Optional, TYPE_CHECKING

from loguru import logger

from gage_eval.config.pipeline_config import (
    BackendSpec,
    DatasetSpec,
    MetricSpec,
    ModelSpec,
    PipelineConfig,
    RoleAdapterSpec,
)

if TYPE_CHECKING:
    from gage_eval.assets.prompts.assets import PromptTemplateAsset
    from gage_eval.metrics.registry import MetricRegistry
    from gage_eval.registry.manager import RegistryManager


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
        from gage_eval.registry import registry
        
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
        agent_backends: Optional[Dict[str, Any]] = None,
        sandbox_profiles: Optional[Dict[str, Dict[str, Any]]] = None,
        mcp_clients: Optional[Dict[str, Any]] = None,
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
        if spec.prompt_id:
            if not prompts or spec.prompt_id not in prompts:
                raise KeyError(
                    f"Prompt '{spec.prompt_id}' referenced by adapter '{spec.adapter_id}' is not defined"
                )
            prompt_renderer = prompts[spec.prompt_id].instantiate(spec.prompt_params)
            adapter_kwargs.setdefault("prompt_renderer", prompt_renderer)
        agent_backend_obj: Any = None
        if spec.agent_backend is not None:
            agent_backend_obj = self._build_inline_agent_backend(spec.agent_backend)
        elif spec.agent_backend_id:
            if not agent_backends or spec.agent_backend_id not in agent_backends:
                raise KeyError(
                    f"Agent backend '{spec.agent_backend_id}' referenced by adapter '{spec.adapter_id}' is not defined"
                )
            agent_backend_obj = agent_backends[spec.agent_backend_id]
        if agent_backend_obj is not None:
            adapter_kwargs.setdefault("agent_backend", agent_backend_obj)
        if spec.mcp_client_id:
            adapter_kwargs.setdefault("mcp_client_id", spec.mcp_client_id)
            if mcp_clients and spec.mcp_client_id in mcp_clients:
                adapter_kwargs.setdefault("mcp_client", mcp_clients[spec.mcp_client_id])
        if spec.role_type == "dut_agent" and sandbox_profiles is not None:
            adapter_kwargs.setdefault("sandbox_profiles", sandbox_profiles)
        if spec.role_type == "dut_agent" and mcp_clients is not None:
            adapter_kwargs.setdefault("mcp_clients", mcp_clients)
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
        from gage_eval.registry import registry
        
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
        agent_backends: Optional[Dict[str, Any]] = None,
        sandbox_profiles: Optional[Dict[str, Dict[str, Any]]] = None,
        mcp_clients: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Instantiate all role adapters declared in the pipeline config."""

        instances: Dict[str, Any] = {}
        for spec in config.role_adapters:
            instances[spec.adapter_id] = self.resolve_role_adapter(
                spec,
                backends=backends,
                prompts=prompts,
                agent_backends=agent_backends,
                sandbox_profiles=sandbox_profiles,
                mcp_clients=mcp_clients,
            )
        return instances

    def materialize_backends(self, config: PipelineConfig) -> Dict[str, Any]:
        """Instantiate backend objects declared at the pipeline level."""

        from gage_eval.role.model.backends.builder import build_backend  # delayed import to avoid heavy deps

        instances: Dict[str, Any] = {}
        for spec in config.backends:
            instances[spec.backend_id] = build_backend({"type": spec.type, "config": spec.config})
        return instances

    def materialize_agent_backends(
        self,
        config: PipelineConfig,
        *,
        backends: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Instantiate agent backend objects declared at the pipeline level."""

        from gage_eval.role.agent.backends import build_agent_backend

        instances: Dict[str, Any] = {}
        for spec in config.agent_backends:
            config_payload = dict(spec.config or {})
            backend_id = spec.backend_id or config_payload.pop("backend_id", None)
            if spec.backend_id:
                config_payload.pop("backend_id", None)
            if spec.type == "model_backend" and backend_id:
                if not backends or backend_id not in backends:
                    raise KeyError(
                        f"Agent backend '{spec.agent_backend_id}' references unknown backend '{backend_id}'"
                    )
                config_payload["backend"] = backends[backend_id]
            instances[spec.agent_backend_id] = build_agent_backend(
                {"type": spec.type, "config": config_payload}
            )
        return instances

    def materialize_sandbox_profiles(self, config: PipelineConfig) -> Dict[str, Dict[str, Any]]:
        profiles: Dict[str, Dict[str, Any]] = {}
        for spec in config.sandbox_profiles:
            profiles[spec.sandbox_id] = spec.to_dict()
        return profiles

    def materialize_mcp_clients(self, config: PipelineConfig) -> Dict[str, Any]:
        """Instantiate MCP clients declared at the pipeline level."""

        instances: Dict[str, Any] = {}
        for spec in config.mcp_clients:
            client_cls = _resolve_mcp_client_class(spec.transport)
            instances[spec.mcp_client_id] = client_cls(
                mcp_client_id=spec.mcp_client_id,
                transport=spec.transport,
                endpoint=spec.endpoint,
                timeout_s=spec.timeout_s,
                allowlist=list(spec.allowlist or ()),
                params=spec.params,
            )
        return instances

    def materialize_prompts(self, config: PipelineConfig) -> Dict[str, PromptTemplateAsset]:
        from gage_eval.assets.prompts.assets import PromptTemplateAsset
        from gage_eval.registry import registry

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
        from gage_eval.metrics.registry import MetricRegistry

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

    def _build_inline_agent_backend(self, spec: Dict[str, Any]) -> Any:
        from gage_eval.role.agent.backends import build_agent_backend

        backend_payload: Dict[str, Any] = dict(spec)
        backend_type = backend_payload.get("type")
        if not backend_type:
            raise ValueError("Inline agent backend must declare field 'type'")
        backend_payload.setdefault("config", backend_payload.get("config", {}) or {})
        return build_agent_backend(backend_payload)

    # ------------------------------------------------------------------
    # Convenience utilities
    # ------------------------------------------------------------------


def _resolve_mcp_client_class(transport: Optional[str]) -> Any:
    if transport == "streamable_http":
        from gage_eval.sandbox.integrations.appworld.mcp_client import AppWorldStreamableMcpClient

        return AppWorldStreamableMcpClient
    from gage_eval.mcp import McpClient

    return McpClient
