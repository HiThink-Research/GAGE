"""Configuration registry responsible for wiring plugins at runtime."""

from __future__ import annotations

import importlib
import os
from typing import Any, Dict, Iterable, Optional, TYPE_CHECKING

from loguru import logger

from gage_eval.config.pipeline_config import (
    BackendSpec,
    DatasetSpec,
    MetricSpec,
    ModelSpec,
    PipelineConfig,
    RoleAdapterSpec,
)
from gage_eval.registry import RuntimeAssetPlanner, import_asset_from_manifest, import_kind_from_manifest

if TYPE_CHECKING:
    from gage_eval.assets.prompts.assets import PromptTemplateAsset
    from gage_eval.metrics.registry import MetricRegistry
    from gage_eval.registry.manager import RegistryManager
    from gage_eval.registry.runtime import RuntimeRegistryContext


class ConfigRegistry:
    """Central registry that resolves datasets, role adapters and metrics.

    The class is intentionally lightweight but extensible: new
    registration surfaces can be added without touching callers. The
    implementation is reminiscent of OpenAI Evals' Registry yet adopts
    some pragmatic shortcuts from llm-eval's YAML-driven loaders.
    """

    def __init__(
        self,
        *,
        registry_view=None,
        runtime_registry_context: Optional["RuntimeRegistryContext"] = None,
        bootstrap_coordinator=None,
        asset_planner: Optional[RuntimeAssetPlanner] = None,
    ) -> None:
        """ConfigRegistry no longer stores per-kind builders; registry handles extensibility."""
        from gage_eval.registry import RegistryBootstrapCoordinator, registry

        self._registry_view = registry_view
        self._runtime_registry_context = runtime_registry_context
        self._bootstrap_coordinator = bootstrap_coordinator or RegistryBootstrapCoordinator(registry)
        self._asset_planner = asset_planner or RuntimeAssetPlanner()

    @property
    def registry_view(self):
        return self._registry_view

    @property
    def runtime_registry_context(self) -> Optional["RuntimeRegistryContext"]:
        return self._runtime_registry_context

    def with_runtime_registry_context(self, context: "RuntimeRegistryContext") -> "ConfigRegistry":
        return ConfigRegistry(
            registry_view=context.view,
            runtime_registry_context=context,
            bootstrap_coordinator=self._bootstrap_coordinator,
            asset_planner=self._asset_planner,
        )

    def prepare_runtime_registry_context(
        self,
        config: PipelineConfig,
        *,
        run_id: str,
    ) -> "RuntimeRegistryContext":
        from gage_eval.assets.prompts.assets import PromptTemplateAsset
        from gage_eval.registry import DiscoveryPolicy, RegistryOverlayAsset

        discovery_plan = self._asset_planner.build_plan(config)
        overlays = [
            RegistryOverlayAsset(
                kind="prompts",
                name=spec.prompt_id,
                obj=PromptTemplateAsset(
                    prompt_id=spec.prompt_id,
                    renderer_type=spec.renderer,
                    template=spec.template,
                    default_args=spec.params,
                ),
                desc=f"Prompt asset '{spec.prompt_id}' ({spec.renderer})",
                extra={
                    "renderer": spec.renderer,
                    "has_template": bool(spec.template),
                },
            )
            for spec in config.prompts
        ]
        return self._bootstrap_coordinator.prepare_runtime_context(
            run_id=run_id,
            discovery_plan=discovery_plan,
            required_packages=_collect_runtime_registry_packages(config),
            overlay_assets=overlays,
            policy=DiscoveryPolicy(
                mode=_resolve_discovery_mode(),
                strategy=_resolve_discovery_strategy(),
                freeze_strict=_resolve_freeze_strict(),
                dev_auto_refresh=_resolve_dev_auto_refresh(),
            ),
        )

    def _registry_lookup(self):
        if self._registry_view is not None:
            return self._registry_view
        from gage_eval.registry import registry

        return registry

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
        lookup = self._registry_lookup()

        hub_params = dict(spec.hub_params or spec.params.get("hub_params", {}))
        try:
            hub_cls = lookup.get("dataset_hubs", hub_name)
        except KeyError:
            if self._registry_view is not None:
                raise
            import_asset_from_manifest(
                "dataset_hubs",
                str(hub_name),
                registry=lookup,
                source=f"dataset:{spec.dataset_id}.hub",
            )
            hub_cls = lookup.get("dataset_hubs", hub_name)
        hub = hub_cls(spec, hub_args=hub_params)
        hub_handle = hub.resolve()
        try:
            loader_cls = lookup.get("dataset_loaders", loader_name)
        except KeyError:
            if self._registry_view is not None:
                raise
            import_asset_from_manifest(
                "dataset_loaders",
                str(loader_name),
                registry=lookup,
                source=f"dataset:{spec.dataset_id}.loader",
            )
            loader_cls = lookup.get("dataset_loaders", loader_name)
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
        lookup = self._registry_lookup()
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
        if spec.role_type in {"helper_model", "context_provider", "judge_extend", "arena"}:
            adapter_kwargs.setdefault("registry_view", lookup)
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

        lookup = self._registry_lookup()
        try:
            return lookup.get("roles", spec.role_type)
        except KeyError as exc:
            if self._registry_view is None:
                import_asset_from_manifest(
                    "roles",
                    spec.role_type,
                    registry=lookup,
                    source=f"role:{spec.adapter_id}",
                )
                try:
                    return lookup.get("roles", spec.role_type)
                except KeyError:
                    pass
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

        lookup = self._registry_lookup()
        instances: Dict[str, Any] = {}
        for spec in config.backends:
            instances[spec.backend_id] = build_backend(
                {"type": spec.type, "config": spec.config},
                registry_view=lookup,
            )
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

        lookup = self._registry_lookup()
        prompts: Dict[str, PromptTemplateAsset] = {}
        if self._registry_view is None:
            import_kind_from_manifest("prompts", registry=lookup)
        for entry in lookup.list("prompts"):
            asset = lookup.get("prompts", entry.name)
            if isinstance(asset, PromptTemplateAsset):
                prompts[entry.name] = asset
        for spec in config.prompts:
            asset = PromptTemplateAsset(
                prompt_id=spec.prompt_id,
                renderer_type=spec.renderer,
                template=spec.template,
                default_args=spec.params,
            )
            prompts.setdefault(spec.prompt_id, asset)
        return prompts

    def materialize_datasets(self, config: PipelineConfig, *, trace=None) -> Dict[str, Any]:
        datasets: Dict[str, Any] = {}
        for spec in config.datasets:
            datasets[spec.dataset_id] = self.resolve_dataset(spec, trace=trace)
        return datasets

    def materialize_models(self, config: PipelineConfig) -> Dict[str, Any]:
        if not config.models:
            return {}
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

        metric_registry = MetricRegistry(registry_view=self._registry_lookup())
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
        return build_backend(backend_payload, registry_view=self._registry_lookup())

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


def _env_truthy(value: Optional[str]) -> bool:
    return str(value or "").strip().lower() in {"1", "true", "yes", "on"}


def _resolve_discovery_mode() -> str:
    if "GAGE_EVAL_DISCOVERY_STRICT" in os.environ:
        return "strict" if _env_truthy(os.environ.get("GAGE_EVAL_DISCOVERY_STRICT")) else "warn"
    return "strict" if _env_truthy(os.environ.get("CI")) else "warn"


def _resolve_discovery_strategy() -> str:
    strategy = str(os.environ.get("GAGE_EVAL_DISCOVERY_MODE") or "manifest").strip().lower()
    if strategy in {"legacy", "hybrid", "manifest"}:
        return strategy
    return "manifest"


def _resolve_dev_auto_refresh() -> bool:
    return _env_truthy(os.environ.get("GAGE_EVAL_DISCOVERY_DEV_AUTO_REFRESH"))


def _resolve_freeze_strict() -> bool:
    if "GAGE_EVAL_REGISTRY_FREEZE_STRICT" in os.environ:
        return _env_truthy(os.environ.get("GAGE_EVAL_REGISTRY_FREEZE_STRICT"))
    return True


def _collect_runtime_registry_packages(config: PipelineConfig) -> Dict[str, Iterable[str]]:
    packages: Dict[str, tuple[str, ...]] = {
        "pipeline_steps": ("gage_eval.pipeline.steps",),
        "summary_generators": ("gage_eval.reporting.summary_generators",),
    }
    if any(spec.role_type for spec in config.role_adapters):
        packages["roles"] = ("gage_eval.role.adapters", "gage_eval.role.toolchain")
    if config.prompts or any(spec.prompt_id for spec in config.role_adapters):
        packages["prompts"] = ("gage_eval.assets.prompts.catalog",)
    if any(
        spec.role_type == "helper_model" and spec.params.get("implementation")
        for spec in config.role_adapters
    ):
        packages["helper_impls"] = ("gage_eval.role.helper",)
    if any(
        spec.role_type == "context_provider" and spec.params.get("implementation")
        for spec in config.role_adapters
    ):
        packages["context_impls"] = ("gage_eval.role.context",)
    if any(
        spec.role_type == "judge_extend" and spec.params.get("implementation")
        for spec in config.role_adapters
    ):
        packages["judge_impls"] = ("gage_eval.role.judge",)
    if any(_may_require_gamearena_runtime_packages(spec) for spec in config.role_adapters):
        packages["game_kits"] = ("gage_eval.game_kits.registry",)
        packages["scheduler_bindings"] = ("gage_eval.role.arena.schedulers.specs",)
        packages["support_workflows"] = ("gage_eval.role.arena.support.specs",)
    return packages


def _may_require_gamearena_runtime_packages(spec: RoleAdapterSpec) -> bool:
    return spec.role_type == "arena"
