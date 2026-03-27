"""Runtime asset planning for manifest-first discovery."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, TYPE_CHECKING

if TYPE_CHECKING:
    from gage_eval.config.pipeline_config import PipelineConfig


@dataclass(frozen=True, slots=True)
class DiscoveryRequest:
    """Single asset request derived from runtime configuration."""

    kind: str
    name: str
    source: str


@dataclass(frozen=True, slots=True)
class DiscoveryPlan:
    """Normalized runtime discovery requests."""

    requests: tuple[DiscoveryRequest, ...]
    eager_kinds: tuple[str, ...] = ()

    def requests_for_kind(self, kind: str) -> tuple[DiscoveryRequest, ...]:
        lookup_kind = str(kind).strip()
        return tuple(request for request in self.requests if request.kind == lookup_kind)


class RuntimeAssetPlanner:
    """Collects manifest-backed asset requirements from PipelineConfig."""

    _EAGER_KINDS = ("pipeline_steps", "summary_generators")

    def build_plan(self, config: "PipelineConfig") -> DiscoveryPlan:
        collector = _PlanCollector(eager_kinds=self._EAGER_KINDS)
        self._collect_role_adapter_requests(config, collector)
        self._collect_prompt_requests(config, collector)
        self._collect_dataset_requests(config, collector)
        self._collect_model_requests(config, collector)
        self._collect_backend_requests(config, collector)
        self._collect_metric_requests(config, collector)
        self._collect_pipeline_step_requests(config, collector)
        self._collect_observability_requests(config, collector)
        return collector.build()

    def _collect_role_adapter_requests(self, config: "PipelineConfig", collector: "_PlanCollector") -> None:
        for spec in config.role_adapters:
            collector.add("roles", spec.role_type, source=f"role_adapters:{spec.adapter_id}")
            if spec.prompt_id:
                collector.add("prompts", spec.prompt_id, source=f"role_adapters:{spec.adapter_id}.prompt_id")
            self._collect_role_extension_requests(spec, collector)
            if spec.role_type == "arena":
                self._collect_arena_requests(spec.params or {}, adapter_id=spec.adapter_id, collector=collector)

    def _collect_role_extension_requests(self, spec: Any, collector: "_PlanCollector") -> None:
        impl_kind_by_role = {
            "helper_model": "helper_impls",
            "context_provider": "context_impls",
            "judge_extend": "judge_impls",
        }
        impl_kind = impl_kind_by_role.get(spec.role_type)
        if impl_kind is None:
            return
        collector.add(impl_kind, spec.params.get("implementation"), source=f"role_adapters:{spec.adapter_id}")

    def _collect_arena_requests(
        self,
        params: dict[str, Any],
        *,
        adapter_id: str,
        collector: "_PlanCollector",
    ) -> None:
        collector.add(
            "game_kits",
            params.get("game_kit"),
            source=f"arena:{adapter_id}.game_kit",
        )

    def _collect_prompt_requests(self, config: "PipelineConfig", collector: "_PlanCollector") -> None:
        for spec in config.prompts:
            collector.add("prompts", spec.prompt_id, source=f"prompts:{spec.prompt_id}")

    def _collect_dataset_requests(self, config: "PipelineConfig", collector: "_PlanCollector") -> None:
        for spec in config.datasets:
            collector.add(
                "dataset_loaders",
                spec.loader or spec.params.get("loader"),
                source=f"datasets:{spec.dataset_id}.loader",
            )
            collector.add(
                "dataset_hubs",
                _resolve_dataset_hub_name(spec.loader, spec.hub, spec.params),
                source=f"datasets:{spec.dataset_id}.hub",
            )
            collector.add("bundles", spec.params.get("bundle"), source=f"datasets:{spec.dataset_id}.bundle")
            collector.add(
                "dataset_preprocessors",
                spec.params.get("preprocess"),
                source=f"datasets:{spec.dataset_id}.preprocess",
            )
            for index, step in enumerate(spec.preprocess_chain or ()):
                if not isinstance(step, dict):
                    continue
                collector.add(
                    "dataset_preprocessors",
                    step.get("name") or step.get("type") or step.get("preprocess"),
                    source=f"datasets:{spec.dataset_id}.preprocess_chain[{index}]",
                )

    def _collect_model_requests(self, config: "PipelineConfig", collector: "_PlanCollector") -> None:
        for spec in config.models:
            collector.add("model_hubs", spec.hub or spec.source or "huggingface", source=f"models:{spec.model_id}.hub")

    def _collect_backend_requests(self, config: "PipelineConfig", collector: "_PlanCollector") -> None:
        for spec in config.backends:
            collector.add("backends", spec.type, source=f"backends:{spec.backend_id}")
        for spec in config.role_adapters:
            if isinstance(spec.backend, dict):
                collector.add("backends", spec.backend.get("type"), source=f"role_adapters:{spec.adapter_id}.backend")

    def _collect_metric_requests(self, config: "PipelineConfig", collector: "_PlanCollector") -> None:
        for spec in config.metrics:
            implementation = str(spec.implementation or spec.metric_id or "").strip()
            if implementation and ":" not in implementation and "." not in implementation:
                collector.add("metrics", implementation, source=f"metrics:{spec.metric_id}")

    def _collect_pipeline_step_requests(self, config: "PipelineConfig", collector: "_PlanCollector") -> None:
        for step_type in _iter_step_types(config):
            collector.add("pipeline_steps", step_type, source=f"step:{step_type}")
        for generator in config.summary_generators or ():
            collector.add("summary_generators", generator, source="summary_generators")

    def _collect_observability_requests(self, config: "PipelineConfig", collector: "_PlanCollector") -> None:
        observability = dict(config.observability or {})
        collector.add(
            "observability_plugins",
            observability.get("plugin") or observability.get("plugin_id"),
            source="observability.plugin",
        )
        collector.add(
            "reporting_sinks",
            observability.get("sink") or observability.get("sink_id"),
            source="observability.sink",
        )


class _PlanCollector:
    """Mutable accumulator for DiscoveryPlan construction."""

    def __init__(self, *, eager_kinds: Iterable[str]) -> None:
        self._requests: list[DiscoveryRequest] = []
        self._seen: set[tuple[str, str]] = set()
        self._eager_kinds: set[str] = set(eager_kinds)

    def add(self, kind: str, name: str | None, *, source: str) -> None:
        normalized_kind = str(kind or "").strip()
        normalized_name = str(name or "").strip()
        if not normalized_kind or not normalized_name:
            return
        key = (normalized_kind, normalized_name)
        if key in self._seen:
            return
        self._seen.add(key)
        self._requests.append(DiscoveryRequest(kind=normalized_kind, name=normalized_name, source=source))

    def add_eager_kind(self, kind: str) -> None:
        normalized_kind = str(kind or "").strip()
        if normalized_kind:
            self._eager_kinds.add(normalized_kind)

    def build(self) -> DiscoveryPlan:
        return DiscoveryPlan(requests=tuple(self._requests), eager_kinds=tuple(sorted(self._eager_kinds)))


def _resolve_dataset_hub_name(loader: str | None, hub: str | None, params: dict[str, Any]) -> str | None:
    hub_name = hub or params.get("hub")
    loader_name = str(loader or params.get("loader") or "").strip()
    if hub_name:
        return hub_name
    if loader_name not in {"hf_hub", "modelscope"}:
        return "inline"
    return None


def _iter_step_types(config: "PipelineConfig") -> Iterable[str]:
    if config.custom:
        for step in config.custom.steps or ():
            step_type = getattr(step, "step_type", None) or getattr(step, "get", lambda *_: None)("step")
            if step_type:
                yield str(step_type)
    for task in config.tasks or ():
        for step in task.steps or ():
            step_type = getattr(step, "step_type", None) or getattr(step, "get", lambda *_: None)("step")
            if step_type:
                yield str(step_type)
