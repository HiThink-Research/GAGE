"""Core configuration objects for the gage-eval framework.

The structures below intentionally mirror the blueprint documented in
ARCHITECTURE_DESIGN.md. They are implemented as dataclasses so that
all downstream components can rely on type-stable, immutable inputs.

The module is inspired by OpenAI Evals' Registry objects and the
configuration dataclasses inside llm-eval-master's `config/` tree,
but adapted to the new PipelineConfig -> BuiltinPipeline flow.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

from gage_eval.config.schema import SchemaValidationError, normalize_pipeline_payload


@dataclass(frozen=True)
class BackendSpec:
    """Describes a reusable backend instance that adapters can reference."""

    backend_id: str
    type: str
    config: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class RoleAdapterSpec:
    """Declarative description of a role adapter.

    The spec is intentionally generic so that it can represent classic
    llm-eval EngineBackends, LMMS-style multi-modal adapters, or even
    future Ray-based runtime adapters without the rest of the system
    having to change. Every field is pure data; initialization logic
    lives in :mod:`gage_eval.config.registry` and
    :mod:`gage_eval.role.role_manager`.
    """

    adapter_id: str
    role_type: str
    class_path: Optional[str] = None
    parallel: Dict[str, Any] = field(default_factory=dict)
    sandbox: Dict[str, Any] = field(default_factory=dict)
    resource_requirement: Dict[str, Any] = field(default_factory=dict)
    capabilities: Sequence[str] = field(default_factory=tuple)
    params: Dict[str, Any] = field(default_factory=dict)
    backend_id: Optional[str] = None
    backend: Optional[Dict[str, Any]] = None
    prompt_id: Optional[str] = None
    prompt_params: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class PromptTemplateSpec:
    """Reusable prompt asset referenced by RoleAdapters."""

    prompt_id: str
    renderer: str
    template: Optional[str] = None
    params: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class DatasetSpec:
    """Describes a dataset entry that the ConfigRegistry can resolve."""

    dataset_id: str
    loader: str
    hub: Optional[str] = None
    hub_params: Dict[str, Any] = field(default_factory=dict)
    preprocess_chain: Sequence[Dict[str, Any]] = field(default_factory=tuple)
    params: Dict[str, Any] = field(default_factory=dict)
    schema: Optional[Dict[str, Any]] = None


@dataclass(frozen=True)
class ModelSpec:
    """Describes a model asset that can be resolved via ModelRegistry."""

    model_id: str
    source: Optional[str] = None
    hub: Optional[str] = None
    hub_params: Dict[str, Any] = field(default_factory=dict)
    params: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class MetricSpec:
    """Defines a metric implementation and aggregation policy.

    ``implementation`` 可以是注册表名称（如 ``exact_match``），也可以是
    ``module.submodule:ClassName`` 路径，便于直接复用自定义 Metric。
    """

    metric_id: str
    implementation: str
    aggregation: Optional[str] = None
    params: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class BuiltinPipelineSpec:
    """Reference to a built-in pipeline template."""

    pipeline_id: str
    params: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class CustomPipelineStep:
    """Represents a single custom pipeline step declared inline."""

    step_type: str
    adapter_id: Optional[str] = None
    params: Dict[str, Any] = field(default_factory=dict)

    def get(self, key: str, default: Any = None) -> Any:
        if key in {"step", "step_type"}:
            return self.step_type
        if key in {"role_ref", "adapter_id"}:
            return self.adapter_id
        if key in {"args", "params"}:
            return self.params
        return default


@dataclass(frozen=True)
class CustomPipelineSpec:
    """A collection of inline steps forming a bespoke pipeline."""

    steps: Sequence[CustomPipelineStep]


@dataclass(frozen=True)
class TaskSpec:
    """Declarative description for running a dataset as an evaluation task."""

    task_id: str
    dataset_id: str
    steps: Sequence[CustomPipelineStep] = field(default_factory=tuple)
    metric_overrides: Sequence[MetricSpec] = field(default_factory=tuple)
    reporting: Dict[str, Any] = field(default_factory=dict)
    max_samples: Optional[int] = None
    shuffle: Optional[bool] = None
    shuffle_seed: Optional[int] = None
    concurrency: Optional[int] = None
    prefetch_factor: Optional[int] = None
    max_inflight: Optional[int] = None


@dataclass(frozen=True)
class PipelineConfig:
    """Root configuration entity consumed by the CLI/Portal/SDK.

    The structure aligns with the architecture doc: Builtin pipelines
    and custom steps can co-exist. Downstream modules are expected to
    interact with this dataclass instead of raw dictionaries so that
    validation and schema evolution remain centralized.
    """

    metadata: Dict[str, Any] = field(default_factory=dict)
    builtin: Optional[BuiltinPipelineSpec] = None
    custom: Optional[CustomPipelineSpec] = None
    datasets: Sequence[DatasetSpec] = field(default_factory=tuple)
    models: Sequence[ModelSpec] = field(default_factory=tuple)
    backends: Sequence[BackendSpec] = field(default_factory=tuple)
    prompts: Sequence[PromptTemplateSpec] = field(default_factory=tuple)
    role_adapters: Sequence[RoleAdapterSpec] = field(default_factory=tuple)
    metrics: Sequence[MetricSpec] = field(default_factory=tuple)
    tasks: Sequence[TaskSpec] = field(default_factory=tuple)
    observability: Dict[str, Any] = field(default_factory=dict)

    @property
    def pipeline_id(self) -> Optional[str]:
        if self.builtin:
            return self.builtin.pipeline_id
        return self.metadata.get("name")

    @staticmethod
    def from_dict(payload: Dict[str, Any]) -> "PipelineConfig":
        """Create a PipelineConfig from an untyped dictionary.

        The helper mirrors llm-eval's YAML loaders but keeps the result
        immutable. Only a subset of validations are performed here;
        deeper checks belong to :class:`gage_eval.config.registry.ConfigRegistry`.
        """

        try:
            normalized = normalize_pipeline_payload(payload)
        except SchemaValidationError as exc:
            raise ValueError(str(exc)) from exc

        metadata = normalized.get("metadata", {})
        builtin = None
        if normalized.get("builtin"):
            builtin_entry = normalized["builtin"]
            builtin = BuiltinPipelineSpec(
                pipeline_id=builtin_entry.get("pipeline_id"),
                params=builtin_entry.get("params") or builtin_entry.get("args", {}),
            )

        custom = None
        if normalized.get("custom"):
            steps = []
            for step in normalized["custom"].get("steps", []):
                steps.append(
                    CustomPipelineStep(
                        step_type=step.get("step"),
                        adapter_id=step.get("adapter_id") or step.get("role_ref"),
                        params=step.get("params") or {},
                    )
                )
            custom = CustomPipelineSpec(steps=tuple(steps))

        models = tuple(
            ModelSpec(
                model_id=item.get("model_id"),
                source=item.get("source"),
                hub=item.get("hub"),
                hub_params=item.get("hub_params") or item.get("hub_args", {}),
                params=item.get("params") or {},
            )
            for item in normalized.get("models", [])
        )

        datasets = tuple(
            DatasetSpec(
                dataset_id=item.get("dataset_id"),
                loader=item.get("loader"),
                hub=item.get("hub"),
                hub_params=item.get("hub_params") or item.get("hub_args", {}),
                preprocess_chain=tuple(item.get("preprocess_chain", [])),
                params=item.get("params") or {},
                schema=item.get("schema"),
            )
            for item in normalized.get("datasets", [])
        )

        backends = tuple(
            BackendSpec(
                backend_id=item.get("backend_id"),
                type=item.get("type"),
                config=item.get("config", {}),
            )
            for item in normalized.get("backends", [])
        )

        prompts = tuple(
            PromptTemplateSpec(
                prompt_id=item.get("prompt_id"),
                renderer=item.get("renderer"),
                template=item.get("template"),
                params=item.get("params") or {},
            )
            for item in normalized.get("prompts", [])
        )

        role_adapters = tuple(
            RoleAdapterSpec(
                adapter_id=item.get("adapter_id"),
                role_type=item.get("role_type"),
                class_path=item.get("class_path"),
                parallel=item.get("parallel", {}),
                sandbox=item.get("sandbox", {}),
                resource_requirement=item.get("resource_requirement", {}),
                capabilities=tuple(item.get("capabilities", [])),
                params=item.get("params") or {},
                backend_id=item.get("backend_id"),
                backend=dict(item.get("backend", {})) if item.get("backend") else None,
                prompt_id=item.get("prompt_id"),
                prompt_params=item.get("prompt_params") or item.get("prompt_args", {}),
            )
            for item in normalized.get("role_adapters", [])
        )

        metrics = tuple(
            MetricSpec(
                metric_id=item.get("metric_id"),
                implementation=item.get("implementation"),
                aggregation=item.get("aggregation"),
                params=item.get("params") or {},
            )
            for item in normalized.get("metrics", [])
        )

        tasks = tuple(
            TaskSpec(
                task_id=item.get("task_id"),
                dataset_id=item.get("dataset_id") or item.get("dataset_ref"),
                steps=tuple(
                    CustomPipelineStep(
                        step_type=step.get("step"),
                        adapter_id=step.get("adapter_id") or step.get("role_ref"),
                        params=step.get("params") or {},
                    )
                    for step in item.get("steps", [])
                ),
                metric_overrides=tuple(
                    MetricSpec(
                        metric_id=metric.get("metric_id"),
                        implementation=metric.get("implementation"),
                        aggregation=metric.get("aggregation"),
                        params=metric.get("params") or {},
                    )
                    for metric in item.get("metric_overrides", [])
                ),
                reporting=item.get("reporting", {}),
                max_samples=item.get("max_samples"),
                shuffle=item.get("shuffle"),
                shuffle_seed=item.get("shuffle_seed"),
                concurrency=item.get("concurrency"),
                prefetch_factor=item.get("prefetch_factor"),
                max_inflight=item.get("max_inflight"),
            )
            for item in normalized.get("tasks", [])
        )

        observability = normalized.get("observability") or {}

        return PipelineConfig(
            metadata=metadata,
            builtin=builtin,
            custom=custom,
            datasets=datasets,
            models=models,
            backends=backends,
            prompts=prompts,
            role_adapters=role_adapters,
            metrics=metrics,
            tasks=tasks,
            observability=observability,
        )
