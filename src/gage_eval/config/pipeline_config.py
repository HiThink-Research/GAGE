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

from gage_eval.observability.logger import ObservableLogger

_logger = ObservableLogger()


@dataclass(frozen=True)
class BackendSpec:
    """Describes a reusable backend instance that adapters can reference."""

    backend_id: str
    type: str
    config: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class AgentBackendSpec:
    """Describes a reusable agent backend instance."""

    agent_backend_id: str
    type: str
    config: Dict[str, Any] = field(default_factory=dict)
    backend_id: Optional[str] = None


@dataclass(frozen=True)
class SandboxProfileSpec:
    """Describes a reusable sandbox profile template."""

    sandbox_id: str
    runtime: Optional[str] = None
    image: Optional[str] = None
    resources: Dict[str, Any] = field(default_factory=dict)
    runtime_configs: Dict[str, Any] = field(default_factory=dict)
    params: Dict[str, Any] = field(default_factory=dict)
    extra: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        payload = dict(self.extra)
        payload.update(
            {
                "sandbox_id": self.sandbox_id,
                "runtime": self.runtime,
                "image": self.image,
                "resources": self.resources,
                "runtime_configs": self.runtime_configs,
            }
        )
        if self.params:
            payload.update(self.params)
        return {k: v for k, v in payload.items() if v is not None}


@dataclass(frozen=True)
class McpClientSpec:
    """Describes an MCP client definition."""

    mcp_client_id: str
    transport: Optional[str] = None
    endpoint: Optional[str] = None
    timeout_s: Optional[int] = None
    allowlist: Sequence[str] = field(default_factory=tuple)
    params: Dict[str, Any] = field(default_factory=dict)


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
    agent_backend_id: Optional[str] = None
    agent_backend: Optional[Dict[str, Any]] = None
    agent_runtime_id: Optional[str] = None
    compat_runtime_id: Optional[str] = None
    mcp_client_id: Optional[str] = None
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

    `implementation` can be either a registry name (for example: `exact_match`)
    or a `module.submodule:ClassName` import path, which enables reusing a
    custom Metric implementation without adding a new registry entry.
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
    shuffle_strategy: Optional[str] = None
    shuffle_small_dataset_threshold: Optional[int] = None
    keep_shuffle_artifacts: Optional[bool] = None
    concurrency: Optional[int] = None
    prefetch_factor: Optional[int] = None
    max_inflight: Optional[int] = None
    failure_policy: Optional[str] = None
    metric_concurrency: Optional[int] = None
    report_partial_on_failure: Optional[bool] = None
    support_payload_policy: Dict[str, Any] = field(default_factory=dict)


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
    agent_backends: Sequence[AgentBackendSpec] = field(default_factory=tuple)
    sandbox_profiles: Sequence[SandboxProfileSpec] = field(default_factory=tuple)
    mcp_clients: Sequence[McpClientSpec] = field(default_factory=tuple)
    prompts: Sequence[PromptTemplateSpec] = field(default_factory=tuple)
    role_adapters: Sequence[RoleAdapterSpec] = field(default_factory=tuple)
    metrics: Sequence[MetricSpec] = field(default_factory=tuple)
    tasks: Sequence[TaskSpec] = field(default_factory=tuple)
    summary_generators: Sequence[str] = field(default_factory=tuple)
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
        from gage_eval.config.pipeline_builder import PipelineConfigBuilder

        return (
            PipelineConfigBuilder(payload)
            .normalize()
            .build_root()
            .build_assets()
            .build_role_adapters()
            .build_metrics()
            .build_tasks()
            .build()
        )


def _normalize_summary_generator_entry(entry: Any) -> str:
    if isinstance(entry, str):
        return entry
    if isinstance(entry, dict):
        name = entry.get("generator_id") or entry.get("name")
        if not name:
            raise ValueError(f"Summary generator entry missing name: {entry!r}")
        return str(name)
    raise ValueError(f"Unsupported summary generator entry: {entry!r}")


def _strip_known_keys(item: Dict[str, Any], keys: set[str]) -> Dict[str, Any]:
    return {k: v for k, v in item.items() if k not in keys}


def _normalize_metric_entry(entry: Any) -> Dict[str, Any]:
    """Parse metric entries supporting string/KV shortcuts or full dict."""

    if isinstance(entry, str):
        # Function-style sugar: exact_match(a=1,b=2)
        parsed = _parse_fnstyle_metric(entry)
        if parsed:
            return parsed
        return {
            "metric_id": entry,
            "implementation": entry,
            "aggregation": None,
            "params": {},
        }

    if isinstance(entry, dict):
        # KV sugar: {exact_match: {case_sensitive: true}}
        if len(entry) == 1 and "metric_id" not in entry and "implementation" not in entry:
            metric_id, payload = next(iter(entry.items()))
            params: Dict[str, Any] = {}
            aggregation = None
            if isinstance(payload, dict):
                params = dict(payload)
                aggregation = params.pop("aggregation", None)
            return {
                "metric_id": metric_id,
                "implementation": metric_id,
                "aggregation": aggregation,
                "params": params,
            }
        metric_id = entry.get("metric_id") or entry.get("implementation")
        impl = entry.get("implementation") or metric_id
        return {
            "metric_id": metric_id,
            "implementation": impl,
            "aggregation": entry.get("aggregation"),
            "params": entry.get("params") or {},
        }

    raise ValueError(f"Unsupported metric entry format: {entry!r}")


def _coerce_value(raw: str) -> Any:
    lowered = raw.lower()
    if lowered == "true":
        return True
    if lowered == "false":
        return False
    if lowered == "none" or lowered == "null":
        return None
    try:
        return int(raw)
    except ValueError:
        pass
    try:
        return float(raw)
    except ValueError:
        pass
    # Strip paired quotes.
    if (raw.startswith('"') and raw.endswith('"')) or (raw.startswith("'") and raw.endswith("'")):
        return raw[1:-1]
    return raw


def _parse_fnstyle_metric(entry: str) -> Optional[Dict[str, Any]]:
    if "(" not in entry or not entry.endswith(")"):
        return None
    name, rest = entry.split("(", 1)
    name = name.strip()
    if not name:
        raise ValueError(f"Invalid metric entry: {entry!r}")
    args_str = rest[:-1].strip()  # drop trailing ')'
    params: Dict[str, Any] = {}
    if args_str:
        for token in args_str.split(","):
            token = token.strip()
            if not token:
                continue
            if "=" not in token:
                raise ValueError(f"Invalid metric param token '{token}' in entry '{entry}'")
            key, value = token.split("=", 1)
            params[key.strip()] = _coerce_value(value.strip())
    aggregation = params.pop("aggregation", None)
    _logger.debug("metric_sugar", "Parsed fnstyle metric id={} params={}", name, params)
    return {
        "metric_id": name,
        "implementation": name,
        "aggregation": aggregation,
        "params": params,
    }
