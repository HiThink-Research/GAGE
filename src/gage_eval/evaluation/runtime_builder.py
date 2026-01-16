"""Helpers that bootstrap a PipelineRuntime from configuration objects."""

from __future__ import annotations

from dataclasses import dataclass
import os
import time
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set
from typing import TYPE_CHECKING
from urllib.parse import urlparse

if TYPE_CHECKING:
    from gage_eval.config.registry import ConfigRegistry

from loguru import logger
from gage_eval.config.pipeline_config import PipelineConfig, RoleAdapterSpec
from gage_eval.assets.datasets.manager import DataManager, DataSource
from gage_eval.assets.datasets.validation import SampleValidator
from gage_eval.evaluation.cache import EvalCache
from gage_eval.metrics import MetricRegistry
from gage_eval.evaluation.pipeline import PipelineFactory, PipelineRuntime
from gage_eval.evaluation.sample_loop import SampleLoop
from gage_eval.pipeline.steps.report import ReportStep
from gage_eval.evaluation.task_plan import TaskPlanSpec, build_task_plan_specs
from gage_eval.evaluation.task_planner import TaskPlanner
from gage_eval.observability.config import configure_observability, get_observability_config
from gage_eval.observability.trace import ObservabilityTrace
from gage_eval.role.resource_profile import ResourceProfile
from gage_eval.role.role_manager import RoleManager


def _record_config_metadata(config: PipelineConfig, cache_store: EvalCache) -> None:
    backend_specs = [
        {
            "backend_id": spec.backend_id,
            "type": spec.type,
            "config": spec.config,
        }
        for spec in (config.backends or [])
    ]
    if backend_specs:
        cache_store.set_metadata("backends", backend_specs)

    model_specs = [
        {
            "model_id": spec.model_id,
            "source": spec.source,
            "hub": spec.hub,
            "hub_params": spec.hub_params,
            "params": spec.params,
        }
        for spec in (config.models or [])
    ]
    if model_specs:
        cache_store.set_metadata("models", model_specs)

    role_specs = [
        {
            "adapter_id": spec.adapter_id,
            "role_type": spec.role_type,
            "backend_id": spec.backend_id,
            "backend_inline": spec.backend,
            "capabilities": list(spec.capabilities or ()),
            "prompt_id": spec.prompt_id,
        }
        for spec in (config.role_adapters or [])
    ]
    if role_specs:
        cache_store.set_metadata("role_adapters", role_specs)

    if config.summary_generators:
        cache_store.set_metadata("summary_generators", list(config.summary_generators))


def build_runtime(
    config: PipelineConfig,
    registry: "ConfigRegistry",
    resource_profile: ResourceProfile,
    *,
    dataset_id: Optional[str] = None,
    trace: Optional[ObservabilityTrace] = None,
) -> PipelineRuntime | "TaskOrchestratorRuntime":
    """Materialize a runtime wired to the new DataManager."""

    trace = trace or ObservabilityTrace()
    configure_observability(config.observability)
    logger.info(
        "Bootstrapping runtime for pipeline='{}' (datasets={}, tasks={})",
        config.pipeline_id,
        len(config.datasets or []),
        len(config.tasks or []),
    )
    cache_store = EvalCache(
        base_dir=os.environ.get("GAGE_EVAL_SAVE_DIR"),
        run_id=trace.run_id,
    )
    data_manager = DataManager()
    dataset_start = time.perf_counter()
    datasets: Dict[str, DataSource] = registry.materialize_datasets(config, trace=trace)
    dataset_elapsed = time.perf_counter() - dataset_start
    cache_store.record_timing("dataset_materialization_s", dataset_elapsed)
    registry.materialize_models(config)
    if not datasets:
        raise ValueError("PipelineConfig must declare at least one dataset")
    for source in datasets.values():
        data_manager.register_source(source, trace=trace)

    role_manager = RoleManager(resource_profile)
    factory = PipelineFactory(registry)

    if config.tasks:
        logger.info("Detected task orchestrator config with {} tasks", len(config.tasks))
        return _build_task_orchestrator_runtime(
            config=config,
            registry=registry,
            data_manager=data_manager,
            datasets=datasets,
            role_manager=role_manager,
            trace=trace,
            cache_store=cache_store,
            resource_profile=resource_profile,
        )

    return _build_single_runtime(
        config=config,
        registry=registry,
        factory=factory,
        data_manager=data_manager,
        datasets=datasets,
        resource_profile=resource_profile,
        role_manager=role_manager,
        dataset_id=dataset_id,
        trace=trace,
        cache_store=cache_store,
    )


def _build_single_runtime(
    config: PipelineConfig,
    registry: "ConfigRegistry",
    factory: PipelineFactory,
    data_manager: DataManager,
    datasets: Dict[str, DataSource],
    resource_profile: ResourceProfile,
    role_manager: RoleManager,
    dataset_id: Optional[str],
    trace: ObservabilityTrace,
    cache_store: EvalCache,
) -> PipelineRuntime:
    sandbox_profiles = registry.materialize_sandbox_profiles(config)
    selected_dataset_id = _select_dataset_id(dataset_id, config, datasets)
    selected_source = data_manager.get(selected_dataset_id)
    logger.info(
        "Selected dataset_id='{}' (streaming={}) for single runtime",
        selected_dataset_id,
        selected_source.streaming,
    )
    trace.emit(
        "runtime_bootstrap",
        {
            "datasets": list(datasets.keys()),
            "selected_dataset": selected_dataset_id,
            "streaming": selected_source.streaming,
        },
    )
    raw_samples = data_manager.iter_samples(selected_dataset_id, trace=trace)
    samples = _validate_samples(
        raw_samples,
        validator=selected_source.validator,
        dataset_id=selected_dataset_id,
        trace=trace,
    )

    concurrency = _resolve_concurrency(None, resource_profile)
    concurrency = _apply_fixed_port_guard(
        scope=f"pipeline '{config.pipeline_id or selected_dataset_id}'",
        concurrency=concurrency,
        role_bindings={spec.adapter_id: spec for spec in config.role_adapters},
        sandbox_profiles=sandbox_profiles,
    )
    role_manager.update_concurrency_hint(concurrency)
    sample_loop = SampleLoop(
        samples,
        streaming=selected_source.streaming,
        concurrency=concurrency,
        sandbox_profiles=sandbox_profiles,
    )
    task_planner = TaskPlanner()
    runtime = factory.create_runtime(
        config=config,
        role_manager=role_manager,
        sample_loop=sample_loop,
        task_planner=task_planner,
        trace=trace,
        cache_store=cache_store,
    )
    trace.emit("runtime_ready", {"selected_dataset": selected_dataset_id})
    return runtime


def _build_task_orchestrator_runtime(
    config: PipelineConfig,
    registry: "ConfigRegistry",
    data_manager: DataManager,
    datasets: Dict[str, DataSource],
    role_manager: RoleManager,
    trace: ObservabilityTrace,
    cache_store: EvalCache,
    resource_profile: ResourceProfile,
) -> "TaskOrchestratorRuntime":
    report_step = ReportStep(auto_eval_step=None, cache_store=cache_store)
    _record_config_metadata(config, cache_store)
    task_plan_specs = build_task_plan_specs(config)
    concurrency_hint = _max_concurrency_from_tasks(task_plan_specs, resource_profile)
    role_manager.update_concurrency_hint(concurrency_hint)
    _materialize_role_adapters(config, registry, role_manager)
    sandbox_profiles = registry.materialize_sandbox_profiles(config)

    task_entries = _prepare_task_entries(
        task_plans=task_plan_specs,
        config=config,
        data_manager=data_manager,
        datasets=datasets,
        trace=trace,
        cache_store=cache_store,
        resource_profile=resource_profile,
        sandbox_profiles=sandbox_profiles,
    )
    logger.info("Prepared {} task runtime entries", len(task_entries))
    trace.emit(
        "runtime_bootstrap",
        {
            "datasets": list(datasets.keys()),
            "tasks": [entry.task_id for entry in task_entries],
        },
    )
    trace.emit(
        "runtime_ready",
        {
            "tasks": [entry.task_id for entry in task_entries],
            "datasets": list(datasets.keys()),
        },
    )
    return TaskOrchestratorRuntime(task_entries, role_manager, trace, report_step)


def _validate_samples(
    raw_samples: Iterable[Dict[str, Any]],
    *,
    validator: Optional[SampleValidator],
    dataset_id: str,
    trace: Optional[ObservabilityTrace],
) -> Iterable[Dict[str, Any]]:
    for record in raw_samples:
        sample = dict(record)
        if validator:
            validated = validator.validate_envelope(
                sample,
                dataset_id=dataset_id,
                sample_id=str(sample.get("id")),
                trace=trace,
            )
            if validated is None:
                logger.debug(
                    "Validator skipped sample for dataset_id='{}'",
                    dataset_id,
                )
                continue
            sample = validated
        logger.trace("Validated sample id='{}'", sample.get("id"))
        yield sample


def _select_dataset_id(
    explicit_dataset_id: Optional[str],
    config: PipelineConfig,
    datasets: Dict[str, DataSource],
) -> str:
    if explicit_dataset_id:
        if explicit_dataset_id not in datasets:
            raise KeyError(f"Dataset '{explicit_dataset_id}' not resolved")
        return explicit_dataset_id
    if config.datasets:
        default_dataset_id = config.datasets[0].dataset_id
        if default_dataset_id in datasets:
            return default_dataset_id
    return next(iter(datasets))


def _materialize_role_adapters(
    config: PipelineConfig,
    registry: "ConfigRegistry",
    role_manager: RoleManager,
) -> None:
    backend_instances = registry.materialize_backends(config)
    agent_backend_instances = registry.materialize_agent_backends(
        config,
        backends=backend_instances,
    )
    sandbox_profiles = registry.materialize_sandbox_profiles(config)
    mcp_clients = registry.materialize_mcp_clients(config)
    prompt_assets = registry.materialize_prompts(config)
    adapters = registry.materialize_role_adapters(
        config,
        backends=backend_instances,
        prompts=prompt_assets,
        agent_backends=agent_backend_instances,
        sandbox_profiles=sandbox_profiles,
        mcp_clients=mcp_clients,
    )
    for adapter_id, adapter in adapters.items():
        role_manager.register_role_adapter(adapter_id, adapter)
        logger.debug("Registered role adapter '{}' with RoleManager", adapter_id)


@dataclass
class _TaskRuntimeEntry:
    task_id: str
    dataset_id: str
    dataset_metadata: Dict[str, Any]
    sample_loop: SampleLoop
    task_planner: TaskPlanner
    reporting: Dict[str, Any]


class TaskOrchestratorRuntime:
    """Runtime that executes multiple TaskSpec definitions within a single run."""

    def __init__(
        self,
        tasks: Sequence[_TaskRuntimeEntry],
        role_manager: RoleManager,
        trace: ObservabilityTrace,
        report_step: ReportStep,
    ) -> None:
        self._tasks = list(tasks)
        self._role_manager = role_manager
        self._trace = trace
        self._report_step = report_step
        self._wall_clock_start: Optional[float] = None
        self._shutdown_called = False

    def run(self) -> None:
        # Summaries accumulate per-task metadata so the ReportStep can emit a global view.
        summaries: List[Dict[str, Any]] = []
        total_start = time.perf_counter()
        wall_start = self._wall_clock_start or total_start
        inference_total = 0.0
        try:
            for entry in self._tasks:
                self._trace.emit(
                    "task_start",
                    {
                        "task_id": entry.task_id,
                        "dataset_id": entry.dataset_id,
                        "dataset_metadata": entry.dataset_metadata,
                    },
                )
                task_start = time.perf_counter()
                entry.sample_loop.run(
                    planner=entry.task_planner,
                    role_manager=self._role_manager,
                    trace=self._trace,
                )
                inference_total += time.perf_counter() - task_start
                auto_eval = entry.task_planner.get_auto_eval_step()
                metrics = auto_eval.aggregated_metrics() if auto_eval else []
                summary = {
                    "task_id": entry.task_id,
                    "dataset_id": entry.dataset_id,
                    "dataset_metadata": entry.dataset_metadata,
                    "metrics": metrics,
                    "sample_count": entry.sample_loop.processed_count,
                    "reporting": entry.reporting,
                }
                summaries.append(summary)
                self._trace.emit(
                    "task_end",
                    {
                        "task_id": entry.task_id,
                        "dataset_id": entry.dataset_id,
                        "sample_count": entry.sample_loop.processed_count,
                    },
                )
            metrics_payload: List[Dict[str, Any]] = []
            for summary in summaries:
                for metric in summary["metrics"]:
                    enriched = dict(metric)
                    enriched["task_id"] = summary["task_id"]
                    metrics_payload.append(enriched)
            eval_elapsed = 0.0

            def pre_write_hook() -> None:
                nonlocal eval_elapsed
                eval_elapsed = time.perf_counter() - report_start
                execution_total = time.perf_counter() - total_start
                self._report_step.record_timing("inference_s", inference_total)
                self._report_step.record_timing("evaluation_s", eval_elapsed)
                self._report_step.record_timing("execution_runtime_s", execution_total)
                dataset_time = self._report_step.get_timing("dataset_materialization_s") or 0.0
                self._report_step.record_timing("total_runtime_s", dataset_time + execution_total)
                wall_total = time.perf_counter() - wall_start
                self._report_step.record_timing("wall_runtime_s", wall_total)
                self._record_throughput_metrics(
                    sample_count=self._resolve_sample_count(),
                    wall_runtime_s=wall_total,
                    inference_s=inference_total,
                    evaluation_s=eval_elapsed,
                )
                # Merge fine-grained timings emitted via @observable_stage.
                detailed_timings = get_observability_config().snapshot_timings()
                for stage, metrics in detailed_timings.items():
                    total_s = metrics.get("total_s")
                    if total_s is None:
                        continue
                    self._report_step.record_timing(f"{stage}_total_s", total_s)

            report_start = time.perf_counter()
            self._report_step.finalize(
                self._trace,
                metrics=metrics_payload,
                tasks=summaries,
                pre_write_hook=pre_write_hook,
            )
        finally:
            try:
                self.shutdown()
            finally:
                self._trace.flush()

    def attach_report_step(self, step: ReportStep) -> None:
        self._report_step = step

    def shutdown(self) -> None:
        if self._shutdown_called:
            return
        self._shutdown_called = True
        try:
            for entry in self._tasks:
                entry.sample_loop.shutdown()
        finally:
            self._role_manager.shutdown()

    def set_wall_clock_start(self, start_s: float) -> None:
        self._wall_clock_start = start_s

    def _resolve_sample_count(self) -> int:
        if self._report_step and hasattr(self._report_step, "_cache"):
            try:
                return int(self._report_step._cache.sample_count)  # type: ignore[attr-defined]
            except Exception:
                pass
        total = 0
        for entry in self._tasks:
            try:
                total += int(entry.sample_loop.processed_count)
            except Exception:
                continue
        return total

    def _record_throughput_metrics(
        self,
        *,
        sample_count: int,
        wall_runtime_s: float,
        inference_s: float,
        evaluation_s: float,
    ) -> None:
        if not self._report_step or sample_count <= 0:
            return
        if wall_runtime_s > 0:
            self._report_step.record_timing("throughput_total_samples_per_s", sample_count / wall_runtime_s)
            self._report_step.record_timing("latency_total_ms_per_sample", wall_runtime_s * 1000.0 / sample_count)
        if inference_s > 0:
            self._report_step.record_timing("throughput_inference_samples_per_s", sample_count / inference_s)
            self._report_step.record_timing("latency_inference_ms_per_sample", inference_s * 1000.0 / sample_count)
        if evaluation_s > 0:
            self._report_step.record_timing("throughput_auto_eval_samples_per_s", sample_count / evaluation_s)


def _prepare_task_entries(
    *,
    task_plans: Sequence[TaskPlanSpec],
    config: PipelineConfig,
    data_manager: DataManager,
    datasets: Dict[str, DataSource],
    trace: ObservabilityTrace,
    cache_store: EvalCache,
    resource_profile: ResourceProfile,
    sandbox_profiles: Dict[str, Dict[str, Any]],
) -> Sequence[_TaskRuntimeEntry]:
    entries: List[_TaskRuntimeEntry] = []
    for plan in task_plans:
        dataset_id = plan.dataset_id
        if dataset_id not in datasets:
            raise KeyError(f"Dataset '{dataset_id}' referenced by task '{plan.task_id}' is not registered")
        source = data_manager.get(dataset_id)
        raw_samples = data_manager.iter_samples(dataset_id, trace=trace)
        samples = _validate_samples(
            raw_samples,
            validator=source.validator,
            dataset_id=dataset_id,
            trace=trace,
        )
        concurrency = _resolve_concurrency(plan.runtime_policy.concurrency, resource_profile)
        concurrency = _apply_fixed_port_guard(
            scope=f"task '{plan.task_id}'",
            concurrency=concurrency,
            role_bindings=plan.role_bindings,
            sandbox_profiles=sandbox_profiles,
        )
        sample_loop = SampleLoop(
            samples,
            shuffle=plan.runtime_policy.shuffle,
            shuffle_seed=plan.runtime_policy.shuffle_seed,
            max_samples=plan.runtime_policy.max_samples,
            concurrency=concurrency,
            prefetch_factor=plan.runtime_policy.prefetch_factor,
            max_inflight=plan.runtime_policy.max_inflight,
            streaming=source.streaming,
            task_id=plan.task_id,
            sandbox_profiles=sandbox_profiles,
        )
        metric_specs = plan.metrics or config.metrics
        metric_registry = MetricRegistry()
        task_planner = TaskPlanner()
        task_planner.configure_metrics(metric_specs, metric_registry, cache_store=cache_store)
        sample_loop.configure_custom_steps(plan.steps)
        task_planner.attach_task_plan_spec(plan)
        entries.append(
            _TaskRuntimeEntry(
                task_id=plan.task_id,
                dataset_id=dataset_id,
                dataset_metadata=source.metadata or {},
                sample_loop=sample_loop,
                task_planner=task_planner,
                reporting=plan.reporting or {},
            )
        )
    return entries


def _resolve_concurrency(explicit: Optional[int], resource_profile: ResourceProfile) -> int:
    """Determine SampleLoop concurrency from overrides/env/resource profile."""

    if explicit is not None:
        return max(1, int(explicit))
    env_threads = os.environ.get("GAGE_EVAL_THREADS")
    if env_threads and env_threads.isdigit():
        return max(1, int(env_threads))
    total_gpus = sum(max(node.gpus, 0) for node in resource_profile.nodes)
    if total_gpus > 0:
        return max(1, total_gpus)
    total_cpus = sum(max(node.cpus, 0) for node in resource_profile.nodes)
    cpu_default = os.cpu_count() or 1
    total_cpus = total_cpus or cpu_default
    return max(1, min(total_cpus, 4))


_SANDBOX_ENDPOINT_KEYS: Set[str] = {
    "env_endpoint",
    "environment_endpoint",
    "env_url",
    "environment_url",
    "apis_endpoint",
    "apis_url",
    "mcp_endpoint",
    "mcp_url",
}


def _apply_fixed_port_guard(
    *,
    scope: str,
    concurrency: int,
    role_bindings: Dict[str, RoleAdapterSpec],
    sandbox_profiles: Dict[str, Dict[str, Any]],
) -> int:
    if concurrency <= 1:
        return concurrency
    reasons = _find_fixed_port_sandboxes(role_bindings, sandbox_profiles)
    if not reasons:
        return concurrency
    logger.warning(
        "Fixed sandbox ports detected for {} ({}); forcing concurrency from {} to 1.",
        scope,
        ", ".join(sorted(reasons)),
        concurrency,
    )
    return 1


def _find_fixed_port_sandboxes(
    role_bindings: Dict[str, RoleAdapterSpec],
    sandbox_profiles: Dict[str, Dict[str, Any]],
) -> Set[str]:
    reasons: Set[str] = set()
    for adapter_id, spec in role_bindings.items():
        sandbox_config = dict(spec.sandbox or {})
        if not sandbox_config:
            continue
        effective = _merge_sandbox_config(sandbox_config, sandbox_profiles)
        fixed_ports = _extract_fixed_ports(effective)
        if not fixed_ports:
            continue
        sandbox_id = effective.get("sandbox_id") or sandbox_config.get("sandbox_id") or adapter_id
        reasons.add(f"{sandbox_id}:{sorted(fixed_ports)}")
    return reasons


def _merge_sandbox_config(
    sandbox_config: Dict[str, Any],
    sandbox_profiles: Dict[str, Dict[str, Any]],
) -> Dict[str, Any]:
    base = dict(sandbox_config or {})
    sandbox_id = base.get("sandbox_id") or base.get("template_name")
    if sandbox_id and sandbox_id in sandbox_profiles:
        return _deep_merge(dict(sandbox_profiles[sandbox_id]), base)
    return base


def _extract_fixed_ports(sandbox_config: Dict[str, Any]) -> Set[int]:
    fixed_ports: Set[int] = set()
    runtime_configs = dict(sandbox_config.get("runtime_configs") or {})
    for mapping in _normalize_ports(runtime_configs.get("ports")):
        port = _extract_host_port(str(mapping))
        if port and port > 0:
            fixed_ports.add(port)
    for key in _SANDBOX_ENDPOINT_KEYS:
        endpoint = runtime_configs.get(key) or sandbox_config.get(key)
        if not endpoint:
            continue
        port = _parse_endpoint_port(str(endpoint))
        if port and port > 0:
            fixed_ports.add(port)
    return fixed_ports


def _normalize_ports(raw: Any) -> List[str]:
    if raw is None:
        return []
    if isinstance(raw, dict):
        return [
            f"{host}:{container}" if container is not None else str(host)
            for host, container in raw.items()
        ]
    if isinstance(raw, (list, tuple)):
        entries: List[str] = []
        for item in raw:
            if isinstance(item, (list, tuple)) and len(item) == 2:
                entries.append(f"{item[0]}:{item[1]}")
            else:
                entries.append(str(item))
        return entries
    return [str(raw)]


def _extract_host_port(mapping: str) -> Optional[int]:
    raw = mapping
    if not raw:
        return None
    if "/" in raw:
        raw = raw.split("/", 1)[0]
    parts = raw.split(":")
    if len(parts) == 1:
        candidate = parts[0]
    elif len(parts) >= 3:
        candidate = parts[-2]
    else:
        candidate = parts[0]
    try:
        return int(candidate)
    except ValueError:
        return None


def _parse_endpoint_port(endpoint: str) -> Optional[int]:
    raw = endpoint.strip()
    if "://" not in raw:
        raw = f"http://{raw}"
    parsed = urlparse(raw)
    return parsed.port


def _deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    merged = dict(base or {})
    for key, value in (override or {}).items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def _max_concurrency_from_tasks(
    task_plans: Sequence["TaskPlanSpec"],
    resource_profile: ResourceProfile,
) -> Optional[int]:
    hint = 0
    for plan in task_plans:
        hint = max(hint, _resolve_concurrency(plan.runtime_policy.concurrency, resource_profile))
    return hint or None
