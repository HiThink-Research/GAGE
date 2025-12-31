"""Pipeline composition utilities."""

from __future__ import annotations

from typing import Callable, Dict, Iterable, List, Optional
import os
import time

from loguru import logger

from gage_eval.config.pipeline_config import (
    BuiltinPipelineSpec,
    CustomPipelineSpec,
    PipelineConfig,
)
from gage_eval.config.registry import ConfigRegistry
from gage_eval.evaluation.sample_loop import SampleLoop
from gage_eval.evaluation.task_planner import TaskPlanner
from gage_eval.metrics import MetricRegistry
from gage_eval.evaluation.cache import EvalCache
from gage_eval.pipeline.steps.report import ReportStep
from gage_eval.role.role_manager import RoleManager
from gage_eval.observability.trace import ObservabilityTrace


class PipelineRuntime:
    """Runtime container bundling SampleLoop, TaskPlanner and RoleManager."""

    def __init__(
        self,
        sample_loop: SampleLoop,
        task_planner: TaskPlanner,
        role_manager: RoleManager,
        trace: ObservabilityTrace,
        report_step: Optional[ReportStep] = None,
    ) -> None:
        self.sample_loop = sample_loop
        self.task_planner = task_planner
        self.role_manager = role_manager
        self.trace = trace
        self.report_step = report_step
        self.wall_clock_start: Optional[float] = None

    def run(self) -> None:
        """Kick off execution for every sample."""

        try:
            logger.info("PipelineRuntime started (report_step={})", bool(self.report_step))
            total_start = time.perf_counter()
            wall_start = self.wall_clock_start or total_start
            inference_start = time.perf_counter()
            self.sample_loop.run(
                planner=self.task_planner,
                role_manager=self.role_manager,
                trace=self.trace,
            )
            inference_elapsed = time.perf_counter() - inference_start
            if self.report_step:
                eval_elapsed = 0.0

                def pre_write_hook() -> None:
                    nonlocal eval_elapsed
                    eval_elapsed = time.perf_counter() - report_start
                    self.report_step.record_timing("inference_s", inference_elapsed)
                    self.report_step.record_timing("evaluation_s", eval_elapsed)
                    execution_total = time.perf_counter() - total_start
                    self.report_step.record_timing("execution_runtime_s", execution_total)
                    dataset_time = self.report_step.get_timing("dataset_materialization_s") or 0.0
                    self.report_step.record_timing("total_runtime_s", dataset_time + execution_total)
                    wall_total = time.perf_counter() - wall_start
                    self.report_step.record_timing("wall_runtime_s", wall_total)
                    self._record_throughput_metrics(
                        sample_count=self._resolve_sample_count(),
                        wall_runtime_s=wall_total,
                        inference_s=inference_elapsed,
                        evaluation_s=eval_elapsed,
                    )

                report_start = time.perf_counter()
                self.report_step.finalize(self.trace, pre_write_hook=pre_write_hook)
            else:
                eval_elapsed = 0.0
            total_elapsed = time.perf_counter() - total_start
            logger.info(
                "PipelineRuntime finished successfully (inference={:.2f}s eval={:.2f}s total={:.2f}s)",
                inference_elapsed,
                eval_elapsed,
                total_elapsed,
            )
        finally:
            self.trace.flush()

    def attach_report_step(self, step: ReportStep) -> None:
        self.report_step = step

    def set_wall_clock_start(self, start_s: float) -> None:
        self.wall_clock_start = start_s

    def _resolve_sample_count(self) -> int:
        if self.report_step and hasattr(self.report_step, "_cache"):
            try:
                return int(self.report_step._cache.sample_count)  # type: ignore[attr-defined]
            except Exception:
                pass
        try:
            return int(self.sample_loop.processed_count)
        except Exception:
            return 0

    def _record_throughput_metrics(
        self,
        *,
        sample_count: int,
        wall_runtime_s: float,
        inference_s: float,
        evaluation_s: float,
    ) -> None:
        if not self.report_step or sample_count <= 0:
            return
        # Overall throughput and latency (wall clock).
        if wall_runtime_s > 0:
            self.report_step.record_timing("throughput_total_samples_per_s", sample_count / wall_runtime_s)
            self.report_step.record_timing("latency_total_ms_per_sample", wall_runtime_s * 1000.0 / sample_count)
        # Inference-stage throughput and latency.
        if inference_s > 0:
            self.report_step.record_timing("throughput_inference_samples_per_s", sample_count / inference_s)
            self.report_step.record_timing("latency_inference_ms_per_sample", inference_s * 1000.0 / sample_count)
        # Auto-eval stage throughput.
        if evaluation_s > 0:
            self.report_step.record_timing("throughput_auto_eval_samples_per_s", sample_count / evaluation_s)


class BuiltinPipeline:
    """Simple template object that knows how to build a runtime."""

    def __init__(
        self,
        pipeline_id: str,
        builder: Callable[[PipelineConfig, RoleManager], PipelineRuntime],
    ) -> None:
        self.pipeline_id = pipeline_id
        self._builder = builder

    def materialize(self, config: PipelineConfig, role_manager: RoleManager) -> PipelineRuntime:
        return self._builder(config, role_manager)


class PipelineFactory:
    """Factory able to create built-in or custom runtimes on demand."""

    def __init__(self, registry: ConfigRegistry) -> None:
        self._builtin_registry: Dict[str, BuiltinPipeline] = {}
        self._registry = registry

    def register_builtin(self, pipeline: BuiltinPipeline) -> None:
        self._builtin_registry[pipeline.pipeline_id] = pipeline

    def create_runtime(
        self,
        config: PipelineConfig,
        role_manager: RoleManager,
        sample_loop: SampleLoop,
        task_planner: TaskPlanner,
        trace: ObservabilityTrace,
        *,
        cache_store: Optional[EvalCache] = None,
    ) -> PipelineRuntime:
        """Resolve either a built-in template or custom inline steps."""

        logger.info("Creating runtime for pipeline_id='{}'", config.pipeline_id)
        metric_registry = MetricRegistry()
        cache_store = cache_store or EvalCache(
            base_dir=os.environ.get("GAGE_EVAL_SAVE_DIR"),
            run_id=trace.run_id,
        )
        task_planner.configure_metrics(config.metrics, metric_registry, cache_store=cache_store)
        report_step = ReportStep(task_planner.get_auto_eval_step(), cache_store)
        backend_instances = self._registry.materialize_backends(config)
        prompt_assets = self._registry.materialize_prompts(config)
        backend_specs = [
            {
                "backend_id": spec.backend_id,
                "type": spec.type,
                "config": spec.config,
            }
            for spec in config.backends
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
            for spec in config.models
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
            for spec in config.role_adapters
        ]
        if role_specs:
            cache_store.set_metadata("role_adapters", role_specs)
        if config.summary_generators:
            cache_store.set_metadata("summary_generators", list(config.summary_generators))
        adapters = self._registry.materialize_role_adapters(
            config,
            backends=backend_instances,
            prompts=prompt_assets,
        )
        for adapter_id, adapter in adapters.items():
            role_manager.register_role_adapter(adapter_id, adapter)
            logger.debug("Registered role adapter '{}' during runtime creation", adapter_id)

        if config.builtin:
            builtin = self._builtin_registry.get(config.builtin.pipeline_id)
            if not builtin:
                raise KeyError(f"Builtin pipeline '{config.builtin.pipeline_id}' not registered")
            logger.info("Using builtin pipeline '{}'", config.builtin.pipeline_id)
            runtime = builtin.materialize(config, role_manager)
            runtime.attach_report_step(report_step)
            return runtime

        if config.custom:
            logger.info("Materializing custom pipeline with {} inline steps", len(config.custom.steps))
            return self._create_custom_runtime(
                config.custom,
                role_manager,
                sample_loop,
                task_planner,
                trace,
                report_step,
            )

        raise ValueError("PipelineConfig must declare either builtin or custom pipeline")

    def _create_custom_runtime(
        self,
        spec: CustomPipelineSpec,
        role_manager: RoleManager,
        sample_loop: SampleLoop,
        task_planner: TaskPlanner,
        trace: ObservabilityTrace,
        report_step: ReportStep,
    ) -> PipelineRuntime:
        """Assemble a runtime whose Support/Inference/Judge steps are declared inline."""

        sample_loop.configure_custom_steps(spec.steps)
        task_planner.configure_custom_steps(spec.steps)
        # Custom pipelines rely on the same runtime container; only the per-sample steps differ.
        return PipelineRuntime(
            sample_loop=sample_loop,
            task_planner=task_planner,
            role_manager=role_manager,
            trace=trace,
            report_step=report_step,
        )
