"""Task planning utilities."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Sequence

from loguru import logger
from gage_eval.config.pipeline_config import CustomPipelineStep, MetricSpec
from gage_eval.evaluation.support_artifacts import build_support_slot_id
from gage_eval.evaluation.execution_controller import TaskExecutionController
from gage_eval.observability.trace import ObservabilityTrace
from gage_eval.metrics import MetricRegistry
from gage_eval.evaluation.sample_ingress import resolve_runtime_sample_id
from gage_eval.evaluation.sample_envelope import (
    append_arena_contract,
    append_predict_result,
    ensure_arena_header,
    update_eval_result,
)
from gage_eval.evaluation.step_factory import StepFactory, TaskStepBundle
from gage_eval.pipeline.step_contracts import (
    collect_step_sequence_issues,
    get_step_adapter_id,
    get_step_type,
)
from gage_eval.sandbox.provider import SandboxProvider

if TYPE_CHECKING:  # pragma: no cover
    from gage_eval.evaluation.task_plan import TaskPlanSpec
    from gage_eval.pipeline.steps.auto_eval import AutoEvalStep


@dataclass
class TaskPlan:
    """Represents the per-sample plan to be executed by the SampleLoop."""

    support_steps: Sequence[Dict[str, Any]] = field(default_factory=tuple)
    inference_role: Optional[str] = None
    arena_role: Optional[str] = None
    judge_role: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    metric_specs: Sequence[MetricSpec] = field(default_factory=tuple)
    metric_registry: Optional[MetricRegistry] = None
    auto_eval_step: Optional[AutoEvalStep] = None
    auto_eval_enabled: bool = False
    support_payload_policy: Dict[str, Any] = field(default_factory=dict)
    steps: Sequence[Any] = field(default_factory=tuple)
    step_bundle: Optional[TaskStepBundle] = None

    def create_context(
        self,
        sample: dict,
        trace: ObservabilityTrace,
        role_manager,
        *,
        sandbox_provider: Optional[SandboxProvider] = None,
    ):
        bundle = self.step_bundle or StepFactory().build_bundle(
            support_steps=self.support_steps,
            inference_role=self.inference_role,
            arena_role=self.arena_role,
            judge_role=self.judge_role,
            auto_eval_step=self.auto_eval_step,
        )
        return StepExecutionContext(
            sample,
            bundle.support,
            bundle.inference,
            bundle.arena,
            bundle.judge,
            bundle.auto_eval_step,
            trace,
            role_manager,
            metadata=self.metadata,
            auto_eval_enabled=self.auto_eval_enabled,
            support_payload_policy=self.support_payload_policy,
            sandbox_provider=sandbox_provider,
        )


class TaskPlanner:
    """Generate TaskPlan objects based on sample metadata and config."""

    def __init__(self, metric_specs: Optional[Sequence[MetricSpec]] = None, metric_registry: Optional[MetricRegistry] = None) -> None:
        self._custom_steps: Sequence[dict] = ()
        self._metric_specs: Sequence[MetricSpec] = metric_specs or ()
        self._metric_registry = metric_registry or MetricRegistry()
        self._auto_eval_step: Optional[AutoEvalStep] = None
        self._execution_controller: Optional[TaskExecutionController] = None
        self._cached_support_steps: Sequence[dict] = ()
        self._cached_inference_role: Optional[str] = None
        self._cached_arena_role: Optional[str] = None
        self._cached_judge_role: Optional[str] = None
        self._auto_eval_requested: bool = False
        self._plan_spec: Optional["TaskPlanSpec"] = None
        self._cached_step_sequence: Sequence[Any] = ()
        self._cached_step_bundle: Optional[TaskStepBundle] = None
        self._step_factory = StepFactory()

    def configure_custom_steps(self, steps: Sequence[dict]) -> None:
        normalized_steps = _annotate_support_steps(steps)
        self._custom_steps = normalized_steps
        self._cached_step_sequence = tuple(normalized_steps)
        (
            self._cached_support_steps,
            self._cached_inference_role,
            self._cached_arena_role,
            self._cached_judge_role,
            self._auto_eval_requested,
        ) = self._derive_layout(normalized_steps)
        self._cached_step_bundle = self._step_factory.build_bundle(
            support_steps=self._cached_support_steps,
            inference_role=self._cached_inference_role,
            arena_role=self._cached_arena_role,
            judge_role=self._cached_judge_role,
            auto_eval_step=self._auto_eval_step,
        )
        logger.debug(
            "TaskPlanner configured {} custom steps (inference={}, arena={}, judge={}, auto_eval={})",
            len(steps),
            self._cached_inference_role,
            self._cached_arena_role,
            self._cached_judge_role,
            self._auto_eval_requested,
        )

    def attach_task_plan_spec(self, plan_spec: "TaskPlanSpec") -> None:
        """Binds a precomputed TaskPlanSpec to avoid re-parsing the step layout."""

        self._plan_spec = plan_spec
        self.configure_custom_steps(plan_spec.steps)

    def configure_metrics(
        self,
        metric_specs: Sequence[MetricSpec],
        metric_registry: Optional[MetricRegistry] = None,
        *,
        cache_store=None,
    ) -> None:
        from gage_eval.pipeline.steps.auto_eval import AutoEvalStep

        self._metric_specs = metric_specs
        if metric_registry is not None:
            self._metric_registry = metric_registry
        self._auto_eval_step = AutoEvalStep(
            metric_specs=self._metric_specs,
            metric_registry=self._metric_registry,
            cache_store=cache_store,
            execution_controller=self._execution_controller,
        )
        if self._cached_step_sequence:
            self._cached_step_bundle = self._step_factory.build_bundle(
                support_steps=self._cached_support_steps,
                inference_role=self._cached_inference_role,
                arena_role=self._cached_arena_role,
                judge_role=self._cached_judge_role,
                auto_eval_step=self._auto_eval_step,
            )
        logger.info(
            "TaskPlanner registered {} metric specs (cache_enabled={})",
            len(metric_specs),
            cache_store is not None,
        )

    def attach_execution_controller(
        self, controller: Optional[TaskExecutionController]
    ) -> None:
        self._execution_controller = controller
        if self._auto_eval_step is not None:
            self._auto_eval_step.attach_execution_controller(controller)

    def prepare_plan(
        self,
        sample: dict,
        custom_steps: Optional[Sequence[dict]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> TaskPlan:
        if custom_steps is not None:
            ordered_steps = _annotate_support_steps(custom_steps)
            support_steps, inference, arena, judge, auto_eval_requested = self._derive_layout(ordered_steps)
            step_bundle = self._step_factory.build_bundle(
                support_steps=support_steps,
                inference_role=inference,
                arena_role=arena,
                judge_role=judge,
                auto_eval_step=self._auto_eval_step,
            )
        else:
            support_steps = self._cached_support_steps
            inference = self._cached_inference_role
            arena = self._cached_arena_role
            judge = self._cached_judge_role
            auto_eval_requested = self._auto_eval_requested
            ordered_steps = self._cached_step_sequence
            step_bundle = self._cached_step_bundle
        resolved_task_id = None
        if metadata and metadata.get("task_id") is not None:
            resolved_task_id = str(metadata["task_id"])
        final_metadata = {
            "sample_id": resolve_runtime_sample_id(sample, task_id=resolved_task_id)
        }
        if metadata:
            final_metadata.update(metadata)
        logger.debug(
            "Prepared TaskPlan metadata={} inference={} arena={} judge={}",
            final_metadata,
            inference,
            arena,
            judge,
        )
        return TaskPlan(
            support_steps=support_steps,
            inference_role=inference,
            arena_role=arena,
            judge_role=judge,
            metadata=final_metadata,
            metric_specs=self._metric_specs,
            metric_registry=self._metric_registry,
            auto_eval_step=self._auto_eval_step,
            auto_eval_enabled=auto_eval_requested,
            support_payload_policy=self._resolve_support_payload_policy(),
            steps=ordered_steps,
            step_bundle=step_bundle,
        )

    def get_auto_eval_step(self) -> Optional[AutoEvalStep]:
        return self._auto_eval_step

    @staticmethod
    def _derive_layout(
        steps: Sequence[Any],
    ) -> tuple[Sequence[dict], Optional[str], Optional[str], Optional[str], bool]:
        issues = collect_step_sequence_issues(
            steps,
            owner_label="Configured step",
        )
        if issues:
            raise ValueError(issues[0].message)
        # Support steps are executed sequentially before handing off to inference/judge roles.
        support_steps = tuple(step for step in steps if get_step_type(step) == "support")
        inference = _resolve_single_role_binding(steps, "inference")
        arena = _resolve_single_role_binding(steps, "arena")
        judge = _resolve_single_role_binding(steps, "judge")
        auto_eval = any(get_step_type(step) == "auto_eval" for step in steps)
        return support_steps, inference, arena, judge, auto_eval

    def _resolve_support_payload_policy(self) -> Dict[str, Any]:
        if self._plan_spec is None:
            return {}
        return dict(self._plan_spec.runtime_policy.support_payload_policy or {})


def _resolve_single_role_binding(steps: Sequence[Any], step_type: str) -> Optional[str]:
    bindings = [
        get_step_adapter_id(step)
        for step in steps
        if get_step_type(step) == step_type
    ]
    if len(bindings) > 1:
        raise ValueError(
            f"Configured step '{step_type}' appears {len(bindings)} times, but only one role binding is supported"
        )
    return bindings[0] if bindings else None


class StepExecutionContext:
    """Utility wrapper bundling all step instances for a sample."""

    def __init__(
        self,
        sample: dict,
        support,
        inference,
        arena,
        judge,
        auto_eval_step,
        trace: ObservabilityTrace,
        role_manager,
        metadata: Optional[Dict[str, Any]] = None,
        auto_eval_enabled: bool = False,
        support_payload_policy: Optional[Dict[str, Any]] = None,
        sandbox_provider: Optional[SandboxProvider] = None,
    ) -> None:
        self.sample = sample
        self.support = support
        self.inference = inference
        self.arena = arena
        self.judge = judge
        self.auto_eval_step = auto_eval_step
        self.auto_eval_enabled = auto_eval_enabled
        self.support_payload_policy = dict(support_payload_policy or {})
        self.trace = trace
        self.role_manager = role_manager
        self.metadata = metadata or {}
        self.sandbox_provider = sandbox_provider
        self._model_output: Optional[dict] = None
        self._judge_output: Optional[dict] = None
        self.sample.setdefault("predict_result", self.sample.get("predict_result") or [])
        self.sample.setdefault("eval_result", self.sample.get("eval_result") or {})

    def execute_support(self) -> None:
        support_steps = getattr(self.support, "_steps", ())
        logger.trace("Executing support step with {} entries", len(support_steps))
        self.support.execute(
            self.sample,
            self.role_manager,
            self.trace,
            support_payload_policy=self.support_payload_policy,
            sandbox_provider=self.sandbox_provider,
        )

    def execute_support_step(self, step) -> None:
        logger.trace("Executing support entry adapter={}", step.get("adapter_id") if hasattr(step, "get") else None)
        if hasattr(self.support, "execute_single"):
            self.support.execute_single(
                step,
                self.sample,
                self.role_manager,
                self.trace,
                support_payload_policy=self.support_payload_policy,
                sandbox_provider=self.sandbox_provider,
            )
        else:
            self.support.execute(
                self.sample,
                self.role_manager,
                self.trace,
                support_payload_policy=self.support_payload_policy,
                sandbox_provider=self.sandbox_provider,
            )

    def execute_inference(self) -> None:
        logger.trace("Executing inference step adapter={}", getattr(self.inference, "_adapter_id", None))
        self._model_output = self.inference.execute(
            self.sample,
            self.role_manager,
            self.trace,
            sandbox_provider=self.sandbox_provider,
        )
        append_predict_result(self.sample, self._model_output)

    def execute_arena(self) -> None:
        logger.trace("Executing arena step adapter={}", getattr(self.arena, "_adapter_id", None))
        start_time_ms = int(time.time() * 1000)
        ensure_arena_header(self.sample, start_time_ms=start_time_ms)
        self._model_output = self.arena.execute(
            self.sample,
            self.role_manager,
            self.trace,
            sandbox_provider=self.sandbox_provider,
        )
        append_arena_contract(
            self.sample,
            self._model_output,
            end_time_ms=int(time.time() * 1000),
        )

    def execute_judge(self) -> None:
        logger.trace("Executing judge step adapter={}", getattr(self.judge, "_adapter_id", None))
        payload = {
            "sample": self.sample,
            "model_output": self._model_output or {},
            "trace": self.trace,
            "sandbox_provider": self.sandbox_provider,
        }
        self._judge_output = self.judge.execute(payload, self.role_manager, self.trace)
        update_eval_result(self.sample, self._judge_output)

    def execute_auto_eval(self, sample_id: str) -> None:
        if not self.auto_eval_enabled or not self.auto_eval_step:
            logger.debug("Auto-eval step skipped for sample_id={} (disabled)", sample_id)
            return
        if not self.auto_eval_step.has_metrics():
            logger.debug("Auto-eval requested without metrics for sample_id={}", sample_id)
        self.auto_eval_step.execute(
            sample_id=sample_id,
            sample=self.sample,
            model_output=self._model_output or {},
            judge_output=self._judge_output or {},
            trace=self.trace,
            task_id=self.metadata.get("task_id"),
        )


def _annotate_support_steps(steps: Sequence[Any]) -> Sequence[Any]:
    annotated: List[Any] = []
    support_ordinal = 0
    for step in steps:
        if get_step_type(step) != "support":
            annotated.append(step)
            continue
        slot_id = build_support_slot_id(step, support_ordinal)
        support_ordinal += 1
        annotated.append(_clone_step_with_param(step, "support_slot_id", slot_id))
    return tuple(annotated)


def _clone_step_with_param(step: Any, key: str, value: Any) -> Any:
    if isinstance(step, CustomPipelineStep):
        params = dict(step.params or {})
        params.setdefault(key, value)
        return CustomPipelineStep(
            step_type=step.step_type,
            adapter_id=step.adapter_id,
            params=params,
        )
    if isinstance(step, dict):
        cloned = dict(step)
        params = dict(cloned.get("params") or {})
        params.setdefault(key, value)
        cloned["params"] = params
        return cloned
    if hasattr(step, "params"):
        params = dict(getattr(step, "params") or {})
        params.setdefault(key, value)
        try:
            return step.__class__(
                step_type=getattr(step, "step_type"),
                adapter_id=getattr(step, "adapter_id", None),
                params=params,
            )
        except Exception:
            pass
    return step
