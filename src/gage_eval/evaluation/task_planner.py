"""Task planning utilities."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Sequence

from loguru import logger
from gage_eval.config.pipeline_config import MetricSpec
from gage_eval.observability.trace import ObservabilityTrace
from gage_eval.metrics import MetricRegistry
from gage_eval.evaluation.sample_envelope import append_predict_result, update_eval_result

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
    steps: Sequence[Any] = field(default_factory=tuple)

    def create_context(self, sample: dict, trace: ObservabilityTrace, role_manager):
        from gage_eval.pipeline.steps.support import SupportStep
        from gage_eval.pipeline.steps.inference import InferenceStep
        from gage_eval.pipeline.steps.arena import ArenaStep
        from gage_eval.pipeline.steps.judge import JudgeStep

        support = SupportStep(self.support_steps)
        inference = InferenceStep(self.inference_role)
        arena = ArenaStep(self.arena_role)
        judge = JudgeStep(self.judge_role)
        return StepExecutionContext(
            sample,
            support,
            inference,
            arena,
            judge,
            self.auto_eval_step,
            trace,
            role_manager,
            metadata=self.metadata,
            auto_eval_enabled=self.auto_eval_enabled,
        )


class TaskPlanner:
    """Generate TaskPlan objects based on sample metadata and config."""

    def __init__(self, metric_specs: Optional[Sequence[MetricSpec]] = None, metric_registry: Optional[MetricRegistry] = None) -> None:
        self._custom_steps: Sequence[dict] = ()
        self._metric_specs: Sequence[MetricSpec] = metric_specs or ()
        self._metric_registry = metric_registry or MetricRegistry()
        self._auto_eval_step: Optional[AutoEvalStep] = None
        self._cached_support_steps: Sequence[dict] = ()
        self._cached_inference_role: Optional[str] = None
        self._cached_arena_role: Optional[str] = None
        self._cached_judge_role: Optional[str] = None
        self._auto_eval_requested: bool = False
        self._plan_spec: Optional["TaskPlanSpec"] = None
        self._cached_step_sequence: Sequence[Any] = ()

    def configure_custom_steps(self, steps: Sequence[dict]) -> None:
        self._custom_steps = steps
        self._cached_step_sequence = tuple(steps)
        (
            self._cached_support_steps,
            self._cached_inference_role,
            self._cached_arena_role,
            self._cached_judge_role,
            self._auto_eval_requested,
        ) = self._derive_layout(steps)
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
        )
        logger.info(
            "TaskPlanner registered {} metric specs (cache_enabled={})",
            len(metric_specs),
            cache_store is not None,
        )

    def prepare_plan(
        self,
        sample: dict,
        custom_steps: Optional[Sequence[dict]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> TaskPlan:
        if custom_steps is not None:
            ordered_steps: Sequence[Any] = tuple(custom_steps)
            support_steps, inference, arena, judge, auto_eval_requested = self._derive_layout(ordered_steps)
        else:
            support_steps = self._cached_support_steps
            inference = self._cached_inference_role
            arena = self._cached_arena_role
            judge = self._cached_judge_role
            auto_eval_requested = self._auto_eval_requested
            ordered_steps = self._cached_step_sequence
        final_metadata = {"sample_id": str(sample.get("id", "unknown"))}
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
            steps=ordered_steps,
        )

    def get_auto_eval_step(self) -> Optional[AutoEvalStep]:
        return self._auto_eval_step

    @staticmethod
    def _derive_layout(
        steps: Sequence[dict],
    ) -> tuple[Sequence[dict], Optional[str], Optional[str], Optional[str], bool]:
        # Support steps are executed sequentially before handing off to inference/judge roles.
        support_steps = tuple(step for step in steps if step.get("step") == "support")
        inference = next((step.get("adapter_id") or step.get("role_ref") for step in steps if step.get("step") == "inference"), None)
        arena = next((step.get("adapter_id") or step.get("role_ref") for step in steps if step.get("step") == "arena"), None)
        judge = next((step.get("adapter_id") or step.get("role_ref") for step in steps if step.get("step") == "judge"), None)
        auto_eval = any(step.get("step") == "auto_eval" for step in steps)
        return support_steps, inference, arena, judge, auto_eval


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
    ) -> None:
        self.sample = sample
        self.support = support
        self.inference = inference
        self.arena = arena
        self.judge = judge
        self.auto_eval_step = auto_eval_step
        self.auto_eval_enabled = auto_eval_enabled
        self.trace = trace
        self.role_manager = role_manager
        self.metadata = metadata or {}
        self._model_output: Optional[dict] = None
        self._judge_output: Optional[dict] = None
        self.sample.setdefault("predict_result", self.sample.get("predict_result") or [])
        self.sample.setdefault("eval_result", self.sample.get("eval_result") or {})

    def execute_support(self) -> None:
        support_steps = getattr(self.support, "_steps", ())
        logger.trace("Executing support step with {} entries", len(support_steps))
        self.support.execute(self.sample, self.role_manager, self.trace)

    def execute_support_step(self, step) -> None:
        logger.trace("Executing support entry adapter={}", step.get("adapter_id") if hasattr(step, "get") else None)
        if hasattr(self.support, "execute_single"):
            self.support.execute_single(step, self.sample, self.role_manager, self.trace)
        else:
            self.support.execute(self.sample, self.role_manager, self.trace)

    def execute_inference(self) -> None:
        logger.trace("Executing inference step adapter={}", getattr(self.inference, "_adapter_id", None))
        self._model_output = self.inference.execute(self.sample, self.role_manager, self.trace)
        append_predict_result(self.sample, self._model_output)

    def execute_arena(self) -> None:
        logger.trace("Executing arena step adapter={}", getattr(self.arena, "_adapter_id", None))
        self._model_output = self.arena.execute(self.sample, self.role_manager, self.trace)
        append_predict_result(self.sample, self._model_output)

    def execute_judge(self) -> None:
        logger.trace("Executing judge step adapter={}", getattr(self.judge, "_adapter_id", None))
        payload = {"sample": self.sample, "model_output": self._model_output or {}, "trace": self.trace}
        self._judge_output = self.judge.execute(payload, self.role_manager, self.trace)
        update_eval_result(self.sample, self._judge_output)

    def execute_auto_eval(self, sample_id: str) -> None:
        if (
            not self.auto_eval_enabled
            or not self.auto_eval_step
            or not self.auto_eval_step.has_metrics()
        ):
            logger.debug("Auto-eval step skipped for sample_id={} (no metrics)", sample_id)
            return
        self.auto_eval_step.execute(
            sample_id=sample_id,
            sample=self.sample,
            model_output=self._model_output or {},
            judge_output=self._judge_output or {},
            trace=self.trace,
            task_id=self.metadata.get("task_id"),
        )
