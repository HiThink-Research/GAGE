"""Task-scoped step bundle construction."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from typing import Optional, Sequence, TYPE_CHECKING

from gage_eval.pipeline.steps.arena import ArenaStep
from gage_eval.pipeline.steps.inference import InferenceStep
from gage_eval.pipeline.steps.judge import JudgeStep
from gage_eval.pipeline.steps.support import SupportStep

if TYPE_CHECKING:  # pragma: no cover
    from gage_eval.pipeline.steps.auto_eval import AutoEvalStep


@dataclass(frozen=True)
class TaskStepBundle:
    """Task-scoped reusable step wrappers."""

    support: Optional[SupportStep] = None
    inference: Optional[InferenceStep] = None
    arena: Optional[ArenaStep] = None
    judge: Optional[JudgeStep] = None
    auto_eval_step: Optional["AutoEvalStep"] = None


class StepFactory:
    """Build task-scoped step bundles and freeze reusable wrappers."""

    def build_bundle(
        self,
        *,
        support_steps: Sequence[dict],
        inference_role: Optional[str],
        arena_role: Optional[str],
        judge_role: Optional[str],
        auto_eval_step: Optional["AutoEvalStep"],
    ) -> TaskStepBundle:
        return TaskStepBundle(
            support=self._build_support_step(support_steps),
            inference=self._build_single_role_step(
                "inference",
                lambda: InferenceStep(inference_role),
                enabled=inference_role is not None,
            ),
            arena=self._build_single_role_step(
                "arena",
                lambda: ArenaStep(arena_role),
                enabled=arena_role is not None,
            ),
            judge=self._build_single_role_step(
                "judge",
                lambda: JudgeStep(judge_role),
                enabled=judge_role is not None,
            ),
            auto_eval_step=auto_eval_step,
        )

    def _build_support_step(self, support_steps: Sequence[dict]) -> Optional[SupportStep]:
        if not support_steps:
            return None
        try:
            snapshot = tuple(
                deepcopy(step) if isinstance(step, dict) else step
                for step in support_steps
            )
            return self._freeze_task_scope_step(SupportStep(snapshot))
        except Exception as exc:
            raise ValueError(f"Failed to build task-scoped step 'support': {exc}") from exc

    def _build_single_role_step(self, step_name: str, factory, *, enabled: bool):
        if not enabled:
            return None
        try:
            return self._freeze_task_scope_step(factory())
        except Exception as exc:
            raise ValueError(f"Failed to build task-scoped step '{step_name}': {exc}") from exc

    @staticmethod
    def _freeze_task_scope_step(step):
        if hasattr(step, "freeze"):
            step.freeze()
        return step
