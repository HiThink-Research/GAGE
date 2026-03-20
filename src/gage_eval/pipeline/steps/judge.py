"""Judge step executing judge_model role adapters."""

from __future__ import annotations

from typing import Optional

from loguru import logger

from gage_eval.observability.trace import ObservabilityTrace
from gage_eval.pipeline.steps.base import SampleStep
from gage_eval.registry import registry
from gage_eval.role.runtime.invocation import SampleExecutionContext


@registry.asset(
    "pipeline_steps",
    "judge",
    desc="Pipeline step that runs the judge role",
    tags=("judge",),
    step_kind="sample",
)
class JudgeStep(SampleStep):
    def __init__(self, adapter_id: Optional[str]) -> None:
        super().__init__("JudgeStep")
        self._adapter_id = adapter_id

    def execute(
        self,
        payload: dict,
        role_manager,
        trace: ObservabilityTrace,
        *,
        execution_context: Optional[SampleExecutionContext] = None,
    ):
        trace.emit("judge_start", payload={"adapter_id": self._adapter_id})
        logger.debug("Judge step started adapter_id={}", self._adapter_id)
        invocation_context = (
            execution_context.for_invocation(
                step_type="judge",
                adapter_id=str(self._adapter_id),
            )
            if execution_context is not None and self._adapter_id
            else None
        )
        with role_manager.borrow_role(
            self._adapter_id,
            execution_context=invocation_context,
        ) as role:
            result = role.invoke(payload, trace) if role else {}
        trace.emit("judge_end", payload={"adapter_id": self._adapter_id})
        logger.debug("Judge step finished adapter_id={}", self._adapter_id)
        return result
