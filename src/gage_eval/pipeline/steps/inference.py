"""Inference step that orchestrates DUT model calls."""

from __future__ import annotations

from typing import Optional

from loguru import logger

from gage_eval.observability.trace import ObservabilityTrace
from gage_eval.pipeline.steps._backend_error import raise_for_backend_error
from gage_eval.pipeline.steps.base import SampleStep
from gage_eval.registry import registry
from gage_eval.role.runtime.invocation import SampleExecutionContext
from gage_eval.sandbox.provider import SandboxProvider


@registry.asset(
    "pipeline_steps",
    "inference",
    desc="Pipeline step that runs DUT model inference",
    tags=("dut",),
    step_kind="sample",
)
class InferenceStep(SampleStep):
    def __init__(self, adapter_id: Optional[str]) -> None:
        super().__init__("InferenceStep")
        self._adapter_id = adapter_id

    def execute(
        self,
        sample: dict,
        role_manager,
        trace: ObservabilityTrace,
        *,
        execution_context: Optional[SampleExecutionContext] = None,
        sandbox_provider: Optional[SandboxProvider] = None,
    ):
        trace.emit("inference_start", payload={"adapter_id": self._adapter_id})
        logger.debug("Inference step started adapter_id={}", self._adapter_id)
        payload = {"sample": sample}
        if sandbox_provider is not None:
            payload["sandbox_provider"] = sandbox_provider
        invocation_context = (
            execution_context.for_invocation(
                step_type="inference",
                adapter_id=str(self._adapter_id),
            )
            if execution_context is not None and self._adapter_id
            else None
        )
        with role_manager.borrow_role(
            self._adapter_id,
            execution_context=invocation_context,
        ) as role:
            output = role.invoke(payload, trace) if role else {}
        raise_for_backend_error(
            event_prefix="inference",
            step_label="Inference step",
            adapter_id=self._adapter_id,
            output=output,
            trace=trace,
        )
        trace.emit("inference_end", payload={"adapter_id": self._adapter_id})
        logger.debug("Inference step finished adapter_id={}", self._adapter_id)
        return output
