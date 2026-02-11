"""Inference step that orchestrates DUT model calls."""

from __future__ import annotations

from typing import Optional

from loguru import logger

from gage_eval.observability.trace import ObservabilityTrace
from gage_eval.pipeline.steps.base import SampleStep
from gage_eval.registry import registry
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
        sandbox_provider: Optional[SandboxProvider] = None,
    ):
        trace.emit("inference_start", payload={"adapter_id": self._adapter_id})
        logger.debug("Inference step started adapter_id={}", self._adapter_id)
        payload = {"sample": sample}
        if sandbox_provider is not None:
            payload["sandbox_provider"] = sandbox_provider
        with role_manager.borrow_role(self._adapter_id) as role:
            output = role.invoke(payload, trace) if role else {}
        if isinstance(output, dict) and output.get("error"):
            error_text = str(output.get("error"))
            trace.emit(
                "inference_error",
                payload={
                    "adapter_id": self._adapter_id,
                    "error_type": "backend_error",
                    "failure_reason": "backend_returned_error",
                    "error": error_text,
                },
            )
            logger.error(
                "Inference step failed adapter_id={} error_type=backend_error error={}",
                self._adapter_id,
                error_text,
            )
            raise RuntimeError(f"inference backend returned error: {error_text}")
        trace.emit("inference_end", payload={"adapter_id": self._adapter_id})
        logger.debug("Inference step finished adapter_id={}", self._adapter_id)
        return output
