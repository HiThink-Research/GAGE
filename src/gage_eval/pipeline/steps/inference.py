"""Inference step that orchestrates DUT model calls."""

from __future__ import annotations

from typing import Optional

from loguru import logger

from gage_eval.observability.trace import ObservabilityTrace
from gage_eval.pipeline.steps.base import SampleStep
from gage_eval.registry import registry


@registry.asset(
    "pipeline_steps",
    "inference",
    desc="执行 DUT 模型推理的阶段",
    tags=("dut",),
    step_kind="sample",
)
class InferenceStep(SampleStep):
    def __init__(self, adapter_id: Optional[str]) -> None:
        super().__init__("InferenceStep")
        self._adapter_id = adapter_id

    def execute(self, sample: dict, role_manager, trace: ObservabilityTrace):
        trace.emit("inference_start", payload={"adapter_id": self._adapter_id})
        logger.debug("Inference step started adapter_id={}", self._adapter_id)
        payload = {"sample": sample}
        with role_manager.borrow_role(self._adapter_id) as role:
            output = role.invoke(payload, trace) if role else {}
        trace.emit("inference_end", payload={"adapter_id": self._adapter_id})
        logger.debug("Inference step finished adapter_id={}", self._adapter_id)
        return output
