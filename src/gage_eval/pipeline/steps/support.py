"""Support step responsible for helper/context/toolchain/modal operations."""

from __future__ import annotations

from typing import Dict, Sequence

from loguru import logger
from gage_eval.observability.trace import ObservabilityTrace
from gage_eval.pipeline.steps.base import SampleStep
from gage_eval.registry import registry


@registry.asset(
    "pipeline_steps",
    "support",
    desc="执行 Helper/Toolchain 等辅助角色的阶段",
    tags=("support",),
    step_kind="sample",
)
class SupportStep(SampleStep):
    """Executes the helper pipeline declared for a sample."""

    def __init__(self, steps: Sequence[Dict[str, str]]) -> None:
        super().__init__("SupportStep")
        self._steps = steps

    def execute(self, sample: dict, role_manager, trace: ObservabilityTrace) -> None:
        for step in self._steps:
            self._execute_single(step, sample, role_manager, trace)

    def execute_single(self, step, sample: dict, role_manager, trace: ObservabilityTrace) -> None:
        self._execute_single(step, sample, role_manager, trace)

    def _execute_single(self, step, sample: dict, role_manager, trace: ObservabilityTrace) -> None:
        adapter_id = step.get("adapter_id")
        logger.debug("Support step start adapter_id={}", adapter_id)
        trace.emit("support_start", payload={"step": _serialize_step(step)})
        if adapter_id:
            with role_manager.borrow_role(adapter_id) as role:
                output = role.invoke({"sample": sample, "step": step}, trace) if role else {}
                sample.setdefault("support_outputs", []).append(output)
                logger.trace("Support step output appended keys={}", list(output.keys()))
        trace.emit("support_end", payload={"step": _serialize_step(step)})
        logger.debug("Support step end adapter_id={}", adapter_id)


def _serialize_step(step):
    if isinstance(step, dict):
        return dict(step)
    attrs = {}
    for key in ("step", "step_type", "adapter_id", "role_ref", "params"):
        if hasattr(step, key):
            attrs[key] = getattr(step, key)
    return attrs or str(step)
