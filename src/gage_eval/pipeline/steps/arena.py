"""Arena step that orchestrates interactive game loops."""

from __future__ import annotations

from typing import Optional

from loguru import logger

from gage_eval.observability.trace import ObservabilityTrace
from gage_eval.pipeline.steps.base import SampleStep
from gage_eval.registry import registry
from gage_eval.sandbox.provider import SandboxProvider


@registry.asset(
    "pipeline_steps",
    "arena",
    desc="Pipeline step that runs arena game loops",
    tags=("arena",),
    step_kind="sample",
)
class ArenaStep(SampleStep):
    """Executes an arena role adapter for the current sample."""

    def __init__(self, adapter_id: Optional[str]) -> None:
        super().__init__("ArenaStep")
        self._adapter_id = adapter_id

    def execute(
        self,
        sample: dict,
        role_manager,
        trace: ObservabilityTrace,
        *,
        sandbox_provider: Optional[SandboxProvider] = None,
    ):
        trace.emit("arena_start", payload={"adapter_id": self._adapter_id})
        logger.debug("Arena step started adapter_id={}", self._adapter_id)
        payload = {"sample": sample, "role_manager": role_manager, "trace": trace}
        if sandbox_provider is not None:
            payload["sandbox_provider"] = sandbox_provider
        with role_manager.borrow_role(self._adapter_id) as role:
            output = role.invoke(payload, trace) if role else {}
        trace.emit("arena_end", payload={"adapter_id": self._adapter_id})
        logger.debug("Arena step finished adapter_id={}", self._adapter_id)
        return output
