"""Support step responsible for helper/context/toolchain/modal operations."""

from __future__ import annotations

from typing import Dict, Optional, Sequence

from loguru import logger
from gage_eval.observability.trace import ObservabilityTrace
from gage_eval.pipeline.steps.base import SampleStep
from gage_eval.registry import registry
from gage_eval.sandbox.provider import SandboxProvider


@registry.asset(
    "pipeline_steps",
    "support",
    desc="Pipeline step that runs helper/toolchain roles",
    tags=("support",),
    step_kind="sample",
)
class SupportStep(SampleStep):
    """Executes the helper pipeline declared for a sample."""

    def __init__(self, steps: Sequence[Dict[str, str]]) -> None:
        super().__init__("SupportStep")
        self._steps = steps

    def execute(
        self,
        sample: dict,
        role_manager,
        trace: ObservabilityTrace,
        *,
        sandbox_provider: Optional[SandboxProvider] = None,
    ) -> None:
        for step in self._steps:
            self._execute_single(step, sample, role_manager, trace, sandbox_provider=sandbox_provider)

    def execute_single(
        self,
        step,
        sample: dict,
        role_manager,
        trace: ObservabilityTrace,
        *,
        sandbox_provider: Optional[SandboxProvider] = None,
    ) -> None:
        self._execute_single(step, sample, role_manager, trace, sandbox_provider=sandbox_provider)

    def _execute_single(
        self,
        step,
        sample: dict,
        role_manager,
        trace: ObservabilityTrace,
        *,
        sandbox_provider: Optional[SandboxProvider] = None,
    ) -> None:
        adapter_id = step.get("adapter_id")
        logger.debug("Support step start adapter_id={}", adapter_id)
        trace.emit("support_start", payload={"step": _serialize_step(step)})
        if adapter_id:
            with role_manager.borrow_role(adapter_id) as role:
                payload = {"sample": sample, "step": step}
                if sandbox_provider is not None:
                    payload["sandbox_provider"] = sandbox_provider
                output = role.invoke(payload, trace) if role else {}
                sample.setdefault("support_outputs", []).append(output)
                _emit_tool_doc_metrics(trace, adapter_id, sample, output)
                _emit_observability_events(trace, sample, output)
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


def _emit_tool_doc_metrics(trace: ObservabilityTrace, adapter_id: Optional[str], sample: dict, output: dict) -> None:
    if not isinstance(output, dict):
        return
    meta = output.get("tool_documentation_meta")
    if not isinstance(meta, dict) or not meta:
        return
    payload = dict(meta)
    if adapter_id:
        payload.setdefault("adapter_id", adapter_id)
    sample_id = sample.get("id") if isinstance(sample, dict) else None
    trace.emit_tool_documentation(payload, sample_id=sample_id)


def _emit_observability_events(trace: ObservabilityTrace, sample: dict, output: dict) -> None:
    if not isinstance(output, dict):
        return
    events = output.get("observability_events")
    if not isinstance(events, list):
        return
    sample_id = sample.get("id") if isinstance(sample, dict) else None
    for item in events:
        if not isinstance(item, dict):
            continue
        name = item.get("event")
        payload = item.get("payload")
        if not name or not isinstance(payload, dict):
            continue
        trace.emit(str(name), payload, sample_id=sample_id)
