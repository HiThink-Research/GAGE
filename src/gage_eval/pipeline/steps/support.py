"""Support step responsible for helper/context/toolchain/modal operations."""

from __future__ import annotations

from typing import Any, Dict, Optional, Sequence

from loguru import logger
from gage_eval.observability.trace import ObservabilityTrace
from gage_eval.evaluation.support_artifacts import build_support_slot_id, record_support_output
from gage_eval.pipeline.step_contracts import get_step_adapter_id
from gage_eval.pipeline.steps._role_borrow import borrow_role_with_optional_context
from gage_eval.pipeline.steps.base import SampleStep
from gage_eval.registry import registry
from gage_eval.role.runtime.invocation import SampleExecutionContext
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
        self.static_only = True
        self.support_payload_policy: Dict[str, Any] = {}

    def execute(
        self,
        sample: dict,
        role_manager,
        trace: ObservabilityTrace,
        *,
        support_payload_policy: Optional[Dict[str, Any]] = None,
        execution_context: Optional[SampleExecutionContext] = None,
        sandbox_provider: Optional[SandboxProvider] = None,
    ) -> None:
        for step in self._steps:
            self._execute_single(
                step,
                sample,
                role_manager,
                trace,
                support_payload_policy=support_payload_policy,
                execution_context=execution_context,
                sandbox_provider=sandbox_provider,
            )

    def execute_single(
        self,
        step,
        sample: dict,
        role_manager,
        trace: ObservabilityTrace,
        *,
        support_payload_policy: Optional[Dict[str, Any]] = None,
        execution_context: Optional[SampleExecutionContext] = None,
        sandbox_provider: Optional[SandboxProvider] = None,
    ) -> None:
        self._execute_single(
            step,
            sample,
            role_manager,
            trace,
            support_payload_policy=support_payload_policy,
            execution_context=execution_context,
            sandbox_provider=sandbox_provider,
        )

    def _execute_single(
        self,
        step,
        sample: dict,
        role_manager,
        trace: ObservabilityTrace,
        *,
        support_payload_policy: Optional[Dict[str, Any]] = None,
        execution_context: Optional[SampleExecutionContext] = None,
        sandbox_provider: Optional[SandboxProvider] = None,
    ) -> None:
        adapter_id = get_step_adapter_id(step)
        slot_id = _resolve_support_slot_id(step, self._steps)
        logger.debug("Support step start adapter_id={}", adapter_id)
        trace.emit("support_start", payload={"step": _serialize_step(step), "slot_id": slot_id})
        if adapter_id:
            invocation_context = (
                execution_context.for_invocation(
                    step_type="support",
                    adapter_id=str(adapter_id),
                    step_slot_id=slot_id,
                )
                if execution_context is not None
                else None
            )
            with borrow_role_with_optional_context(
                role_manager,
                adapter_id,
                execution_context=invocation_context,
            ) as role:
                payload = {"sample": sample, "step": step}
                if sandbox_provider is not None:
                    payload["sandbox_provider"] = sandbox_provider
                output = role.invoke(payload, trace) if role else {}
                if isinstance(output, dict):
                    record_support_output(
                        sample,
                        slot_id=slot_id,
                        adapter_id=str(adapter_id),
                        output=output,
                        policy=support_payload_policy,
                    )
                _emit_tool_doc_metrics(trace, adapter_id, sample, output)
                _emit_observability_events(trace, sample, output)
                logger.trace("Support step output appended keys={}", list(output.keys()))
        trace.emit("support_end", payload={"step": _serialize_step(step), "slot_id": slot_id})
        logger.debug("Support step end adapter_id={}", adapter_id)


def _serialize_step(step):
    if isinstance(step, dict):
        return dict(step)
    attrs = {}
    for key in ("step", "step_type", "adapter_id", "role_ref", "params"):
        if hasattr(step, key):
            attrs[key] = getattr(step, key)
    return attrs or str(step)


def _resolve_support_slot_id(step, steps: Sequence[Any]) -> str:
    params = step.get("params") if hasattr(step, "get") else getattr(step, "params", {})
    if isinstance(params, dict):
        candidate = params.get("support_slot_id")
        if isinstance(candidate, str) and candidate:
            return candidate
    for ordinal, item in enumerate(steps):
        if item is step:
            return build_support_slot_id(item, ordinal)
    return build_support_slot_id(step, 0)


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
