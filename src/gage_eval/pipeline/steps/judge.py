"""Judge step executing judge_model role adapters."""

from __future__ import annotations

from typing import Optional

from loguru import logger

from gage_eval.evaluation.sample_envelope import update_eval_result
from gage_eval.evaluation.sample_envelope import resolve_model_output
from gage_eval.evaluation.sample_envelope import resolve_runtime_judge_output
from gage_eval.observability.trace import ObservabilityTrace
from gage_eval.pipeline.steps._backend_error import raise_for_backend_error
from gage_eval.pipeline.steps._role_borrow import borrow_role_with_optional_context
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
        self.adapter_id = adapter_id
        self.judge_binding_resolver = None
        self.failure_interceptor = None
        self.result_writer = None

    def execute(
        self,
        payload: dict,
        role_manager,
        trace: ObservabilityTrace,
        *,
        execution_context: Optional[SampleExecutionContext] = None,
    ):
        model_output = resolve_model_output(payload.get("sample"), payload.get("model_output"))
        runtime_outcome = model_output.get("runtime_judge_outcome")
        if self._resolve_runtime_binding(runtime_outcome):
            trace.emit("judge_runtime_binding", payload={"adapter_id": self._adapter_id})
            judge_output = runtime_outcome.get("judge_output")
            if not isinstance(judge_output, dict):
                judge_output = self._intercept_runtime_failure(runtime_outcome)
            self._write_result(payload.get("sample"), judge_output)
            if isinstance(judge_output, dict):
                return judge_output
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
        with borrow_role_with_optional_context(
            role_manager,
            self._adapter_id,
            execution_context=invocation_context,
        ) as role:
            result = role.invoke(payload, trace) if role else {}
        raise_for_backend_error(
            event_prefix="judge",
            step_label="Judge step",
            adapter_id=self._adapter_id,
            output=result,
            trace=trace,
        )
        self._write_result(payload.get("sample"), result)
        trace.emit("judge_end", payload={"adapter_id": self._adapter_id})
        logger.debug("Judge step finished adapter_id={}", self._adapter_id)
        return result

    def _resolve_runtime_binding(self, runtime_outcome: object) -> bool:
        """Resolves whether the current sample should bind a runtime outcome."""

        if callable(self.judge_binding_resolver):
            resolved = self.judge_binding_resolver(runtime_outcome)
            return bool(resolved)
        return isinstance(runtime_outcome, dict)

    def _intercept_runtime_failure(self, runtime_outcome: object) -> dict:
        """Normalizes runtime-owned failure outcomes into judge output."""

        if callable(self.failure_interceptor):
            intercepted = self.failure_interceptor(runtime_outcome)
            if isinstance(intercepted, dict):
                return intercepted

        return resolve_runtime_judge_output({"runtime_judge_outcome": runtime_outcome})

    def _write_result(self, sample: object, judge_output: object) -> None:
        """Writes the normalized judge result back to the sample envelope."""

        if not isinstance(sample, dict) or not isinstance(judge_output, dict):
            return
        if callable(self.result_writer):
            self.result_writer(sample, judge_output)
            return
        update_eval_result(sample, judge_output)
