from __future__ import annotations

import inspect
import json
import re
import time
from dataclasses import replace
from typing import Any

from gage_eval.assets.prompts.renderers import PromptContext
from gage_eval.agent_runtime.contracts.failure import FailureEnvelope, FailureEnvelopeError
from gage_eval.agent_runtime.contracts.scheduler import SchedulerResult
from gage_eval.agent_runtime.failure_mapper import FailureMapper
from gage_eval.agent_runtime.session import AgentRuntimeSession
from gage_eval.agent_runtime.tooling.contracts import ToolCallIR, ToolExecutionContext, ToolResultIR, ToolSchemaIR, ToolingError
from gage_eval.agent_runtime.tooling.provider_adapters import OpenAIProviderAdapter, Tau2ToolDialectParser
from gage_eval.agent_runtime.tooling.registry import RuntimeToolRegistry
from gage_eval.agent_runtime.serialization import to_json_compatible
from gage_eval.agent_runtime.trace_schema import TRACE_INLINE_TEXT_LIMIT_BYTES
from gage_eval.registry.utils import run_sync


_DEFAULT_MAX_OBSERVATION_CHARS = 100_000
_TRUNCATED_OBSERVATION_TIP = (
    "Tip: use `head -n 50`, `tail -n 50`, `grep -n PATTERN`, or "
    "`sed -n 'A,Bp'` to read specific ranges."
)


class StaticModelBackendAdapter:
    """Adapts materialized static model backends for the framework agent loop."""

    def __init__(self, static_backend: Any) -> None:
        self.static_backend = static_backend

    def invoke(self, payload: dict[str, Any]) -> dict[str, Any]:
        request = self._build_request(payload)
        backend_call = getattr(self.static_backend, "invoke", None)
        if callable(backend_call):
            return self._normalize_response(backend_call(request))
        if callable(self.static_backend):
            return self._normalize_response(self.static_backend(request))
        async_backend_call = getattr(self.static_backend, "ainvoke", None)
        if callable(async_backend_call):
            return self._normalize_response(run_sync(async_backend_call(request)))
        raise RuntimeError("static_model_backend_missing_invoke")

    async def ainvoke(self, payload: dict[str, Any]) -> dict[str, Any]:
        request = self._build_request(payload)
        async_backend_call = getattr(self.static_backend, "ainvoke", None)
        if callable(async_backend_call):
            return self._normalize_response(await async_backend_call(request))
        backend_call = getattr(self.static_backend, "invoke", None)
        if callable(backend_call):
            result = backend_call(request)
        elif callable(self.static_backend):
            result = self.static_backend(request)
        else:
            raise RuntimeError("static_model_backend_missing_invoke")
        if inspect.isawaitable(result):
            result = await result
        return self._normalize_response(result)

    def __call__(self, payload: dict[str, Any]) -> dict[str, Any]:
        return self.invoke(payload)

    def _build_request(self, payload: dict[str, Any]) -> dict[str, Any]:
        request: dict[str, Any] = {
            "messages": list(payload.get("messages") or []),
            "tools": list(payload.get("tools") or []),
            "tool_choice": payload.get("tool_choice"),
        }
        if "sample" in payload:
            request["sample"] = payload.get("sample")
        sampling_params = payload.get("sampling_params")
        if isinstance(sampling_params, dict):
            request["sampling_params"] = dict(sampling_params)
        return request

    def _normalize_response(self, response: Any) -> dict[str, Any]:
        if isinstance(response, dict):
            normalized = dict(response)
            normalized.setdefault("agent_trace", [])
            if "answer" not in normalized:
                normalized["answer"] = _extract_response_answer(normalized)
            if "tool_calls" not in normalized:
                tool_calls = _extract_response_tool_calls(normalized)
                if tool_calls:
                    normalized["tool_calls"] = tool_calls
            normalized.setdefault("answer", "")
            return normalized
        if isinstance(response, str):
            return {"answer": response, "agent_trace": []}
        if response is None:
            return {"answer": "", "agent_trace": []}
        return {"answer": str(response), "agent_trace": []}


class FrameworkLoopScheduler:
    """Runs the internal framework loop as a self-contained scheduler."""

    def __init__(
        self,
        *,
        backend,
        tool_router,
        tool_registry=None,
        prompt_renderer=None,
        max_turns: int = 150,
        pre_hooks=None,
        post_hooks=None,
        mcp_clients: dict[str, Any] | None = None,
    ) -> None:
        self._backend = backend
        self._tool_router = tool_router
        self._tool_registry = tool_registry
        self._prompt_renderer = prompt_renderer
        self._max_turns = max_turns
        self._pre_hooks = pre_hooks
        self._post_hooks = post_hooks
        self._mcp_clients = dict(mcp_clients or {})
        self._failure_mapper = FailureMapper()

    async def arun(
        self,
        *,
        session: AgentRuntimeSession,
        sample: dict[str, Any],
        payload: dict[str, Any],
        workflow_bundle,
        sandbox_provider,
    ) -> SchedulerResult:
        """Execute the framework loop with runtime-owned inputs and tools."""

        # STEP 1: Build messages, prompt context, and tool schema.
        loop_payload = {"messages": list(sample.get("messages") or payload.get("messages") or [])}
        workflow_path = _resolve_workflow_path(session)
        runtime_tool_registry, registry_failure = _resolve_runtime_tool_registry(self._tool_registry, session=session)
        if registry_failure is not None:
            return SchedulerResult(
                scheduler_type="framework_loop",
                benchmark_kit_id=session.benchmark_kit_id,
                status="failed",
                agent_output={},
                failure=registry_failure,
            )
        self._tool_registry = runtime_tool_registry
        if callable(workflow_bundle.build_loop_inputs):
            try:
                built = workflow_bundle.build_loop_inputs(
                    session=session,
                    sample=sample,
                    payload=payload,
                )
            except Exception as exc:
                raise FailureEnvelopeError(
                    self._failure_mapper.map_exception(
                        exc,
                        failure_domain="input_projection",
                        failure_stage="prepare_inputs",
                        component_kind="scheduler",
                        component_id=f"{workflow_bundle.bundle_id}.build_loop_inputs",
                        owner=f"{session.benchmark_kit_id}_kit",
                        failure_code=_workflow_prepare_failure_code(
                            session=session,
                            workflow_bundle=workflow_bundle,
                            step="build_loop_inputs",
                        ),
                        first_bad_step=f"{workflow_bundle.bundle_id}.build_loop_inputs",
                        suspect_files=(
                            workflow_path,
                            "src/gage_eval/agent_runtime/schedulers/framework_loop.py",
                        ),
                    )
                ) from exc
            if isinstance(built, dict):
                loop_payload.update(built)
        if callable(workflow_bundle.inject_prompt_context):
            try:
                prompt_context = workflow_bundle.inject_prompt_context(
                    session=session,
                    sample=sample,
                    payload=payload,
                )
            except Exception as exc:
                raise FailureEnvelopeError(
                    self._failure_mapper.map_exception(
                        exc,
                        failure_domain="input_projection",
                        failure_stage="prepare_inputs",
                        component_kind="scheduler",
                        component_id=f"{workflow_bundle.bundle_id}.inject_prompt_context",
                        owner=f"{session.benchmark_kit_id}_kit",
                        failure_code=_workflow_prepare_failure_code(
                            session=session,
                            workflow_bundle=workflow_bundle,
                            step="inject_prompt_context",
                        ),
                        first_bad_step=f"{workflow_bundle.bundle_id}.inject_prompt_context",
                        suspect_files=(
                            workflow_path,
                            "src/gage_eval/agent_runtime/schedulers/framework_loop.py",
                        ),
                    )
                ) from exc
            if isinstance(prompt_context, dict):
                session.prompt_context.update(prompt_context)
        tools_failure: FailureEnvelope | None = None
        should_collect_dynamic_tools = not runtime_tool_registry.entries() or session.benchmark_kit_id in {"tau2", "appworld"}
        if callable(workflow_bundle.inject_tool_schemas) and should_collect_dynamic_tools:
            try:
                injected_tools = workflow_bundle.inject_tool_schemas(
                    session=session,
                    sample=sample,
                    payload=payload,
                )
            except Exception as exc:
                raise FailureEnvelopeError(
                    self._failure_mapper.map_exception(
                        exc,
                        failure_domain="input_projection",
                        failure_stage="prepare_inputs",
                        component_kind="scheduler",
                        component_id=f"{workflow_bundle.bundle_id}.inject_tool_schemas",
                        owner=f"{session.benchmark_kit_id}_kit",
                        failure_code=_workflow_prepare_failure_code(
                            session=session,
                            workflow_bundle=workflow_bundle,
                            step="inject_tool_schemas",
                        ),
                        first_bad_step=f"{workflow_bundle.bundle_id}.inject_tool_schemas",
                        suspect_files=(
                            workflow_path,
                            "src/gage_eval/agent_runtime/schedulers/framework_loop.py",
                        ),
                    )
                ) from exc
            if isinstance(injected_tools, list):
                tools_failure = _register_raw_tool_schemas(
                    runtime_tool_registry,
                    injected_tools,
                    session=session,
                    provider="workflow",
                    mcp_clients=self._mcp_clients,
                )

        if tools_failure is not None:
            return SchedulerResult(
                scheduler_type="framework_loop",
                benchmark_kit_id=session.benchmark_kit_id,
                status="failed",
                agent_output={},
                failure=tools_failure,
            )
        tools = runtime_tool_registry.project_tool_schemas()

        if self._prompt_renderer is not None:
            try:
                rendered_messages, system_prompt = self._render_messages(
                    session=session,
                    sample=sample,
                    payload=payload,
                    loop_messages=list(loop_payload.get("messages") or []),
                )
            except Exception as exc:
                raise FailureEnvelopeError(
                    self._failure_mapper.map_exception(
                        exc,
                        failure_domain="input_projection",
                        failure_stage="prepare_inputs",
                        component_kind="scheduler",
                        component_id=f"{workflow_bundle.bundle_id}.render_prompt",
                        owner="runtime_scheduler_core",
                        failure_code=(
                            f"input_projection.prepare_inputs."
                            f"{workflow_bundle.bundle_id}.render_prompt_failed"
                        ),
                        first_bad_step=f"{workflow_bundle.bundle_id}.render_prompt",
                        suspect_files=(
                            "src/gage_eval/agent_runtime/schedulers/framework_loop.py",
                        ),
                    )
                ) from exc
            loop_payload["messages"] = rendered_messages
            if system_prompt:
                session.scheduler_state["system_prompt"] = system_prompt

        # STEP 2: Execute the runtime tooling loop using ToolCallIR/ToolResultIR directly.
        try:
            raw_result = await _run_runtime_tooling_loop(
                backend=self._backend,
                tool_router=self._tool_router,
                session=session,
                sample=sample,
                payload=payload,
                loop_payload=loop_payload,
                tools=tools,
                max_turns=self._max_turns,
            )
        except ToolingError as exc:
            raise FailureEnvelopeError(
                _tooling_failure(
                    session=session,
                    workflow_bundle=workflow_bundle,
                    error=exc,
                )
            ) from exc
        except TimeoutError as exc:
            raise FailureEnvelopeError(
                _model_request_timeout_failure(
                    session=session,
                    workflow_bundle=workflow_bundle,
                    error=exc,
                )
            ) from exc
        except Exception as exc:
            if _is_timeout_exception(exc):
                raise FailureEnvelopeError(
                    _model_request_timeout_failure(
                        session=session,
                        workflow_bundle=workflow_bundle,
                        error=exc,
                    )
                ) from exc
            raise FailureEnvelopeError(
                self._failure_mapper.map_exception(
                    exc,
                    failure_domain="client_execution",
                    failure_stage="run_scheduler",
                    component_kind="scheduler",
                    component_id=f"{workflow_bundle.bundle_id}.agent_loop.run",
                    owner="runtime_scheduler_core",
                    failure_code=(
                        f"client_execution.run_scheduler."
                        f"{workflow_bundle.bundle_id}.agent_loop_failed"
                    ),
                    first_bad_step=f"{workflow_bundle.bundle_id}.agent_loop.run",
                    suspect_files=(
                        "src/gage_eval/agent_runtime/schedulers/framework_loop.py",
                    ),
                )
            ) from exc

        # STEP 3: Normalize the loop output and artifact mappings.
        finalized_output = raw_result
        if callable(workflow_bundle.finalize_loop_result):
            try:
                finalized = workflow_bundle.finalize_loop_result(
                    session=session,
                    sample=sample,
                    scheduler_output=raw_result,
                    sandbox_provider=sandbox_provider,
                )
            except Exception as exc:
                raise FailureEnvelopeError(
                    self._failure_mapper.map_exception(
                        exc,
                        failure_domain="artifact_capture",
                        failure_stage="normalize_result",
                        component_kind="scheduler",
                        component_id=f"{workflow_bundle.bundle_id}.finalize_loop_result",
                        owner=f"{session.benchmark_kit_id}_kit",
                        failure_code=(
                            f"artifact_capture.normalize_result."
                            f"{workflow_bundle.bundle_id}.finalize_loop_result_failed"
                        ),
                        first_bad_step=f"{workflow_bundle.bundle_id}.finalize_loop_result",
                        suspect_files=(
                            workflow_path,
                            "src/gage_eval/agent_runtime/schedulers/framework_loop.py",
                        ),
                    )
                ) from exc
            if isinstance(finalized, dict):
                finalized_output = finalized
        system_prompt = session.scheduler_state.get("system_prompt")
        if isinstance(finalized_output, dict) and isinstance(system_prompt, str) and system_prompt.strip():
            finalized_output.setdefault("system_prompt", system_prompt)
        artifact_paths = dict(finalized_output.get("artifact_paths") or {})
        runtime_state = dict(session.benchmark_state or {})
        finalized_runtime_state = finalized_output.get("runtime_state")
        if isinstance(finalized_runtime_state, dict):
            runtime_state.update(finalized_runtime_state)
        status = str(finalized_output.get("status") or raw_result.get("status") or "completed")
        if status not in {"completed", "failed", "aborted"}:
            status = "completed"
        if callable(workflow_bundle.failure_normalizer):
            normalized = workflow_bundle.failure_normalizer(
                session=session,
                sample=sample,
                scheduler_output=finalized_output,
                artifact_paths=artifact_paths,
                sandbox_provider=sandbox_provider,
            )
            if isinstance(normalized, dict):
                status = str(normalized.get("status") or status)
                normalized_artifacts = normalized.get("artifact_paths")
                if isinstance(normalized_artifacts, dict):
                    artifact_paths.update(
                        {str(key): str(value) for key, value in normalized_artifacts.items() if value}
                    )
                normalized_runtime_state = normalized.get("runtime_state")
                if isinstance(normalized_runtime_state, dict):
                    runtime_state.update(normalized_runtime_state)
                normalized_output = normalized.get("agent_output")
                if isinstance(normalized_output, dict):
                    finalized_output = normalized_output
        return SchedulerResult(
            scheduler_type="framework_loop",
            benchmark_kit_id=session.benchmark_kit_id,
            status=status,  # type: ignore[arg-type]
            agent_output=finalized_output,
            artifact_paths=artifact_paths,
            runtime_state=runtime_state,
        )

    def _render_messages(
        self,
        *,
        session: AgentRuntimeSession,
        sample: dict[str, Any],
        payload: dict[str, Any],
        loop_messages: list[dict[str, Any]],
    ) -> tuple[list[dict[str, Any]], str | None]:
        """Renders loop messages through the optional prompt renderer."""

        render_sample = dict(sample or {})
        if session.prompt_context:
            render_sample["prompt_context"] = dict(session.prompt_context)
        if session.runtime_context:
            render_sample["runtime_context"] = dict(session.runtime_context)

        render_payload = dict(payload or {})
        render_payload["messages"] = list(loop_messages)
        render_payload.setdefault("instruction", session.prompt_context.get("instruction") or sample.get("instruction"))
        render_payload.setdefault("max_steps", self._max_turns)
        if session.prompt_context:
            render_payload.setdefault("prompt_context", dict(session.prompt_context))
            for key, value in session.prompt_context.items():
                render_payload.setdefault(str(key), value)
        if session.runtime_context:
            render_payload.setdefault("runtime_context", dict(session.runtime_context))

        rendered = self._prompt_renderer.render(
            PromptContext(
                sample=render_sample,
                payload=render_payload,
                history=[],
                extras={
                    "scheduler_type": session.scheduler_type,
                    "benchmark_kit_id": session.benchmark_kit_id,
                },
            )
        )
        if rendered.messages is not None:
            return rendered.messages, _extract_system_prompt(rendered.messages)
        if rendered.prompt:
            messages = [{"role": "system", "content": rendered.prompt}] + list(loop_messages)
            return messages, rendered.prompt
        return list(loop_messages), None


def _resolve_runtime_tool_registry(
    tool_registry: Any,
    *,
    session: AgentRuntimeSession,
) -> tuple[RuntimeToolRegistry, FailureEnvelope | None]:
    """Resolve scheduler-facing tools to the runtime registry contract."""

    if tool_registry is None:
        return RuntimeToolRegistry(), None
    if isinstance(tool_registry, RuntimeToolRegistry):
        return tool_registry, None
    return RuntimeToolRegistry(), _tool_schema_invalid_failure(
        session=session,
        summary="framework_loop received tools outside RuntimeToolRegistry",
        details={"tool_registry_type": tool_registry.__class__.__name__},
    )


def _workflow_prepare_failure_code(
    *,
    session: AgentRuntimeSession,
    workflow_bundle: Any,
    step: str,
) -> str:
    if session.benchmark_kit_id == "swebench":
        return "input_projection.workflow.prepare_failed"
    return f"input_projection.prepare_inputs.{workflow_bundle.bundle_id}.{step}_failed"


def _register_raw_tool_schemas(
    registry: RuntimeToolRegistry,
    raw_tools: list[dict[str, Any]],
    *,
    session: AgentRuntimeSession,
    provider: str,
    mcp_clients: dict[str, Any] | None = None,
) -> FailureEnvelope | None:
    try:
        for raw_tool in raw_tools:
            schema = ToolSchemaIR.from_provider_schema(raw_tool, provider=provider)
            if registry.get(schema.name) is not None:
                continue
            mcp_client_id = _extract_mcp_client_id(raw_tool)
            mcp_client = (mcp_clients or {}).get(mcp_client_id) if mcp_client_id else None
            metadata = _extract_raw_tool_metadata(raw_tool)
            if mcp_client is not None:
                registry.register_mcp_tool(
                    schema,
                    _resolve_mcp_client_ref(mcp_client),
                    metadata={"server_id": mcp_client_id, "source": provider, **metadata},
                )
                continue
            registry.register_schema(schema, provider_kind="environment", metadata=metadata)
    except Exception as exc:
        return _tool_schema_invalid_failure(
            session=session,
            summary="workflow tool schema could not be normalized through RuntimeToolRegistry",
            details={"error_type": exc.__class__.__name__, "error": str(exc)},
        )
    return None


def _extract_mcp_client_id(raw_tool: dict[str, Any]) -> str | None:
    metadata = _extract_raw_tool_metadata(raw_tool)
    if metadata.get("mcp_client_id"):
        return str(metadata["mcp_client_id"])
    return None


def _extract_raw_tool_metadata(raw_tool: dict[str, Any]) -> dict[str, Any]:
    x_gage = raw_tool.get("x-gage")
    if isinstance(x_gage, dict):
        return dict(x_gage)
    function = raw_tool.get("function")
    if isinstance(function, dict) and isinstance(function.get("x-gage"), dict):
        return dict(function["x-gage"])
    return {}


def _resolve_mcp_client_ref(client_ref: Any) -> Any:
    return getattr(client_ref, "client", client_ref)


async def _run_runtime_tooling_loop(
    *,
    backend: Any,
    tool_router: Any,
    session: AgentRuntimeSession,
    sample: dict[str, Any],
    payload: dict[str, Any],
    loop_payload: dict[str, Any],
    tools: list[dict[str, Any]],
    max_turns: int,
) -> dict[str, Any]:
    adapter = OpenAIProviderAdapter()
    messages = list(loop_payload.get("messages") or [])
    tool_choice = loop_payload.get("tool_choice", payload.get("tool_choice"))
    sampling_params = _resolve_sampling_params(loop_payload=loop_payload, payload=payload, sample=sample)
    metadata = payload.get("metadata") or sample.get("metadata") or {}
    usage: dict[str, Any] | None = None
    agent_trace: list[dict[str, Any]] = []
    answer = ""
    sink = _resolve_artifact_sink(session=session, payload=payload)
    trial_id = _resolve_trial_id(session=session, payload=payload)
    required_tool = _required_tool_name(session=session, tools=tools, payload=payload, loop_payload=loop_payload)
    max_observation_chars = _resolve_max_observation_chars(payload=payload, loop_payload=loop_payload)
    cost_limit_usd = _resolve_cost_limit_usd(payload=payload, loop_payload=loop_payload)
    total_cost_usd = 0.0
    retry_budget = int(payload.get("tool_call_retry_budget") or loop_payload.get("tool_call_retry_budget") or 3)
    retry_count = 0

    for turn_index in range(1, max(1, int(max_turns)) + 1):
        request_tool_choice = _effective_tool_choice(
            base_tool_choice=tool_choice,
            payload=payload,
            loop_payload=loop_payload,
            turn_index=turn_index,
            tools=tools,
        )
        request = adapter.serialize_request(
            messages=messages,
            tools=tools,
            tool_choice=request_tool_choice,
            sampling_params=sampling_params or {},
            turn_index=turn_index,
            metadata=metadata,
            sample=sample,
        )
        request_ref = _write_tooling_artifact(
            sink=sink,
            session=session,
            trial_id=trial_id,
            name=f"model_request_turn_{turn_index}.json",
            content=request,
        )
        _append_tooling_trace(
            sink=sink,
            session=session,
            trial_id=trial_id,
            actor="scheduler",
            event_type="model.request",
            payload={
                "turn_index": turn_index,
                "provider": "openai",
                "backend_id": _backend_id(backend),
                "tool_schema_count": len(tools),
                "raw_request_ref": _ref_payload(request_ref),
            },
            artifact_refs=[request_ref] if request_ref is not None else None,
        )
        raw_response = await _invoke_runtime_backend(backend, request)
        response_ref = _write_tooling_artifact(
            sink=sink,
            session=session,
            trial_id=trial_id,
            name=f"model_response_turn_{turn_index}.json",
            content=raw_response,
        )
        turn_usage = _extract_usage(raw_response)
        if turn_usage is not None:
            usage = turn_usage
            total_cost_usd += _usage_cost_usd(turn_usage)
        if cost_limit_usd is not None and total_cost_usd > cost_limit_usd:
            usage = dict(usage or {})
            usage["cost_usd"] = total_cost_usd
            return {
                "answer": "",
                "agent_trace": agent_trace,
                "usage": usage,
                "artifacts": [],
                "status": "failed",
                "loop_exit_reason": "cost_limit",
                "failure_code": "client_execution.cost_limit_exceeded",
                "failure_reason": "cost_limit_exceeded",
                "cost_limit_usd": cost_limit_usd,
            }
        tool_calls = _extract_runtime_tool_calls(
            adapter,
            raw_response,
            backend=backend,
            session=session,
            payload=payload,
            tools=tools,
            turn_index=turn_index,
        )
        _append_tooling_trace(
            sink=sink,
            session=session,
            trial_id=trial_id,
            actor="scheduler",
            event_type="model.response",
            payload={
                "turn_index": turn_index,
                "raw_response_ref": _ref_payload(response_ref),
                "finish_reason": _finish_reason(raw_response),
                "tool_call_count": len(tool_calls),
            },
            artifact_refs=[response_ref] if response_ref is not None else None,
        )
        if not tool_calls:
            answer = adapter.extract_final_answer(raw_response)
            if _should_implicit_tau2_respond(
                session=session,
                payload=payload,
                backend=backend,
                required_tool=required_tool,
                answer=answer,
            ):
                messages.append(_assistant_text_message(answer))
                agent_trace.append(
                    _build_trace_step(
                        len(agent_trace) + 1,
                        trace_role="assistant",
                        name="agent_response",
                        input_payload=None,
                        output_payload={"answer": answer},
                        status="success",
                        latency_ms=0.0,
                        usage=usage,
                        turn_index=turn_index,
                    )
                )
                implicit_call = _implicit_tau2_respond_call(answer, turn_index=turn_index)
                tool_result = await _dispatch_runtime_tool_call(
                    call=implicit_call,
                    tool_router=tool_router,
                    session=session,
                    payload=payload,
                    sink=sink,
                    trial_id=trial_id,
                    turn_index=turn_index,
                    agent_trace=agent_trace,
                )
                if tool_result.status != "success":
                    failure_code = _tool_result_failure_code(tool_result)
                    return {
                        "answer": "",
                        "agent_trace": agent_trace,
                        "usage": usage,
                        "artifacts": [],
                        "status": "failed",
                        "loop_exit_reason": "tool_result_error",
                        "failure_code": failure_code,
                        "failure_reason": tool_result.output_text or failure_code,
                        "required_tool": required_tool,
                    }
                final_answer = _final_answer_from_tool_result(tool_result, required_tool=required_tool)
                if final_answer is not None:
                    return {
                        "answer": final_answer,
                        "agent_trace": agent_trace,
                        "usage": usage,
                        "artifacts": [],
                    }
                _append_tau2_user_message_from_tool_result(
                    messages=messages,
                    tool_result=tool_result,
                    sink=sink,
                    session=session,
                    trial_id=trial_id,
                    turn_index=turn_index,
                )
                retry_count = 0
                continue
            if required_tool is not None:
                retry_count += 1
                agent_trace.append(
                    _build_trace_step(
                        len(agent_trace) + 1,
                        trace_role="assistant",
                        name="missing_required_tool_call",
                        input_payload=None,
                        output_payload={
                            "required_tool": required_tool,
                            "retry_count": retry_count,
                            "failure_code": "client_execution.tool_retry_budget_exhausted",
                        },
                        status="retry_required_tool_call",
                        latency_ms=0.0,
                        usage=usage,
                        turn_index=turn_index,
                    )
                )
                if retry_count >= retry_budget:
                    return {
                        "answer": "",
                        "agent_trace": agent_trace,
                        "usage": usage,
                        "artifacts": [],
                        "status": "failed",
                        "loop_exit_reason": "tool_call_retry_budget",
                        "failure_code": "client_execution.tool_retry_budget_exhausted",
                        "failure_reason": "required_tool_missing",
                        "required_tool": required_tool,
                    }
                continue
            agent_trace.append(
                _build_trace_step(
                    len(agent_trace) + 1,
                    trace_role="assistant",
                    name="agent_response",
                    input_payload=None,
                    output_payload={"answer": answer},
                    status="success",
                    latency_ms=0.0,
                    usage=usage,
                    turn_index=turn_index,
                )
            )
            break

        messages.append(_assistant_message_from_raw_response(raw_response, tool_calls))
        for call in tool_calls:
            tool_result = await _dispatch_runtime_tool_call(
                call=call,
                tool_router=tool_router,
                session=session,
                payload=payload,
                sink=sink,
                trial_id=trial_id,
                turn_index=turn_index,
                agent_trace=agent_trace,
            )
            injected_result = _truncate_tool_result_for_model_injection(
                tool_result,
                max_observation_chars=max_observation_chars,
            )
            injected_message = adapter.serialize_tool_result(injected_result)
            messages.append(injected_message)
            _append_tooling_trace(
                sink=sink,
                session=session,
                trial_id=trial_id,
                actor="scheduler",
                event_type="tool.result.injected",
                payload={"tool_call_id": tool_result.call_id, "message": injected_message},
            )
            final_answer = _final_answer_from_tool_result(tool_result, required_tool=required_tool)
            if final_answer is not None:
                return {
                    "answer": final_answer,
                    "agent_trace": agent_trace,
                    "usage": usage,
                    "artifacts": [],
                }
            _append_tau2_user_message_from_tool_result(
                messages=messages,
                tool_result=tool_result,
                sink=sink,
                session=session,
                trial_id=trial_id,
                turn_index=turn_index,
            )

    if required_tool is not None:
        return {
            "answer": "",
            "agent_trace": agent_trace,
            "usage": usage,
            "artifacts": [],
            "status": "failed",
            "loop_exit_reason": "max_turns",
            "failure_code": "client_execution.tool_retry_budget_exhausted",
            "failure_reason": "max_turns",
            "required_tool": required_tool,
        }
    return {"answer": answer, "agent_trace": agent_trace, "usage": usage, "artifacts": []}


async def _dispatch_runtime_tool_call(
    *,
    call: ToolCallIR,
    tool_router: Any,
    session: AgentRuntimeSession,
    payload: dict[str, Any],
    sink: Any | None,
    trial_id: str,
    turn_index: int,
    agent_trace: list[dict[str, Any]],
) -> ToolResultIR:
    _append_tooling_trace(
        sink=sink,
        session=session,
        trial_id=trial_id,
        actor="agent",
        event_type="tool.call.raw",
        payload={"turn_index": turn_index, "raw_message": call.raw_message},
    )
    _append_tooling_trace(
        sink=sink,
        session=session,
        trial_id=trial_id,
        actor="agent",
        event_type="tool.call.normalized",
        payload={"turn_index": turn_index, "tool_call": _tool_call_payload(call)},
    )
    context = _build_tool_execution_context(
        session=session,
        payload=payload,
    )
    tool_result = await tool_router.dispatch(call, context)
    trace_tool_result = await _spill_oversize_tool_output_for_trace(
        tool_result,
        sink=sink,
        session=session,
        trial_id=trial_id,
        turn_index=turn_index,
    )
    agent_trace.append(
        _build_trace_step(
            len(agent_trace) + 1,
            trace_role="tool",
            name=trace_tool_result.name,
            input_payload=call.arguments(),
            output_payload=trace_tool_result.output_json,
            status=trace_tool_result.status,
            latency_ms=float(trace_tool_result.metadata.get("latency_ms") or 0.0),
            turn_index=turn_index,
        )
    )
    _append_tooling_trace(
        sink=sink,
        session=session,
        trial_id=trial_id,
        actor="runtime",
        event_type="tool.result",
        payload={
            "tool_result": _tool_result_payload(trace_tool_result),
            "tool_call_id": trace_tool_result.call_id,
            "name": trace_tool_result.name,
            "status": trace_tool_result.status,
            "latency_ms": float(trace_tool_result.metadata.get("latency_ms") or 0.0),
            "artifact_refs": list(trace_tool_result.artifact_refs),
        },
        artifact_refs=list(trace_tool_result.artifact_refs),
    )
    return tool_result


async def _spill_oversize_tool_output_for_trace(
    tool_result: ToolResultIR,
    *,
    sink: Any | None,
    session: AgentRuntimeSession,
    trial_id: str,
    turn_index: int,
) -> ToolResultIR:
    output = tool_result.output_json
    if sink is None or not isinstance(output, dict):
        return tool_result
    writer = getattr(sink, "write_artifact", None)
    if not callable(writer):
        return tool_result

    new_output = dict(output)
    raw_output_refs = new_output.get("output_artifact_refs")
    output_artifact_refs = list(raw_output_refs) if isinstance(raw_output_refs, list) else []
    artifact_refs = list(tool_result.artifact_refs or [])
    changed = False
    for stream in ("stdout", "stderr"):
        text = new_output.get(stream)
        if not isinstance(text, str):
            continue
        encoded_size = len(text.encode("utf-8"))
        if encoded_size <= TRACE_INLINE_TEXT_LIMIT_BYTES:
            continue
        ref = writer(
            run_id=session.run_id,
            task_id=session.task_id,
            sample_id=session.sample_id,
            trial_id=trial_id,
            owner="agent",
            name=_tool_output_artifact_name(
                call_id=tool_result.call_id,
                stream=stream,
                turn_index=turn_index,
            ),
            content=text,
            metadata={
                "call_id": tool_result.call_id,
                "tool": tool_result.name,
                "stream": stream,
                "turn_index": turn_index,
                "original_size_bytes": encoded_size,
            },
            mime_type="text/plain",
        )
        if inspect.isawaitable(ref):
            ref = await ref
        ref_payload = _ref_payload(ref)
        if ref_payload is not None:
            artifact_refs.append(ref)
            output_artifact_refs.append(ref_payload)
            location = ref_payload.get("path") or ref_payload.get("name") or "output_artifact_refs"
        else:
            location = "output_artifact_refs"
        new_output[stream] = (
            f"<spilled to artifact: {location} ({encoded_size} bytes). "
            f"{_TRUNCATED_OBSERVATION_TIP}>"
        )
        changed = True

    if not changed:
        return tool_result
    new_output["output_artifact_refs"] = output_artifact_refs
    return replace(tool_result, output_json=new_output, artifact_refs=artifact_refs)


def _resolve_max_observation_chars(*, payload: dict[str, Any], loop_payload: dict[str, Any]) -> int:
    raw_value = payload.get("max_observation_chars")
    if raw_value is None:
        raw_value = loop_payload.get("max_observation_chars")
    if raw_value is None:
        raw_value = _DEFAULT_MAX_OBSERVATION_CHARS
    try:
        return max(1, int(raw_value))
    except (TypeError, ValueError):
        return _DEFAULT_MAX_OBSERVATION_CHARS


def _resolve_cost_limit_usd(*, payload: dict[str, Any], loop_payload: dict[str, Any]) -> float | None:
    raw_value = payload.get("cost_limit_usd")
    if raw_value is None:
        raw_value = payload.get("cost_limit")
    if raw_value is None:
        raw_value = loop_payload.get("cost_limit_usd")
    if raw_value is None:
        raw_value = loop_payload.get("cost_limit")
    if raw_value is None:
        return None
    try:
        return max(0.0, float(raw_value))
    except (TypeError, ValueError):
        return None


def _usage_cost_usd(usage: dict[str, Any]) -> float:
    try:
        return float(usage.get("cost_usd") or 0.0)
    except (TypeError, ValueError):
        return 0.0


def _effective_tool_choice(
    *,
    base_tool_choice: Any,
    payload: dict[str, Any],
    loop_payload: dict[str, Any],
    turn_index: int,
    tools: list[dict[str, Any]],
) -> Any:
    mode = str(payload.get("force_tool_choice") or loop_payload.get("force_tool_choice") or "never").strip().lower()
    if mode in {"", "never", "false", "0", "none"}:
        return base_tool_choice
    if not tools:
        return base_tool_choice
    if mode in {"always", "required", "true", "1"}:
        return "required"
    if mode in {"first_turn", "first-turn", "once"} and turn_index <= 1:
        return "required"
    return base_tool_choice


def _truncate_tool_result_for_model_injection(
    tool_result: ToolResultIR,
    *,
    max_observation_chars: int,
) -> ToolResultIR:
    output_json, changed = _truncate_observation_payload(
        tool_result.output_json,
        max_observation_chars=max_observation_chars,
    )
    output_text = tool_result.output_text
    if isinstance(output_text, str) and len(output_text) > max_observation_chars:
        output_text = _observation_truncation_placeholder(
            original_size=len(output_text),
            limit=max_observation_chars,
        )
        changed = True
    if not changed:
        return tool_result
    return replace(tool_result, output_json=output_json, output_text=output_text)


def _truncate_observation_payload(value: Any, *, max_observation_chars: int) -> tuple[Any, bool]:
    if isinstance(value, str):
        if len(value) <= max_observation_chars:
            return value, False
        return (
            _observation_truncation_placeholder(
                original_size=len(value),
                limit=max_observation_chars,
            ),
            True,
        )
    if isinstance(value, dict):
        changed = False
        result: dict[str, Any] = {}
        for key, item in value.items():
            truncated, item_changed = _truncate_observation_payload(
                item,
                max_observation_chars=max_observation_chars,
            )
            result[key] = truncated
            changed = changed or item_changed
        return result, changed
    if isinstance(value, list):
        changed = False
        result = []
        for item in value:
            truncated, item_changed = _truncate_observation_payload(
                item,
                max_observation_chars=max_observation_chars,
            )
            result.append(truncated)
            changed = changed or item_changed
        return result, changed
    return value, False


def _observation_truncation_placeholder(*, original_size: int, limit: int) -> str:
    return f"<output truncated: {original_size} chars exceed max_observation_chars={limit}. {_TRUNCATED_OBSERVATION_TIP}>"


def _final_answer_from_tool_result(
    tool_result: ToolResultIR,
    *,
    required_tool: str | None = None,
) -> str | None:
    if tool_result.status != "success":
        return None
    output = tool_result.output_json
    terminal_answer = _terminal_final_answer(output)
    if terminal_answer is not None:
        return terminal_answer
    if not _required_tool_call_completed(tool_result, required_tool=required_tool):
        return None
    if not isinstance(output, dict):
        return None
    value = output.get("final_answer")
    if value is None:
        return None
    return str(value)


def _terminal_final_answer(output: Any) -> str | None:
    if not isinstance(output, dict):
        return None
    value = output.get("final_answer")
    if value is None:
        return None
    if output.get("error") == "tau2_simulation_terminated" or str(value) == "simulation_terminated":
        return str(value)
    return None


def _required_tool_call_completed(tool_result: ToolResultIR, *, required_tool: str | None) -> bool:
    if required_tool is None:
        return True
    if tool_result.name != required_tool:
        return False
    output = tool_result.output_json
    if isinstance(output, dict) and str(output.get("status") or "").lower() == "review_required":
        return False
    return True


def _tool_result_failure_code(tool_result: ToolResultIR) -> str:
    output = tool_result.output_json
    if isinstance(output, dict) and output.get("failure_code"):
        return str(output["failure_code"])
    return "client_execution.tool_router.failed"


def _should_implicit_tau2_respond(
    *,
    session: AgentRuntimeSession,
    payload: dict[str, Any],
    backend: Any,
    required_tool: str | None,
    answer: str,
) -> bool:
    if session.benchmark_kit_id != "tau2" or required_tool != "respond" or not bool(str(answer or "").strip()):
        return False
    allowlist = _normalize_string_set(
        payload.get("plain_text_response_formats") or payload.get("plain_text_wrapper_formats")
    )
    if not allowlist:
        return True
    return _effective_tau2_tool_dialect(payload=payload, backend=backend) in allowlist


def _implicit_tau2_respond_call(answer: str, *, turn_index: int) -> ToolCallIR:
    return ToolCallIR.from_provider_call(
        {
            "id": f"implicit_respond_{turn_index}",
            "type": "function",
            "function": {
                "name": "respond",
                "arguments": {"message": _strip_think_tail(str(answer or ""))},
            },
        },
        turn_index=turn_index,
        call_index=1,
        provider="tau2_plain_text",
    )


def _strip_think_tail(answer: str) -> str:
    match = re.search(r"</think\s*>\s*(?P<tail>.*)", answer, flags=re.DOTALL | re.IGNORECASE)
    if match and match.group("tail").strip():
        return match.group("tail").strip()
    return answer


def _assistant_text_message(answer: str) -> dict[str, Any]:
    return {"role": "assistant", "content": str(answer or "")}


def _append_tau2_user_message_from_tool_result(
    *,
    messages: list[dict[str, Any]],
    tool_result: ToolResultIR,
    sink: Any | None,
    session: AgentRuntimeSession,
    trial_id: str,
    turn_index: int,
) -> bool:
    if session.benchmark_kit_id != "tau2":
        return False
    output = tool_result.output_json
    if not isinstance(output, dict) or output.get("user_message") is None:
        return False
    user_message = {"role": "user", "content": str(output.get("user_message") or "")}
    messages.append(user_message)
    _append_tooling_trace(
        sink=sink,
        session=session,
        trial_id=trial_id,
        actor="scheduler",
        event_type="user.message.injected",
        payload={
            "turn_index": turn_index,
            "source_tool_call_id": tool_result.call_id,
            "message": user_message,
        },
    )
    return True


def _required_tool_name(
    *,
    session: AgentRuntimeSession,
    tools: list[dict[str, Any]],
    payload: dict[str, Any],
    loop_payload: dict[str, Any],
) -> str | None:
    explicit = payload.get("required_tool") or loop_payload.get("required_tool")
    if explicit:
        return str(explicit)
    if session.benchmark_kit_id != "tau2":
        return None
    tool_choice = payload.get("tool_choice") or loop_payload.get("tool_choice")
    if tool_choice in {"none", None}:
        return None if tool_choice == "none" else "respond"
    if isinstance(tool_choice, dict):
        function = tool_choice.get("function")
        if isinstance(function, dict) and function.get("name"):
            return str(function["name"])
    for tool in tools:
        function = tool.get("function") if isinstance(tool.get("function"), dict) else {}
        if function.get("name") == "respond":
            return "respond"
    return None


async def _invoke_runtime_backend(backend: Any, request: dict[str, Any]) -> Any:
    async_backend_call = getattr(backend, "ainvoke", None)
    if callable(async_backend_call):
        return await async_backend_call(request)
    backend_call = getattr(backend, "invoke", None)
    if callable(backend_call):
        result = backend_call(request)
    elif callable(backend):
        result = backend(request)
    else:
        raise RuntimeError("agent_backend_missing_invoke")
    if inspect.isawaitable(result):
        return await result
    return result


def _extract_runtime_tool_calls(
    adapter: OpenAIProviderAdapter,
    raw_response: Any,
    *,
    backend: Any,
    session: AgentRuntimeSession,
    payload: dict[str, Any],
    tools: list[dict[str, Any]],
    turn_index: int,
) -> list[ToolCallIR]:
    try:
        calls = adapter.extract_tool_calls(raw_response, turn_index=turn_index)
    except ToolingError:
        if not _should_parse_tau2_dialect(session=session, payload=payload):
            raise
        calls = []
    if calls:
        return _filter_tool_calls_by_schema(calls, tools)
    if not _should_parse_tau2_dialect(session=session, payload=payload):
        return []

    parser = Tau2ToolDialectParser()
    dialect = _effective_tau2_tool_dialect(payload=payload, backend=backend)
    for candidate in _tool_text_candidates(raw_response):
        try:
            parsed = parser.parse(candidate, dialect=dialect, turn_index=turn_index)
        except ToolingError:
            continue
        if parsed:
            return _filter_tool_calls_by_schema(parsed, tools)
    return []


def _effective_tau2_tool_dialect(*, payload: dict[str, Any], backend: Any) -> str:
    explicit = (
        payload.get("tool_dialect")
        or payload.get("tooling_dialect")
        or payload.get("tool_call_format")
        or payload.get("tool_format")
    )
    if explicit:
        return _normalize_dialect_name(explicit)
    return _infer_tau2_tool_dialect(backend)


def _infer_tau2_tool_dialect(backend: Any) -> str:
    backend_obj = getattr(backend, "static_backend", backend)
    config = getattr(backend_obj, "config", None)
    config_values: list[str] = []
    if isinstance(config, dict):
        for key in ("tool_dialect", "tool_call_format", "tool_format", "model", "model_name", "model_path"):
            value = config.get(key)
            if value:
                config_values.append(str(value))
    for attr in ("model_name", "model", "model_path", "provider", "api_base"):
        value = getattr(backend_obj, attr, None)
        if value:
            config_values.append(str(value))
    text = " ".join(config_values).lower()
    if "minimax" in text:
        return "minimax"
    if "functiongemma" in text or re.search(r"gemma[_-]?4", text):
        return "gemma"
    if "harmony" in text or "gpt-oss" in text:
        return "harmony_xml"
    if re.search(r"\bqwen(?:2|3)?(?:[\W_]|$)", text):
        return "qwen_xml"
    return "auto"


def _normalize_dialect_name(value: Any) -> str:
    normalized = str(value or "auto").strip().lower()
    return {
        "qwen": "qwen_xml",
        "qwen3": "qwen_xml",
        "qwen3.5": "qwen_xml",
        "qwen3.6": "qwen_xml",
        "gemma4": "gemma",
        "gemma-4": "gemma",
        "gemma_4": "gemma",
        "functiongemma": "gemma",
        "harmony": "harmony_xml",
    }.get(normalized, normalized or "auto")


def _normalize_string_set(value: Any) -> set[str]:
    if value in (None, ""):
        return set()
    if isinstance(value, str):
        items = [value]
    elif isinstance(value, (list, tuple, set)):
        items = list(value)
    else:
        return set()
    return {_normalize_dialect_name(item) for item in items if str(item or "").strip()}


def _filter_tool_calls_by_schema(calls: list[ToolCallIR], tools: list[dict[str, Any]]) -> list[ToolCallIR]:
    names = _declared_tool_names(tools)
    if not names:
        return calls
    return [call for call in calls if call.name in names]


def _declared_tool_names(tools: list[dict[str, Any]]) -> set[str]:
    names: set[str] = set()
    for tool in tools:
        if not isinstance(tool, dict):
            continue
        function = tool.get("function")
        if isinstance(function, dict) and function.get("name"):
            names.add(str(function["name"]))
        elif tool.get("name"):
            names.add(str(tool["name"]))
    return names


def _should_parse_tau2_dialect(*, session: AgentRuntimeSession, payload: dict[str, Any]) -> bool:
    return session.benchmark_kit_id == "tau2" or bool(
        payload.get("tool_dialect")
        or payload.get("tooling_dialect")
        or payload.get("tool_call_format")
        or payload.get("tool_format")
    )


def _tool_text_candidates(raw_response: Any) -> list[Any]:
    candidates: list[Any] = []
    if isinstance(raw_response, str):
        candidates.append(raw_response)
        return candidates
    if not isinstance(raw_response, dict):
        return candidates
    for key in ("content", "answer", "output_text"):
        value = raw_response.get(key)
        if isinstance(value, str) and value.strip():
            candidates.append(value)
    message = _extract_first_response_message(raw_response)
    if isinstance(message, dict):
        content = message.get("content")
        if isinstance(content, str) and content.strip():
            candidates.append(content)
    return candidates


def _assistant_message_from_raw_response(raw_response: Any, tool_calls: list[Any]) -> dict[str, Any]:
    if isinstance(raw_response, dict):
        message = _extract_first_response_message(raw_response)
        if isinstance(message, dict):
            return {
                "role": "assistant",
                "content": message.get("content") or "",
                "tool_calls": message.get("tool_calls")
                or [
                    {
                        "id": call.call_id,
                        "type": "function",
                        "function": {"name": call.name, "arguments": call.arguments_json},
                    }
                    for call in tool_calls
                ],
            }
    return {
        "role": "assistant",
        "content": "",
        "tool_calls": [
            {
                "id": call.call_id,
                "type": "function",
                "function": {"name": call.name, "arguments": call.arguments_json},
            }
            for call in tool_calls
        ],
    }


def _build_tool_execution_context(
    *,
    session: AgentRuntimeSession,
    payload: dict[str, Any],
) -> ToolExecutionContext:
    trial_id = _resolve_trial_id(session=session, payload=payload)
    metadata = session.scheduler_state.setdefault("tool_execution_metadata", {})
    if not isinstance(metadata, dict):
        metadata = {}
        session.scheduler_state["tool_execution_metadata"] = metadata
    metadata.setdefault("session_id", session.session_id)
    metadata.setdefault("benchmark_kit_id", session.benchmark_kit_id)
    return ToolExecutionContext(
        run_id=session.run_id,
        task_id=session.task_id,
        sample_id=session.sample_id,
        trial_id=trial_id,
        resource_lease=session.resource_lease,
        environment_lease=_resolve_environment_lease(session=session, payload=payload),
        artifact_sink=_resolve_artifact_sink(session=session, payload=payload),
        metadata=metadata,
    )


def _resolve_trial_id(*, session: AgentRuntimeSession, payload: dict[str, Any]) -> str:
    return str(
        payload.get("trial_id")
        or session.runtime_context.get("trial_id")
        or session.scheduler_state.get("trial_id")
        or "trial_0001"
    )


def _resolve_environment_lease(*, session: AgentRuntimeSession, payload: dict[str, Any]) -> Any | None:
    if payload.get("environment_lease") is not None:
        return payload.get("environment_lease")
    if session.runtime_context.get("environment_lease") is not None:
        return session.runtime_context.get("environment_lease")
    if hasattr(session.resource_lease, "environment"):
        return session.resource_lease
    return None


def _resolve_artifact_sink(*, session: AgentRuntimeSession, payload: dict[str, Any]) -> Any | None:
    return payload.get("artifact_sink") or session.runtime_context.get("artifact_sink")


def _write_tooling_artifact(
    *,
    sink: Any | None,
    session: AgentRuntimeSession,
    trial_id: str,
    name: str,
    content: Any,
) -> Any | None:
    if sink is None:
        return None
    writer = getattr(sink, "write_artifact", None)
    if not callable(writer):
        return None
    return writer(
        run_id=session.run_id,
        task_id=session.task_id,
        sample_id=session.sample_id,
        trial_id=trial_id,
        owner="agent",
        name=name,
        content=to_json_compatible(content),
        mime_type="application/json",
    )


def _append_tooling_trace(
    *,
    sink: Any | None,
    session: AgentRuntimeSession,
    trial_id: str,
    actor: str,
    event_type: str,
    payload: dict[str, Any],
    artifact_refs: list[Any] | None = None,
) -> None:
    if sink is None:
        return
    appender = getattr(sink, "append_trace_event", None)
    if not callable(appender):
        return
    appender(
        run_id=session.run_id,
        task_id=session.task_id,
        sample_id=session.sample_id,
        trial_id=trial_id,
        actor=actor,
        event_type=event_type,
        payload=to_json_compatible(payload),
        artifact_refs=[ref for ref in (artifact_refs or []) if ref is not None],
    )


def _ref_payload(ref: Any | None) -> dict[str, Any] | None:
    if ref is None:
        return None
    if hasattr(ref, "model_dump"):
        return ref.model_dump(mode="python")
    if isinstance(ref, dict):
        return dict(ref)
    return {"ref": str(ref)}


def _tool_output_artifact_name(*, call_id: str, stream: str, turn_index: int) -> str:
    call_fragment = "".join(char if char.isalnum() or char in {"-", "_", "."} else "_" for char in call_id)
    call_fragment = call_fragment.strip("._-") or "call"
    return f"tool_{call_fragment}_{stream}_turn{turn_index}.txt"


def _tool_call_payload(call: Any) -> dict[str, Any]:
    return {
        "provider": call.provider,
        "call_id": call.call_id,
        "name": call.name,
        "arguments_json": call.arguments_json,
        "metadata": dict(call.metadata or {}),
    }


def _tool_result_payload(result: Any) -> dict[str, Any]:
    return {
        "provider": result.provider,
        "call_id": result.call_id,
        "name": result.name,
        "status": result.status,
        "output_text": result.output_text,
        "output_json": result.output_json,
        "artifact_refs": list(result.artifact_refs or []),
    }


def _backend_id(backend: Any) -> str:
    return str(getattr(backend, "backend_id", None) or getattr(backend, "client_id", None) or backend.__class__.__name__)


def _finish_reason(raw_response: Any) -> str | None:
    if not isinstance(raw_response, dict):
        return None
    choices = raw_response.get("choices")
    if isinstance(choices, list) and choices and isinstance(choices[0], dict):
        reason = choices[0].get("finish_reason")
        return str(reason) if reason is not None else None
    return None


def _extract_usage(raw_response: Any) -> dict[str, Any] | None:
    if isinstance(raw_response, dict) and isinstance(raw_response.get("usage"), dict):
        return dict(raw_response["usage"])
    return None


def _build_trace_step(
    trace_step: int,
    *,
    trace_role: str,
    name: str | None,
    input_payload: Any,
    output_payload: Any,
    status: str | None,
    latency_ms: float | None,
    usage: dict[str, Any] | None = None,
    turn_index: int | None = None,
) -> dict[str, Any]:
    step: dict[str, Any] = {
        "trace_step": trace_step,
        "trace_role": trace_role,
        "name": name or "",
        "input": input_payload,
        "output": output_payload,
        "status": status or "success",
        "latency_ms": float(latency_ms or 0.0),
        "timestamp": int(time.time()),
    }
    if turn_index is not None:
        step["turn_index"] = turn_index
    if usage:
        step["input_tokens"] = usage.get("input_tokens") or usage.get("prompt_tokens")
        step["output_tokens"] = usage.get("output_tokens") or usage.get("completion_tokens")
        step["cost_usd"] = usage.get("cost_usd")
    return step


def _tool_schema_invalid_failure(
    *,
    session: AgentRuntimeSession,
    summary: str,
    details: dict[str, Any],
) -> FailureEnvelope:
    return FailureEnvelope(
        failure_domain="client_execution",
        failure_stage="run_scheduler",
        failure_code="client_execution.tool_schema_invalid",
        component_kind="scheduler",
        component_id="framework_loop.tool_registry",
        owner="runtime_scheduler_core",
        retryable=False,
        summary=summary,
        first_bad_step="framework_loop.project_tool_registry",
        suspect_files=(
            "src/gage_eval/agent_runtime/tooling/registry.py",
            "src/gage_eval/agent_runtime/schedulers/framework_loop.py",
        ),
        details={
            "session_id": session.session_id,
            "benchmark_kit_id": session.benchmark_kit_id,
            **details,
        },
    )


def _tooling_failure(
    *,
    session: AgentRuntimeSession,
    workflow_bundle: Any,
    error: ToolingError,
) -> FailureEnvelope:
    return FailureEnvelope(
        failure_domain="client_execution",
        failure_stage="run_scheduler",
        failure_code=error.code,
        component_kind="provider",
        component_id=f"{workflow_bundle.bundle_id}.tooling",
        owner="runtime_scheduler_core",
        retryable=False,
        summary=str(error),
        first_bad_step="framework_loop.runtime_tooling",
        suspect_files=(
            "src/gage_eval/agent_runtime/tooling/provider_adapters.py",
            "src/gage_eval/agent_runtime/schedulers/framework_loop.py",
        ),
        details={
            "session_id": session.session_id,
            "benchmark_kit_id": session.benchmark_kit_id,
            "tooling_code": error.code,
            **dict(error.details or {}),
        },
    )


def _model_request_timeout_failure(
    *,
    session: AgentRuntimeSession,
    workflow_bundle: Any,
    error: BaseException,
) -> FailureEnvelope:
    return FailureEnvelope(
        failure_domain="client_execution",
        failure_stage="run_scheduler",
        failure_code="client_execution.model_request_timeout",
        component_kind="provider",
        component_id=f"{workflow_bundle.bundle_id}.model_request",
        owner="runtime_scheduler_core",
        retryable=True,
        summary=str(error) or "model request timed out",
        first_bad_step="framework_loop.model_request",
        suspect_files=(
            "src/gage_eval/agent_runtime/clients/runner.py",
            "src/gage_eval/agent_runtime/schedulers/framework_loop.py",
        ),
        details={
            "session_id": session.session_id,
            "benchmark_kit_id": session.benchmark_kit_id,
            "exception_type": error.__class__.__name__,
        },
    )


def _is_timeout_exception(error: BaseException) -> bool:
    if isinstance(error, TimeoutError):
        return True
    name = error.__class__.__name__
    if name in {"Timeout", "APITimeoutError", "TimeoutException"}:
        return True
    message = str(error).lower()
    return "timed out" in message or "request timeout" in message


def _resolve_workflow_path(session: AgentRuntimeSession) -> str:
    """Resolve the benchmark-owned workflow implementation path."""

    return (
        f"src/gage_eval/agent_eval_kits/{session.benchmark_kit_id}"
        f"/sub_workflows/{session.scheduler_type}.py"
    )


def _extract_system_prompt(messages: list[dict[str, Any]]) -> str | None:
    """Extracts the rendered system prompt for observability/debugging."""

    for message in messages:
        if not isinstance(message, dict) or message.get("role") != "system":
            continue
        content = message.get("content")
        if isinstance(content, str) and content.strip():
            return content
    return None


def _resolve_sampling_params(
    *,
    loop_payload: dict[str, Any],
    payload: dict[str, Any],
    sample: dict[str, Any],
) -> dict[str, Any] | None:
    for source in (loop_payload, payload, sample):
        value = source.get("sampling_params") if isinstance(source, dict) else None
        if isinstance(value, dict):
            return dict(value)
    return None


def _extract_response_answer(response: dict[str, Any]) -> str:
    content = response.get("content") or response.get("output_text")
    if isinstance(content, str):
        return content
    message = _extract_first_response_message(response)
    if isinstance(message, dict):
        message_content = message.get("content")
        if isinstance(message_content, str):
            return message_content
        if isinstance(message_content, list):
            parts = [
                str(part.get("text"))
                for part in message_content
                if isinstance(part, dict) and isinstance(part.get("text"), str)
            ]
            if parts:
                return "".join(parts)
    return ""


def _extract_response_tool_calls(response: dict[str, Any]) -> list[dict[str, Any]]:
    direct = response.get("tool_calls")
    if isinstance(direct, list):
        return [call for call in direct if isinstance(call, dict)]
    message = _extract_first_response_message(response)
    if isinstance(message, dict) and isinstance(message.get("tool_calls"), list):
        return [call for call in message["tool_calls"] if isinstance(call, dict)]
    return []


def _extract_first_response_message(response: dict[str, Any]) -> dict[str, Any] | None:
    choices = response.get("choices")
    if isinstance(choices, list) and choices:
        choice = choices[0]
        if isinstance(choice, dict):
            message = choice.get("message") or choice.get("delta")
            if isinstance(message, dict):
                return message
    return None
