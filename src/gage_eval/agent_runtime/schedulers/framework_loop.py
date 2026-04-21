from __future__ import annotations

from typing import Any

from gage_eval.assets.prompts.renderers import PromptContext
from gage_eval.agent_runtime.contracts.failure import FailureEnvelopeError
from gage_eval.agent_runtime.contracts.scheduler import SchedulerResult
from gage_eval.agent_runtime.failure_mapper import FailureMapper
from gage_eval.agent_runtime.session import AgentRuntimeSession
from gage_eval.role.agent.loop import AgentLoop


class FrameworkLoopScheduler:
    """Runs the internal framework loop as a self-contained scheduler."""

    def __init__(
        self,
        *,
        backend,
        tool_router,
        prompt_renderer=None,
        max_turns: int = 8,
        tool_call_retry_budget: int = 3,
        pre_hooks=None,
        post_hooks=None,
    ) -> None:
        self._backend = backend
        self._tool_router = tool_router
        self._prompt_renderer = prompt_renderer
        self._max_turns = max_turns
        self._tool_call_retry_budget = tool_call_retry_budget
        self._pre_hooks = pre_hooks
        self._post_hooks = post_hooks
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
                        failure_code=(
                            f"input_projection.prepare_inputs."
                            f"{workflow_bundle.bundle_id}.build_loop_inputs_failed"
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
                        failure_code=(
                            f"input_projection.prepare_inputs."
                            f"{workflow_bundle.bundle_id}.inject_prompt_context_failed"
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
        tools = list(sample.get("tools") or [])
        if callable(workflow_bundle.inject_tool_schemas):
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
                        failure_code=(
                            f"input_projection.prepare_inputs."
                            f"{workflow_bundle.bundle_id}.inject_tool_schemas_failed"
                        ),
                        first_bad_step=f"{workflow_bundle.bundle_id}.inject_tool_schemas",
                        suspect_files=(
                            workflow_path,
                            "src/gage_eval/agent_runtime/schedulers/framework_loop.py",
                        ),
                    )
                ) from exc
            if isinstance(injected_tools, list):
                tools = injected_tools

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

        # STEP 2: Execute the shared agent loop.
        loop = AgentLoop(
            backend=self._backend,
            tool_router=self._tool_router,
            max_turns=self._max_turns,
            tool_call_retry_budget=self._tool_call_retry_budget,
            pre_hooks=self._pre_hooks,
            post_hooks=self._post_hooks,
        )
        try:
            raw_result = await loop.arun(
                messages=list(loop_payload.get("messages") or []),
                tools=tools,
                tool_choice=loop_payload.get("tool_choice"),
                sandbox_config=payload.get("sandbox_config"),
                sandbox_provider=sandbox_provider,
                metadata=payload.get("metadata") or sample.get("metadata") or {},
                sample=sample,
            )
        except Exception as exc:
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
                        "src/gage_eval/role/agent/loop.py",
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
