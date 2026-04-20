"""Agent loop state machine for tool-calling workflows."""

from __future__ import annotations

import asyncio
import json
import time
from typing import Any, Dict, Iterable, List, Optional, Sequence

from loguru import logger

from gage_eval.registry.utils import ensure_async, run_sync
from gage_eval.role.agent.backends.base import AgentBackend, normalize_agent_output
from gage_eval.role.agent.hooks import AgentHookContext, AgentLoopHook, build_hook_chain
from gage_eval.role.agent.tool_router import ToolRouter
from gage_eval.sandbox.manager import SandboxHandle
from gage_eval.sandbox.provider import SandboxProvider


class AgentLoop:
    """Runs a tool-calling agent until it reaches a stop condition."""

    def __init__(
        self,
        backend: AgentBackend,
        tool_router: ToolRouter,
        *,
        max_turns: int = 8,
        pre_hooks: Optional[Sequence[Any]] = None,
        post_hooks: Optional[Sequence[Any]] = None,
    ) -> None:
        self._backend = backend
        self._tool_router = tool_router
        self._max_turns = max(1, int(max_turns))
        self._pre_hooks = build_hook_chain(pre_hooks)
        self._post_hooks = build_hook_chain(post_hooks)

    def run(
        self,
        *,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
        tool_choice: Optional[Any] = None,
        sandbox_config: Optional[Dict[str, Any]] = None,
        sandbox_provider: Optional[SandboxProvider] = None,
        metadata: Optional[Dict[str, Any]] = None,
        sample: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Execute the agent loop until a stop condition is reached.

        Args:
            messages: Conversation messages to seed the agent.
            tools: Tool schema list for tool-calling models.
            tool_choice: Optional tool selection strategy.
            sandbox_config: Sandbox configuration for tool execution.
            sandbox_provider: Sample-scoped SandboxProvider instance.
            metadata: Optional metadata payload passed to the backend.

        Returns:
            Normalized agent output containing answer, agent_trace, usage, and artifacts.
        """

        agent_trace: List[Dict[str, Any]] = []
        sandbox_handle: Optional[SandboxHandle] = None
        tool_registry = _build_tool_registry(tools or [])
        hook_context: Optional[AgentHookContext] = None
        hooks_ready = False
        main_error: Optional[BaseException] = None
        effective_metadata = metadata or (sample.get("metadata") if isinstance(sample, dict) else {}) or {}
        try:
            # STEP 1: Setup sandbox and hook context
            if sandbox_config is not None and sandbox_provider is None:
                raise RuntimeError("sandbox_provider_missing")
            if sandbox_provider is not None:
                sandbox_handle = sandbox_provider.get_handle()
            runtime_handle = sandbox_handle.runtime_handle if sandbox_handle else {}
            hook_context = AgentHookContext(
                sample=sample or {},
                metadata=effective_metadata,
                runtime_handle=runtime_handle,
                sandbox_config=sandbox_config or {},
            )
            # STEP 2: Run pre-hooks before model thinking
            _run_hook_chain(self._pre_hooks, hook_context)
            hooks_ready = True
            # STEP 3: Execute agent loop
            step_index = 1
            answer = ""
            usage: Optional[Dict[str, Any]] = None
            artifacts = []
            required_tool_retry_active = False
            logger.debug(
                "AgentLoop start max_turns={} messages={} tools={}",
                self._max_turns,
                len(messages),
                len(tools or []),
            )
            for turn in range(1, self._max_turns + 1):
                if sandbox_handle and hasattr(sandbox_handle.sandbox, "is_alive"):
                    if not sandbox_handle.sandbox.is_alive():
                        raise RuntimeError("sandbox_crashed")
                logger.debug(
                    "AgentLoop turn {} begin messages={} tools={}",
                    turn,
                    len(messages),
                    len(tools or []),
                )
                effective_tool_choice = _resolve_effective_tool_choice(
                    backend=self._backend,
                    tool_choice="required" if required_tool_retry_active else tool_choice,
                    tools=tools or [],
                    turn_index=turn,
                )
                backend_payload = {
                    "messages": messages,
                    "tools": tools or [],
                    "tool_choice": effective_tool_choice,
                    "turn_index": turn,
                    "runtime_handle": runtime_handle,
                    "metadata": effective_metadata,
                }
                start = time.perf_counter()
                raw_output = _invoke_backend_sync(self._backend, backend_payload)
                elapsed_ms = (time.perf_counter() - start) * 1000.0
                output = normalize_agent_output(raw_output)
                usage = output.get("usage") or usage
                artifacts = output.get("artifacts") or artifacts
                tool_calls = _extract_tool_calls(output)
                if tool_calls:
                    required_tool_retry_active = False
                    logger.info(
                        "AgentLoop turn {} tool_calls={} names={}",
                        turn,
                        len(tool_calls),
                        [_tool_call_name(call) for call in tool_calls],
                    )
                    messages.append(_build_tool_call_message(output, tool_calls))
                    stop_after_tool = False
                    for tool_call in tool_calls:
                        tool_result = self._tool_router.execute(
                            tool_call,
                            sandbox_handle.sandbox if sandbox_handle else None,
                            tool_registry=tool_registry,
                        )
                        tool_name = tool_result.get("name") or _tool_call_name(tool_call)
                        trace_step = _build_trace_step(
                            step_index,
                            trace_role="tool",
                            name=tool_name,
                            input_payload=tool_result.get("input"),
                            output_payload=tool_result.get("output"),
                            status=tool_result.get("status"),
                            latency_ms=tool_result.get("latency_ms"),
                            turn_index=turn,
                        )
                        resolved_tool = tool_result.get("resolved_tool")
                        if resolved_tool:
                            trace_step["resolved_tool"] = resolved_tool
                        agent_trace.append(trace_step)
                        resolved_tool = tool_result.get("resolved_tool")
                        if resolved_tool:
                            logger.info(
                                "AgentLoop tool {} resolved={} status={}",
                                tool_result.get("name"),
                                resolved_tool,
                                tool_result.get("status"),
                            )
                        else:
                            logger.debug(
                                "AgentLoop tool {} status={}",
                                tool_result.get("name"),
                                tool_result.get("status"),
                            )
                        step_index += 1
                        messages.append(_build_tool_message(tool_call, tool_result.get("output")))
                        final_answer = _resolve_tool_final_answer(tool_result)
                        if final_answer:
                            answer = final_answer
                            trace_output = {"answer": answer, "final_from_tool": tool_name}
                            agent_trace.append(
                                _build_trace_step(
                                    step_index,
                                    trace_role="assistant",
                                    name="agent_response",
                                    input_payload=None,
                                    output_payload=trace_output,
                                    status="success",
                                    latency_ms=0.0,
                                    usage=usage,
                                    turn_index=turn,
                                )
                            )
                            step_index += 1
                            stop_after_tool = True
                            break
                    if stop_after_tool:
                        break
                    continue
                answer = output.get("answer") or ""
                if _requires_tool_call(effective_tool_choice, tools or []):
                    logger.warning(
                        "AgentLoop turn {} required tool call but backend returned plain answer",
                        turn,
                    )
                    trace_output = _build_agent_output_payload(output, answer)
                    trace_output["error"] = "required_tool_call_missing"
                    agent_trace.append(
                        _build_trace_step(
                            step_index,
                            trace_role="assistant",
                            name="agent_response",
                            input_payload=None,
                            output_payload=trace_output,
                            status="retry_required_tool_call",
                            latency_ms=elapsed_ms,
                            usage=usage,
                            turn_index=turn,
                        )
                    )
                    step_index += 1
                    messages.append({"role": "assistant", "content": answer})
                    messages.append(_build_required_tool_retry_message())
                    required_tool_retry_active = True
                    answer = ""
                    continue
                logger.info("AgentLoop turn {} completed answer_len={}", turn, len(answer))
                trace_output = _build_agent_output_payload(output, answer)
                agent_trace.append(
                    _build_trace_step(
                        step_index,
                        trace_role="assistant",
                        name="agent_response",
                        input_payload=None,
                        output_payload=trace_output,
                        status=_resolve_trace_status(output),
                        latency_ms=elapsed_ms,
                        usage=usage,
                        turn_index=turn,
                    )
                )
                break
            # STEP 4: Return loop output
            return {
                "answer": answer,
                "agent_trace": agent_trace,
                "usage": usage,
                "artifacts": artifacts,
            }
        except BaseException as exc:
            main_error = exc
            raise
        finally:
            # STEP 5: Run post-hooks before finalizing the loop
            if hooks_ready and hook_context is not None:
                post_context = hook_context.with_agent_trace(agent_trace)
                try:
                    _run_hook_chain(self._post_hooks, post_context)
                except BaseException:
                    if main_error is None:
                        raise

    async def arun(
        self,
        *,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
        tool_choice: Optional[Any] = None,
        sandbox_config: Optional[Dict[str, Any]] = None,
        sandbox_provider: Optional[SandboxProvider] = None,
        metadata: Optional[Dict[str, Any]] = None,
        sample: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Execute the agent loop asynchronously until a stop condition is reached.

        Args:
            messages: Conversation messages to seed the agent.
            tools: Tool schema list for tool-calling models.
            tool_choice: Optional tool selection strategy.
            sandbox_config: Sandbox configuration for tool execution.
            sandbox_provider: Sample-scoped SandboxProvider instance.
            metadata: Optional metadata payload passed to the backend.

        Returns:
            Normalized agent output containing answer, agent_trace, usage, and artifacts.
        """

        agent_trace: List[Dict[str, Any]] = []
        sandbox_handle: Optional[SandboxHandle] = None
        tool_registry = _build_tool_registry(tools or [])
        hook_context: Optional[AgentHookContext] = None
        hooks_ready = False
        main_error: Optional[BaseException] = None
        effective_metadata = metadata or (sample.get("metadata") if isinstance(sample, dict) else {}) or {}
        try:
            # STEP 1: Setup sandbox and hook context
            if sandbox_config is not None and sandbox_provider is None:
                raise RuntimeError("sandbox_provider_missing")
            if sandbox_provider is not None:
                sandbox_handle = sandbox_provider.get_handle()
            runtime_handle = sandbox_handle.runtime_handle if sandbox_handle else {}
            hook_context = AgentHookContext(
                sample=sample or {},
                metadata=effective_metadata,
                runtime_handle=runtime_handle,
                sandbox_config=sandbox_config or {},
            )
            # STEP 2: Run pre-hooks before model thinking
            _run_hook_chain(self._pre_hooks, hook_context)
            hooks_ready = True
            # STEP 3: Execute agent loop
            step_index = 1
            answer = ""
            usage: Optional[Dict[str, Any]] = None
            artifacts = []
            required_tool_retry_active = False
            logger.debug(
                "AgentLoop start max_turns={} messages={} tools={}",
                self._max_turns,
                len(messages),
                len(tools or []),
            )
            for turn in range(1, self._max_turns + 1):
                if sandbox_handle and hasattr(sandbox_handle.sandbox, "is_alive"):
                    if not sandbox_handle.sandbox.is_alive():
                        raise RuntimeError("sandbox_crashed")
                logger.debug(
                    "AgentLoop turn {} begin messages={} tools={}",
                    turn,
                    len(messages),
                    len(tools or []),
                )
                effective_tool_choice = _resolve_effective_tool_choice(
                    backend=self._backend,
                    tool_choice="required" if required_tool_retry_active else tool_choice,
                    tools=tools or [],
                    turn_index=turn,
                )
                backend_payload = {
                    "messages": messages,
                    "tools": tools or [],
                    "tool_choice": effective_tool_choice,
                    "turn_index": turn,
                    "runtime_handle": runtime_handle,
                    "metadata": effective_metadata,
                }
                start = time.perf_counter()
                raw_output = await _invoke_backend_async(self._backend, backend_payload)
                elapsed_ms = (time.perf_counter() - start) * 1000.0
                output = normalize_agent_output(raw_output)
                usage = output.get("usage") or usage
                artifacts = output.get("artifacts") or artifacts
                tool_calls = _extract_tool_calls(output)
                if tool_calls:
                    required_tool_retry_active = False
                    logger.info(
                        "AgentLoop turn {} tool_calls={} names={}",
                        turn,
                        len(tool_calls),
                        [_tool_call_name(call) for call in tool_calls],
                    )
                    messages.append(_build_tool_call_message(output, tool_calls))
                    stop_after_tool = False
                    for tool_call in tool_calls:
                        tool_result = self._tool_router.execute(
                            tool_call,
                            sandbox_handle.sandbox if sandbox_handle else None,
                            tool_registry=tool_registry,
                        )
                        tool_name = tool_result.get("name") or _tool_call_name(tool_call)
                        trace_step = _build_trace_step(
                            step_index,
                            trace_role="tool",
                            name=tool_name,
                            input_payload=tool_result.get("input"),
                            output_payload=tool_result.get("output"),
                            status=tool_result.get("status"),
                            latency_ms=tool_result.get("latency_ms"),
                            turn_index=turn,
                        )
                        resolved_tool = tool_result.get("resolved_tool")
                        if resolved_tool:
                            trace_step["resolved_tool"] = resolved_tool
                        agent_trace.append(trace_step)
                        resolved_tool = tool_result.get("resolved_tool")
                        if resolved_tool:
                            logger.info(
                                "AgentLoop tool {} resolved={} status={}",
                                tool_result.get("name"),
                                resolved_tool,
                                tool_result.get("status"),
                            )
                        else:
                            logger.debug(
                                "AgentLoop tool {} status={}",
                                tool_result.get("name"),
                                tool_result.get("status"),
                            )
                        step_index += 1
                        messages.append(_build_tool_message(tool_call, tool_result.get("output")))
                        final_answer = _resolve_tool_final_answer(tool_result)
                        if final_answer:
                            answer = final_answer
                            trace_output = {"answer": answer, "final_from_tool": tool_name}
                            agent_trace.append(
                                _build_trace_step(
                                    step_index,
                                    trace_role="assistant",
                                    name="agent_response",
                                    input_payload=None,
                                    output_payload=trace_output,
                                    status="success",
                                    latency_ms=0.0,
                                    usage=usage,
                                    turn_index=turn,
                                )
                            )
                            step_index += 1
                            stop_after_tool = True
                            break
                    if stop_after_tool:
                        break
                    continue
                answer = output.get("answer") or ""
                if _requires_tool_call(effective_tool_choice, tools or []):
                    logger.warning(
                        "AgentLoop turn {} required tool call but backend returned plain answer",
                        turn,
                    )
                    trace_output = _build_agent_output_payload(output, answer)
                    trace_output["error"] = "required_tool_call_missing"
                    agent_trace.append(
                        _build_trace_step(
                            step_index,
                            trace_role="assistant",
                            name="agent_response",
                            input_payload=None,
                            output_payload=trace_output,
                            status="retry_required_tool_call",
                            latency_ms=elapsed_ms,
                            usage=usage,
                            turn_index=turn,
                        )
                    )
                    step_index += 1
                    messages.append({"role": "assistant", "content": answer})
                    messages.append(_build_required_tool_retry_message())
                    required_tool_retry_active = True
                    answer = ""
                    continue
                logger.info("AgentLoop turn {} completed answer_len={}", turn, len(answer))
                trace_output = _build_agent_output_payload(output, answer)
                agent_trace.append(
                    _build_trace_step(
                        step_index,
                        trace_role="assistant",
                        name="agent_response",
                        input_payload=None,
                        output_payload=trace_output,
                        status=_resolve_trace_status(output),
                        latency_ms=elapsed_ms,
                        usage=usage,
                        turn_index=turn,
                    )
                )
                break
            # STEP 4: Return loop output
            return {
                "answer": answer,
                "agent_trace": agent_trace,
                "usage": usage,
                "artifacts": artifacts,
            }
        except BaseException as exc:
            main_error = exc
            raise
        finally:
            # STEP 5: Run post-hooks before finalizing the loop
            if hooks_ready and hook_context is not None:
                post_context = hook_context.with_agent_trace(agent_trace)
                try:
                    _run_hook_chain(self._post_hooks, post_context)
                except BaseException:
                    if main_error is None:
                        raise


def _invoke_backend_sync(backend: Any, payload: Dict[str, Any]) -> Any:
    backend_call = getattr(backend, "invoke", None)
    if callable(backend_call):
        return backend_call(payload)
    if callable(backend):
        return backend(payload)
    async_backend_call = getattr(backend, "ainvoke", None)
    if callable(async_backend_call):
        _raise_if_active_event_loop()
        return run_sync(async_backend_call(payload))
    raise RuntimeError("agent_backend_missing_invoke")


async def _invoke_backend_async(backend: Any, payload: Dict[str, Any]) -> Any:
    async_backend_call = getattr(backend, "ainvoke", None)
    if callable(async_backend_call):
        return await async_backend_call(payload)
    backend_call = getattr(backend, "invoke", None)
    if callable(backend_call):
        return await ensure_async(backend_call)(payload)
    if callable(backend):
        return await ensure_async(backend)(payload)
    raise RuntimeError("agent_backend_missing_invoke")


def _raise_if_active_event_loop() -> None:
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return
    raise RuntimeError("run_sync() cannot be used inside an active event loop")


def _extract_tool_calls(output: Dict[str, Any]) -> List[Dict[str, Any]]:
    tool_calls = output.get("tool_calls")
    if isinstance(tool_calls, list):
        return tool_calls
    return []


def _resolve_effective_tool_choice(
    *,
    backend: Any,
    tool_choice: Optional[Any],
    tools: List[Dict[str, Any]],
    turn_index: int,
) -> Optional[Any]:
    """Resolve the tool choice enforced by the agent backend wrapper."""

    if not tools:
        return tool_choice
    force_mode = str(getattr(backend, "_force_tool_choice_mode", "never") or "never")
    if force_mode == "always" and tool_choice in (None, "auto"):
        return "required"
    if force_mode == "first_turn" and turn_index <= 1 and tool_choice in (None, "auto"):
        return "required"
    return tool_choice


def _requires_tool_call(tool_choice: Optional[Any], tools: List[Dict[str, Any]]) -> bool:
    if not tools:
        return False
    if isinstance(tool_choice, str):
        return tool_choice.strip().lower() == "required"
    if isinstance(tool_choice, dict):
        return bool(tool_choice)
    return False


def _build_required_tool_retry_message() -> Dict[str, Any]:
    return {
        "role": "user",
        "content": (
            "Your previous response did not call a tool. You must call exactly one "
            "available tool now. If you need to speak to the user, call the respond "
            "tool instead of returning plain text."
        ),
    }


def _build_tool_message(tool_call: Dict[str, Any], output: Any) -> Dict[str, Any]:
    tool_call_id = tool_call.get("id")
    tool_name = tool_call.get("name")
    if "function" in tool_call:
        tool_name = tool_call["function"].get("name", tool_name)
    return {
        "role": "tool",
        "tool_call_id": tool_call_id,
        "name": tool_name,
        "content": json.dumps(output, ensure_ascii=True),
    }


def _build_tool_call_message(output: Dict[str, Any], tool_calls: List[Dict[str, Any]]) -> Dict[str, Any]:
    return {
        "role": "assistant",
        "content": output.get("answer") or "",
        "tool_calls": tool_calls,
    }


def _tool_call_name(tool_call: Dict[str, Any]) -> str:
    if "function" in tool_call:
        fn = tool_call.get("function") or {}
        return str(fn.get("name") or "")
    return str(tool_call.get("name") or "")


def _resolve_tool_final_answer(tool_result: Dict[str, Any]) -> str:
    if tool_result.get("status") != "success":
        return ""
    final_answer = tool_result.get("final_answer")
    if isinstance(final_answer, str) and final_answer.strip():
        return final_answer
    return ""


def _build_tool_registry(tools: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    registry: Dict[str, Dict[str, Any]] = {}
    for tool in tools:
        if not isinstance(tool, dict):
            continue
        if tool.get("type") == "function" and isinstance(tool.get("function"), dict):
            name = tool["function"].get("name")
        else:
            name = tool.get("name")
        if not name:
            continue
        registry[str(name)] = tool
    return registry


def _run_hook_chain(hooks: Sequence[AgentLoopHook], context: AgentHookContext) -> None:
    for hook in hooks:
        if hook is None:
            continue
        handler = getattr(hook, "run", None)
        if callable(handler):
            handler(context)
            continue
        if callable(hook):
            hook(context)


def _build_trace_step(
    trace_step: int,
    *,
    trace_role: str,
    name: Optional[str],
    input_payload: Any,
    output_payload: Any,
    status: Optional[str],
    latency_ms: Optional[float],
    usage: Optional[Dict[str, Any]] = None,
    turn_index: Optional[int] = None,
) -> Dict[str, Any]:
    step: Dict[str, Any] = {
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
        step["input_tokens"] = usage.get("input_tokens")
        step["output_tokens"] = usage.get("output_tokens")
        step["cost_usd"] = usage.get("cost_usd")
    return step


def _build_agent_output_payload(output: Dict[str, Any], answer: str) -> Dict[str, Any]:
    payload: Dict[str, Any] = {"answer": answer}
    if "raw_response" in output:
        payload["raw_response"] = output.get("raw_response")
    if "error" in output:
        payload["error"] = output.get("error")
    if "status" in output:
        payload["status"] = output.get("status")
    return payload


def _resolve_trace_status(output: Dict[str, Any]) -> str:
    if output.get("error"):
        return "error"
    return "success"
