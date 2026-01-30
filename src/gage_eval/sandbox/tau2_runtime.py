"""Tau2 runtime for local in-process evaluation."""

from __future__ import annotations

import copy
import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional

from gage_eval.sandbox.base import BaseSandbox, SandboxOptionalMixin
from gage_eval.utils.benchmark_helpers.tau2 import (
    ensure_tau2_importable,
    resolve_tau2_data_dir,
)


class Tau2Runtime(SandboxOptionalMixin, BaseSandbox):
    """In-process Tau2 runtime wrapper for tool execution."""

    def __init__(self, runtime_configs: Optional[Dict[str, Any]] = None, resources: Optional[Dict[str, Any]] = None) -> None:
        self._runtime_configs = dict(runtime_configs or {})
        self._resources = dict(resources or {})
        self._running = False
        self._data_dir: Optional[str] = None
        self._env = None
        self._user = None
        self._user_state = None
        self._trajectory: List[Any] = []
        self._termination_reason = None
        self._step_count = 0
        self._error_count = 0
        self._max_steps = 200
        self._max_errors = 10
        self._respond_tool_name = "respond"
        self._start_time: Optional[str] = None
        self._task_id: Optional[str] = None
        self._trial: Optional[int] = None
        self._seed: Optional[int] = None
        self._domain: Optional[str] = None
        self._agent_cost_total: Optional[float] = None
        self._user_cost_total: float = 0.0

    def start(self, config: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        # STEP 1: Merge runtime configs and validate tau2 availability.
        self._runtime_configs.update(config.get("runtime_configs", {}) or {})
        ensure_tau2_importable()

        # STEP 2: Resolve and validate tau2 data directory.
        data_dir = resolve_tau2_data_dir(
            self._runtime_configs.get("data_dir") or config.get("data_dir")
        )
        if not data_dir.exists():
            raise FileNotFoundError(
                f"Tau2 data directory not found: {data_dir}. "
                "Set TAU2_DATA_DIR or run 'tau2 check-data' to validate."
            )
        self._data_dir = str(data_dir)

        # STEP 3: Load runtime limits.
        self._max_steps = int(
            config.get("max_steps")
            or self._runtime_configs.get("max_steps")
            or self._max_steps
        )
        self._max_errors = int(
            config.get("max_errors")
            or self._runtime_configs.get("max_errors")
            or self._max_errors
        )
        self._respond_tool_name = str(
            config.get("respond_tool_name")
            or self._runtime_configs.get("respond_tool_name")
            or self._respond_tool_name
        )
        self._running = True
        return {"profile": "tau2_local", "data_dir": self._data_dir}

    def exec(self, command: str, timeout: int = 30):  # pragma: no cover - not used
        raise NotImplementedError("Tau2Runtime does not support raw exec commands")

    def teardown(self) -> None:
        self._running = False
        self._env = None
        self._user = None
        self._user_state = None
        self._trajectory = []

    def is_alive(self, timeout_s: float | None = None) -> bool:
        return self._running

    def initialize_task(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        """Initialize tau2 task state and return initial messages/tools."""

        # STEP 1: Build the tau2 task + environment + user simulator.
        task = _build_tau2_task(sample)
        domain = _resolve_tau2_domain(sample, task)
        env = _build_tau2_environment(domain)
        self._env = env
        self._domain = domain
        self._task_id = str(task.id)
        self._trial = _coerce_int(_read_tau2_meta(sample, "trial"))
        self._seed = _coerce_int(_read_tau2_meta(sample, "seed"))
        self._start_time = _now()

        user_tools = _safe_get_user_tools(env)
        user_sim = _build_tau2_user_simulator(
            tools=user_tools,
            instructions=str(task.user_scenario),
            model=self._runtime_configs.get("user_model") or self._runtime_configs.get("user_llm"),
            model_args=self._runtime_configs.get("user_model_args") or self._runtime_configs.get("user_llm_args"),
            seed=self._seed,
        )
        self._user = user_sim

        initialization_data = getattr(task.initial_state, "initialization_data", None) if task.initial_state else None
        initialization_actions = getattr(task.initial_state, "initialization_actions", None) if task.initial_state else None
        message_history = list(getattr(task.initial_state, "message_history", None) or []) if task.initial_state else []
        try:
            env.set_state(
                initialization_data=initialization_data,
                initialization_actions=initialization_actions,
                message_history=message_history,
            )
        except Exception as exc:
            raise RuntimeError(f"tau2_env_set_state_failed: {exc}") from exc

        # STEP 2: Initialize user state and seed initial messages.
        user_state = user_sim.get_init_state(
            message_history=_filter_user_history(message_history)
        )
        self._user_state = user_state
        trajectory = list(message_history)
        self._trajectory = trajectory
        if not message_history:
            first_assistant = _build_default_greeting()
            trajectory.append(first_assistant)
            user_message, user_state = user_sim.generate_next_message(first_assistant, user_state)
            trajectory.append(user_message)
            self._user_state = user_state
            if getattr(user_message, "tool_calls", None):
                self._resolve_user_tool_calls(user_message)
        else:
            last = message_history[-1]
            if _needs_user_reply(last):
                user_message, user_state = user_sim.generate_next_message(last, user_state)
                trajectory.append(user_message)
                self._user_state = user_state
                if getattr(user_message, "tool_calls", None):
                    self._resolve_user_tool_calls(user_message)

        # STEP 3: Prepare sample payload updates (messages + tools + metadata).
        tools_schema = [tool.openai_schema for tool in env.get_tools()]
        sample_messages = [msg for msg in (_tau2_to_gage_message(m) for m in trajectory) if msg]
        _update_tau2_metadata(sample, env)
        sample["messages"] = sample_messages
        sample["tools"] = tools_schema
        sample.setdefault("tool_choice", "auto")

        return {
            "messages": sample_messages,
            "tools_schema": tools_schema,
            "metadata": sample.get("metadata"),
        }

    def exec_tool(self, name: str, arguments: Any) -> Dict[str, Any]:
        if not self._running:
            raise RuntimeError("tau2_runtime_not_started")
        if self._termination_reason is not None:
            return {"error": "tau2_simulation_terminated", "final_answer": "simulation_terminated"}

        if name == self._respond_tool_name:
            return self._handle_respond(arguments)
        return self._handle_env_tool(name, arguments)

    def get_state(self) -> Dict[str, Any]:
        return {
            "task_id": self._task_id,
            "trial": self._trial,
            "seed": self._seed,
            "domain": self._domain,
            "messages": list(self._trajectory),
            "termination_reason": self._termination_reason,
            "user_cost": self._user_cost_total,
            "agent_cost": self._agent_cost_total,
            "start_time": self._start_time,
        }

    def _handle_env_tool(self, name: str, arguments: Any) -> Dict[str, Any]:
        if self._env is None:
            raise RuntimeError("tau2_env_not_initialized")
        tool_call = _build_tool_call(name, arguments, requestor="assistant")
        assistant_msg = _build_assistant_tool_message([tool_call])
        self._trajectory.append(assistant_msg)
        tool_msg = self._env.get_response(tool_call)
        if getattr(tool_msg, "error", False):
            self._error_count += 1
        self._trajectory.append(tool_msg)
        self._step_count += 1
        self._maybe_terminate()
        return {"content": tool_msg.content, "error": bool(getattr(tool_msg, "error", False))}

    def _handle_respond(self, arguments: Any) -> Dict[str, Any]:
        if self._user is None or self._user_state is None:
            raise RuntimeError("tau2_user_not_initialized")
        message_text = _extract_message(arguments)
        assistant_msg = _build_assistant_text_message(message_text)
        self._trajectory.append(assistant_msg)
        self._step_count += 1
        if _is_agent_stop(message_text):
            self._termination_reason = _termination_reason("agent_stop")
            return {"final_answer": message_text, "user_message": message_text}

        user_msg, self._user_state = self._user.generate_next_message(assistant_msg, self._user_state)
        self._trajectory.append(user_msg)
        self._user_cost_total += _coerce_float(getattr(user_msg, "cost", None)) or 0.0

        if _is_user_stop(user_msg):
            self._termination_reason = _termination_reason("user_stop")
            return {"final_answer": user_msg.content, "user_message": user_msg.content}

        if getattr(user_msg, "tool_calls", None):
            user_msg = self._resolve_user_tool_calls(user_msg)
            if _is_user_stop(user_msg):
                self._termination_reason = _termination_reason("user_stop")
                return {"final_answer": user_msg.content, "user_message": user_msg.content}

        self._maybe_terminate()
        return {"user_message": user_msg.content}

    def _resolve_user_tool_calls(self, user_msg: Any) -> Any:
        if self._env is None or self._user is None or self._user_state is None:
            return user_msg
        max_loops = max(1, self._max_steps)
        current = user_msg
        loops = 0
        while getattr(current, "tool_calls", None) and loops < max_loops:
            loops += 1
            tool_messages = []
            for tool_call in current.tool_calls:
                tool_msg = self._env.get_response(tool_call)
                if getattr(tool_msg, "error", False):
                    self._error_count += 1
                tool_messages.append(tool_msg)
                self._trajectory.append(tool_msg)
            next_input = _build_multi_tool_message(tool_messages)
            next_user_msg, self._user_state = self._user.generate_next_message(next_input, self._user_state)
            self._trajectory.append(next_user_msg)
            self._user_cost_total += _coerce_float(getattr(next_user_msg, "cost", None)) or 0.0
            current = next_user_msg
            if _is_user_stop(current):
                break
        return current

    def _maybe_terminate(self) -> None:
        if self._termination_reason is not None:
            return
        if self._step_count >= self._max_steps:
            self._termination_reason = _termination_reason("max_steps")
        elif self._error_count >= self._max_errors:
            self._termination_reason = _termination_reason("too_many_errors")


def _build_tau2_task(sample: Dict[str, Any]) -> Any:
    raw_assets = sample.get("raw_assets") if isinstance(sample.get("raw_assets"), dict) else {}
    tau2_payload = raw_assets.get("tau2") if isinstance(raw_assets.get("tau2"), dict) else {}
    task_payload = tau2_payload.get("task") or sample.get("task") or sample
    try:
        from tau2.data_model.tasks import Task  # type: ignore
    except Exception as exc:
        raise RuntimeError("tau2 is required to build tasks") from exc
    return Task.model_validate(task_payload)


def _resolve_tau2_domain(sample: Dict[str, Any], task: Any) -> str:
    meta = sample.get("metadata") if isinstance(sample.get("metadata"), dict) else {}
    tau2_meta = meta.get("tau2") if isinstance(meta.get("tau2"), dict) else {}
    return str(tau2_meta.get("domain") or getattr(task.user_scenario, "instructions", None) or "airline")


def _read_tau2_meta(sample: Dict[str, Any], key: str) -> Optional[Any]:
    meta = sample.get("metadata") if isinstance(sample.get("metadata"), dict) else {}
    tau2_meta = meta.get("tau2") if isinstance(meta.get("tau2"), dict) else {}
    return tau2_meta.get(key)


def _build_tau2_environment(domain: str) -> Any:
    try:
        from tau2.registry import registry  # type: ignore
    except Exception as exc:
        raise RuntimeError("tau2 registry unavailable") from exc
    env_ctor = registry.get_env_constructor(domain)
    return env_ctor()


def _safe_get_user_tools(env: Any) -> Optional[list[Any]]:
    try:
        return env.get_user_tools()
    except Exception:
        return None


def _build_tau2_user_simulator(
    *,
    tools: Optional[list[Any]],
    instructions: str,
    model: Optional[str],
    model_args: Optional[dict],
    seed: Optional[int],
) -> Any:
    try:
        from tau2.user.user_simulator import UserSimulator  # type: ignore
    except Exception as exc:
        raise RuntimeError("tau2 UserSimulator unavailable") from exc
    user = UserSimulator(
        tools=tools,
        instructions=instructions,
        llm=model,
        llm_args=model_args,
    )
    if seed is not None:
        try:
            user.set_seed(int(seed))
        except Exception:
            pass
    return user


def _filter_user_history(history: List[Any]) -> List[Any]:
    try:
        from tau2.user.base import is_valid_user_history_message  # type: ignore
    except Exception:
        return list(history)
    return [msg for msg in history if is_valid_user_history_message(msg)]


def _build_default_greeting() -> Any:
    try:
        from tau2.orchestrator.orchestrator import DEFAULT_FIRST_AGENT_MESSAGE  # type: ignore
        from tau2.utils.utils import get_now  # type: ignore
    except Exception:
        return _build_assistant_text_message("Hi! How can I help you today?")
    greeting = copy.deepcopy(DEFAULT_FIRST_AGENT_MESSAGE)
    greeting.timestamp = get_now()
    return greeting


def _needs_user_reply(last_message: Any) -> bool:
    try:
        from tau2.data_model.message import AssistantMessage, ToolMessage  # type: ignore
    except Exception:
        return False
    if isinstance(last_message, AssistantMessage):
        return True
    if isinstance(last_message, ToolMessage) and getattr(last_message, "requestor", None) == "user":
        return True
    return False


def _tau2_to_gage_message(message: Any) -> Optional[Dict[str, Any]]:
    role = getattr(message, "role", None)
    content = getattr(message, "content", None)
    if role in {"assistant", "user"}:
        if content is None:
            return None
        return {
            "role": role,
            "content": [{"type": "text", "text": str(content)}],
        }
    if role == "tool":
        tool_call_id = getattr(message, "id", None)
        payload: Dict[str, Any] = {"role": "tool", "content": content or ""}
        if tool_call_id:
            payload["tool_call_id"] = tool_call_id
        return payload
    return None


def _build_tool_call(name: str, arguments: Any, *, requestor: str) -> Any:
    try:
        from tau2.data_model.message import ToolCall  # type: ignore
    except Exception as exc:
        raise RuntimeError("tau2 ToolCall unavailable") from exc
    payload = _normalize_tool_args(arguments)
    return ToolCall(
        id=str(uuid.uuid4()),
        name=str(name),
        arguments=payload,
        requestor=requestor,
    )


def _normalize_tool_args(arguments: Any) -> Dict[str, Any]:
    if arguments is None:
        return {}
    if isinstance(arguments, dict):
        return dict(arguments)
    return {"value": arguments}


def _build_assistant_text_message(content: str) -> Any:
    try:
        from tau2.data_model.message import AssistantMessage  # type: ignore
    except Exception as exc:
        raise RuntimeError("tau2 AssistantMessage unavailable") from exc
    return AssistantMessage(role="assistant", content=content)


def _build_assistant_tool_message(tool_calls: List[Any]) -> Any:
    try:
        from tau2.data_model.message import AssistantMessage  # type: ignore
    except Exception as exc:
        raise RuntimeError("tau2 AssistantMessage unavailable") from exc
    return AssistantMessage(role="assistant", content=None, tool_calls=tool_calls)


def _build_multi_tool_message(tool_messages: List[Any]) -> Any:
    try:
        from tau2.data_model.message import MultiToolMessage  # type: ignore
    except Exception as exc:
        raise RuntimeError("tau2 MultiToolMessage unavailable") from exc
    return MultiToolMessage(role="tool", tool_messages=tool_messages)


def _is_user_stop(message: Any) -> bool:
    try:
        from tau2.user.user_simulator import UserSimulator  # type: ignore
    except Exception:
        return False
    if message is None:
        return False
    return UserSimulator.is_stop(message)


def _is_agent_stop(content: Optional[str]) -> bool:
    if not content:
        return False
    return "###STOP###" in content


def _termination_reason(reason: str) -> Any:
    try:
        from tau2.data_model.simulation import TerminationReason  # type: ignore
    except Exception:
        return reason
    mapping = {
        "user_stop": TerminationReason.USER_STOP,
        "agent_stop": TerminationReason.AGENT_STOP,
        "max_steps": TerminationReason.MAX_STEPS,
        "too_many_errors": TerminationReason.TOO_MANY_ERRORS,
        "agent_error": TerminationReason.AGENT_ERROR,
        "user_error": TerminationReason.USER_ERROR,
    }
    return mapping.get(reason, TerminationReason.AGENT_ERROR)


def _update_tau2_metadata(sample: Dict[str, Any], env: Any) -> None:
    try:
        from tau2.agent.llm_agent import AGENT_INSTRUCTION  # type: ignore
    except Exception:
        AGENT_INSTRUCTION = "Follow the provided policy to help the user."
    meta = sample.get("metadata") if isinstance(sample.get("metadata"), dict) else {}
    tau2_meta = meta.get("tau2") if isinstance(meta.get("tau2"), dict) else {}
    tau2_meta["policy"] = env.get_policy()
    tau2_meta["agent_instruction"] = AGENT_INSTRUCTION
    tau2_meta["gage_instruction"] = (
        "When you want to reply to the user, call the respond tool with your message. "
        "Do not send plain assistant messages directly."
    )
    meta["tau2"] = tau2_meta
    sample["metadata"] = meta


def _extract_message(arguments: Any) -> str:
    if isinstance(arguments, dict):
        for key in ("message", "content", "text", "response"):
            if key in arguments:
                value = arguments.get(key)
                return "" if value is None else str(value)
        return str(arguments)
    if arguments is None:
        return ""
    return str(arguments)


def _coerce_int(value: Any) -> Optional[int]:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _coerce_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _now() -> str:
    return datetime.now().isoformat()
