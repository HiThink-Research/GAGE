"""Tau2 tool-protocol runtime for local in-process evaluation.

Tau2 is a third-layer runtime that executes semantic tools instead of shell
commands. It runs in-process today, but its primary execution contract is
exec_tool/get_state/initialize_task rather than BaseSandbox.exec().
"""

from __future__ import annotations

import copy
import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional

from gage_eval.sandbox.base import BaseSandbox, ExecResult, SandboxOptionalMixin
from gage_eval.utils.benchmark_helpers.tau2 import (
    ensure_tau2_importable,
    resolve_tau2_termination_reason,
    resolve_tau2_data_dir,
)


def _tau2_import(module_path: str, name: str) -> Any:
    """Import a single symbol from a tau2 submodule, raising RuntimeError on failure."""
    try:
        mod = __import__(module_path, fromlist=[name])
        return getattr(mod, name)
    except Exception as exc:
        raise RuntimeError(f"tau2 {name} unavailable (from {module_path})") from exc


class Tau2Runtime(SandboxOptionalMixin, BaseSandbox):
    """Tau2 simulation runtime implementing the tool-based execution contract.

    Layering:
        transport layer: local / in-process
        runtime layer: tau2 simulation state machine

    Execution entrypoints:
        initialize_task(sample): bootstrap environment + user simulator
        exec_tool(name, arguments): advance the simulation
        get_state(): expose evaluator-facing runtime state

    Raw shell execution via exec() is intentionally unsupported.
    All callers access these entrypoints via duck-typing (getattr + callable).
    """

    def __init__(
        self,
        runtime_configs: Optional[Dict[str, Any]] = None,
        resources: Optional[Dict[str, Any]] = None,
    ) -> None:
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

    def exec(
        self, command: str, timeout: int = 30
    ) -> ExecResult:  # pragma: no cover - protocol mismatch
        """Tau2 does not support shell-style execution.

        This runtime is driven through exec_tool()/initialize_task()/get_state()
        rather than raw shell commands.
        """
        raise NotImplementedError(
            "Tau2Runtime uses the tool protocol (exec_tool/get_state/initialize_task), "
            "not the shell protocol (exec). Use exec_tool(name, arguments) instead."
        )

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
            model=self._runtime_configs.get("user_model")
            or self._runtime_configs.get("user_llm"),
            model_args=self._runtime_configs.get("user_model_args")
            or self._runtime_configs.get("user_llm_args"),
            seed=self._seed,
        )
        self._user = user_sim

        initialization_data = (
            getattr(task.initial_state, "initialization_data", None)
            if task.initial_state
            else None
        )
        initialization_actions = (
            getattr(task.initial_state, "initialization_actions", None)
            if task.initial_state
            else None
        )
        message_history = (
            list(getattr(task.initial_state, "message_history", None) or [])
            if task.initial_state
            else []
        )
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
            user_message, user_state = user_sim.generate_next_message(
                first_assistant, user_state
            )
            trajectory.append(user_message)
            self._user_state = user_state
            self._user_cost_total += _resolve_user_message_cost(user_message)
            if getattr(user_message, "tool_calls", None):
                self._resolve_user_tool_calls(user_message)
        else:
            last = message_history[-1]
            if _needs_user_reply(last):
                user_message, user_state = user_sim.generate_next_message(
                    last, user_state
                )
                trajectory.append(user_message)
                self._user_state = user_state
                self._user_cost_total += _resolve_user_message_cost(user_message)
                if getattr(user_message, "tool_calls", None):
                    self._resolve_user_tool_calls(user_message)

        # STEP 3: Prepare sample payload updates (messages + tools + metadata).
        tools_schema = [tool.openai_schema for tool in env.get_tools()]
        tools_schema = tools_schema + [_build_respond_tool_schema(self._respond_tool_name)]
        sample_messages = [
            msg for msg in (_tau2_to_gage_message(m) for m in trajectory) if msg
        ]
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
            return {
                "error": "tau2_simulation_terminated",
                "final_answer": "simulation_terminated",
            }

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

    def record_agent_usage(self, usage: Any) -> None:
        """Record scheduler-reported agent usage into Tau2 runtime state.

        Args:
            usage: Scheduler-normalized usage payload. Supports either OpenAI-style
                token fields (`total_tokens`, `prompt_tokens`, `completion_tokens`)
                or normalized loop fields (`input_tokens`, `output_tokens`,
                `cost_usd`).
        """

        resolved_cost = _resolve_agent_usage_cost(usage)
        if resolved_cost is None:
            return
        self._agent_cost_total = resolved_cost

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
        return {
            "content": tool_msg.content,
            "error": bool(getattr(tool_msg, "error", False)),
        }

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

        user_msg, self._user_state = self._user.generate_next_message(
            assistant_msg, self._user_state
        )
        self._trajectory.append(user_msg)
        self._user_cost_total += _resolve_user_message_cost(user_msg)

        if _is_user_stop(user_msg):
            self._termination_reason = _termination_reason("user_stop")
            return {"final_answer": user_msg.content, "user_message": user_msg.content}

        if getattr(user_msg, "tool_calls", None):
            user_msg = self._resolve_user_tool_calls(user_msg)
            if _is_user_stop(user_msg):
                self._termination_reason = _termination_reason("user_stop")
                return {
                    "final_answer": user_msg.content,
                    "user_message": user_msg.content,
                }

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
            next_user_msg, self._user_state = self._user.generate_next_message(
                next_input, self._user_state
            )
            self._trajectory.append(next_user_msg)
            self._user_cost_total += _resolve_user_message_cost(next_user_msg)
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
    raw_assets = (
        sample.get("raw_assets") if isinstance(sample.get("raw_assets"), dict) else {}
    )
    tau2_payload = (
        raw_assets.get("tau2") if isinstance(raw_assets.get("tau2"), dict) else {}
    )
    task_payload = tau2_payload.get("task") or sample.get("task") or sample
    Task = _tau2_import("tau2.data_model.tasks", "Task")
    return Task.model_validate(task_payload)


def _resolve_tau2_domain(sample: Dict[str, Any], task: Any) -> str:
    meta = sample.get("metadata") if isinstance(sample.get("metadata"), dict) else {}
    tau2_meta = meta.get("tau2") if isinstance(meta.get("tau2"), dict) else {}
    return str(
        tau2_meta.get("domain")
        or getattr(task.user_scenario, "instructions", None)
        or "airline"
    )


def _read_tau2_meta(sample: Dict[str, Any], key: str) -> Optional[Any]:
    meta = sample.get("metadata") if isinstance(sample.get("metadata"), dict) else {}
    tau2_meta = meta.get("tau2") if isinstance(meta.get("tau2"), dict) else {}
    return tau2_meta.get(key)


def _build_tau2_environment(domain: str) -> Any:
    registry = _tau2_import("tau2.registry", "registry")
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
    UserSimulator = _tau2_import("tau2.user.user_simulator", "UserSimulator")
    normalized_model_args = _normalize_tau2_user_model_args(
        model=model,
        model_args=model_args,
    )
    user = UserSimulator(
        tools=tools,
        instructions=instructions,
        llm=model,
        llm_args=normalized_model_args,
    )
    if seed is not None:
        try:
            user.set_seed(int(seed))
        except Exception:
            pass
    return user


def _normalize_tau2_user_model_args(
    *,
    model: Optional[str],
    model_args: Optional[dict],
) -> Optional[dict]:
    """Normalize LiteLLM user-simulator args for provider-specific routing."""

    if not isinstance(model_args, dict):
        return model_args
    normalized = dict(model_args)
    if isinstance(model, str) and model.startswith("ollama_chat/"):
        api_base = normalized.get("api_base")
        if isinstance(api_base, str) and api_base.endswith("/v1"):
            normalized["api_base"] = api_base[:-3]
    return normalized


def _filter_user_history(history: List[Any]) -> List[Any]:
    try:
        is_valid = _tau2_import("tau2.user.base", "is_valid_user_history_message")
    except RuntimeError:
        return list(history)
    return [msg for msg in history if is_valid(msg)]


def _build_default_greeting() -> Any:
    try:
        DEFAULT_FIRST_AGENT_MESSAGE = _tau2_import(
            "tau2.orchestrator.orchestrator", "DEFAULT_FIRST_AGENT_MESSAGE"
        )
        get_now = _tau2_import("tau2.utils.utils", "get_now")
    except RuntimeError:
        return _build_assistant_text_message("Hi! How can I help you today?")
    greeting = copy.deepcopy(DEFAULT_FIRST_AGENT_MESSAGE)
    greeting.timestamp = get_now()
    return greeting


def _needs_user_reply(last_message: Any) -> bool:
    try:
        AssistantMessage = _tau2_import("tau2.data_model.message", "AssistantMessage")
        ToolMessage = _tau2_import("tau2.data_model.message", "ToolMessage")
    except RuntimeError:
        return False
    if isinstance(last_message, AssistantMessage):
        return True
    if (
        isinstance(last_message, ToolMessage)
        and getattr(last_message, "requestor", None) == "user"
    ):
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


def _build_respond_tool_schema(respond_tool_name: str) -> Dict[str, Any]:
    """Build the OpenAI function schema for the agent-facing respond tool."""
    return {
        "type": "function",
        "function": {
            "name": respond_tool_name,
            "description": "Send a message to the user.",
            "parameters": {
                "type": "object",
                "properties": {
                    "message": {
                        "type": "string",
                        "description": "The message to send to the user.",
                    }
                },
                "required": ["message"],
            },
        },
    }


def _build_tool_call(name: str, arguments: Any, *, requestor: str) -> Any:
    ToolCall = _tau2_import("tau2.data_model.message", "ToolCall")
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
    AssistantMessage = _tau2_import("tau2.data_model.message", "AssistantMessage")
    return AssistantMessage(role="assistant", content=content)


def _build_assistant_tool_message(tool_calls: List[Any]) -> Any:
    AssistantMessage = _tau2_import("tau2.data_model.message", "AssistantMessage")
    return AssistantMessage(role="assistant", content=None, tool_calls=tool_calls)


def _build_multi_tool_message(tool_messages: List[Any]) -> Any:
    MultiToolMessage = _tau2_import("tau2.data_model.message", "MultiToolMessage")
    return MultiToolMessage(role="tool", tool_messages=tool_messages)


def _is_user_stop(message: Any) -> bool:
    if message is None:
        return False
    try:
        UserSimulator = _tau2_import("tau2.user.user_simulator", "UserSimulator")
    except RuntimeError:
        return False
    return UserSimulator.is_stop(message)


def _is_agent_stop(content: Optional[str]) -> bool:
    if not content:
        return False
    return "###STOP###" in content


def _termination_reason(reason: str) -> Any:
    return resolve_tau2_termination_reason(reason, fallback="too_many_errors")


def _update_tau2_metadata(sample: Dict[str, Any], env: Any) -> None:
    try:
        AGENT_INSTRUCTION = _tau2_import("tau2.agent.llm_agent", "AGENT_INSTRUCTION")
    except RuntimeError:
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


def _resolve_user_message_cost(msg: Any) -> float:
    """Resolve cost from a user simulator message, falling back to token counts.

    For local/free models (e.g. Ollama) monetary cost is always 0.0, but the
    message's ``usage`` dict carries token counts from LiteLLM.  This mirrors
    the fallback logic in ``_resolve_agent_usage_cost`` so that ``user_cost``
    and ``agent_cost`` use a consistent unit (tokens) when USD cost is absent.
    """

    cost = _coerce_float(getattr(msg, "cost", None))
    if cost:
        return cost
    usage = getattr(msg, "usage", None)
    if not isinstance(usage, dict):
        return 0.0
    direct = _coerce_float(usage.get("cost_usd"))
    if direct:
        return direct
    total = _coerce_float(usage.get("total_tokens"))
    if total is not None:
        return total
    prompt = _coerce_float(usage.get("prompt_tokens")) or 0.0
    completion = _coerce_float(usage.get("completion_tokens")) or 0.0
    return prompt + completion


def _resolve_agent_usage_cost(usage: Any) -> Optional[float]:
    """Resolve one stable Tau2 agent cost from scheduler usage payloads."""

    if not isinstance(usage, dict):
        return None

    direct_cost = _coerce_float(usage.get("cost_usd"))
    if direct_cost is not None:
        return direct_cost

    total_tokens = _coerce_float(usage.get("total_tokens"))
    if total_tokens is not None:
        return total_tokens

    prompt_tokens = _coerce_float(usage.get("prompt_tokens"))
    completion_tokens = _coerce_float(usage.get("completion_tokens"))
    if prompt_tokens is not None or completion_tokens is not None:
        return float((prompt_tokens or 0.0) + (completion_tokens or 0.0))

    input_tokens = _coerce_float(usage.get("input_tokens"))
    output_tokens = _coerce_float(usage.get("output_tokens"))
    if input_tokens is not None or output_tokens is not None:
        return float((input_tokens or 0.0) + (output_tokens or 0.0))

    return None


def _now() -> str:
    return datetime.now().isoformat()
