from __future__ import annotations

import json
import sys
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any, Optional


STOP = "###STOP###"
TRANSFER = "###TRANSFER###"
OUT_OF_SCOPE = "###OUT-OF-SCOPE###"


def install_tau2_stub(monkeypatch, *, data_dir: Path, force_user_tool_call: bool = False) -> None:
    tau2 = ModuleType("tau2")

    # utils.utils
    utils_utils = ModuleType("tau2.utils.utils")
    utils_utils.DATA_DIR = data_dir
    utils_utils.get_now = lambda: "2025-01-01T00:00:00"
    utils_mod = ModuleType("tau2.utils")
    utils_mod.utils = utils_utils
    tau2.utils = utils_mod

    # data_model.message
    message_mod = ModuleType("tau2.data_model.message")

    @dataclass
    class ToolCall:
        id: str
        name: str
        arguments: dict
        requestor: str = "assistant"

    @dataclass
    class ParticipantMessageBase:
        role: str
        content: Optional[str] = None
        tool_calls: Optional[list[ToolCall]] = None
        cost: Optional[float] = None
        usage: Optional[dict] = None
        raw_data: Optional[dict] = None
        timestamp: Optional[str] = None
        turn_idx: Optional[int] = None

        def is_tool_call(self) -> bool:
            return self.tool_calls is not None

        def has_text_content(self) -> bool:
            return bool(self.content and str(self.content).strip())

    @dataclass
    class AssistantMessage(ParticipantMessageBase):
        role: str = "assistant"

    @dataclass
    class UserMessage(ParticipantMessageBase):
        role: str = "user"

    @dataclass
    class ToolMessage:
        id: str
        role: str = "tool"
        content: Optional[str] = None
        requestor: str = "assistant"
        error: bool = False
        timestamp: Optional[str] = None
        turn_idx: Optional[int] = None

    @dataclass
    class MultiToolMessage:
        role: str = "tool"
        tool_messages: list[ToolMessage] = field(default_factory=list)

    @dataclass
    class SystemMessage:
        role: str = "system"
        content: Optional[str] = None

    message_mod.ToolCall = ToolCall
    message_mod.AssistantMessage = AssistantMessage
    message_mod.UserMessage = UserMessage
    message_mod.ToolMessage = ToolMessage
    message_mod.MultiToolMessage = MultiToolMessage
    message_mod.SystemMessage = SystemMessage
    message_mod.Message = (AssistantMessage, UserMessage, ToolMessage, MultiToolMessage, SystemMessage)

    # data_model.tasks
    tasks_mod = ModuleType("tau2.data_model.tasks")

    class RewardType(str, Enum):
        DB = "DB"
        COMMUNICATE = "COMMUNICATE"
        ACTION = "ACTION"
        ENV_ASSERTION = "ENV_ASSERTION"
        NL_ASSERTION = "NL_ASSERTION"

    @dataclass
    class EvaluationCriteria:
        reward_basis: list[Any] = field(default_factory=lambda: [RewardType.DB])
        actions: list[Any] = field(default_factory=list)
        communicate_info: list[str] = field(default_factory=list)
        nl_assertions: list[str] = field(default_factory=list)

        def model_dump(self, mode: str | None = None) -> dict:
            return {
                "reward_basis": [str(x) for x in self.reward_basis],
                "actions": self.actions,
                "communicate_info": self.communicate_info,
                "nl_assertions": self.nl_assertions,
            }

    @dataclass
    class UserScenario:
        instructions: Any

        def __str__(self) -> str:
            return str(self.instructions)

    @dataclass
    class Task:
        id: str
        user_scenario: UserScenario
        evaluation_criteria: Optional[EvaluationCriteria] = None
        initial_state: Optional[Any] = None

        @classmethod
        def model_validate(cls, payload: dict) -> "Task":
            user_scenario = payload.get("user_scenario") or {}
            instructions = user_scenario.get("instructions") if isinstance(user_scenario, dict) else user_scenario
            evaluation = payload.get("evaluation_criteria")
            eval_obj = None
            if isinstance(evaluation, dict):
                eval_obj = EvaluationCriteria(
                    reward_basis=evaluation.get("reward_basis") or [RewardType.DB]
                )
            return cls(
                id=str(payload.get("id")),
                user_scenario=UserScenario(instructions=instructions),
                evaluation_criteria=eval_obj,
                initial_state=payload.get("initial_state"),
            )

        def model_dump(self, mode: str | None = None) -> dict:
            return {
                "id": self.id,
                "user_scenario": {"instructions": self.user_scenario.instructions},
                "evaluation_criteria": self.evaluation_criteria.model_dump() if self.evaluation_criteria else None,
                "initial_state": self.initial_state,
            }

    tasks_mod.Task = Task
    tasks_mod.RewardType = RewardType

    # data_model.simulation
    sim_mod = ModuleType("tau2.data_model.simulation")

    class TerminationReason(str, Enum):
        USER_STOP = "user_stop"
        AGENT_STOP = "agent_stop"
        MAX_STEPS = "max_steps"
        TOO_MANY_ERRORS = "too_many_errors"
        AGENT_ERROR = "agent_error"
        USER_ERROR = "user_error"

    @dataclass
    class RewardInfo:
        reward: float
        reward_basis: Optional[list[Any]] = None
        reward_breakdown: Optional[dict] = None
        info: Optional[dict] = None

        def model_dump(self, mode: str | None = None) -> dict:
            return {
                "reward": self.reward,
                "reward_basis": [str(x) for x in (self.reward_basis or [])],
                "reward_breakdown": self.reward_breakdown,
                "info": self.info,
            }

    @dataclass
    class SimulationRun:
        id: str
        task_id: str
        start_time: str
        end_time: str
        duration: float
        termination_reason: TerminationReason
        agent_cost: Optional[float] = None
        user_cost: Optional[float] = None
        reward_info: Optional[RewardInfo] = None
        messages: list[Any] = field(default_factory=list)
        trial: Optional[int] = None
        seed: Optional[int] = None

    sim_mod.TerminationReason = TerminationReason
    sim_mod.RewardInfo = RewardInfo
    sim_mod.SimulationRun = SimulationRun

    # environment.tool
    tool_mod = ModuleType("tau2.environment.tool")

    @dataclass
    class Tool:
        name: str
        description: str = ""

        @property
        def openai_schema(self) -> dict:
            return {
                "type": "function",
                "function": {
                    "name": self.name,
                    "description": self.description,
                    "parameters": {"type": "object", "properties": {}},
                },
            }

    tool_mod.Tool = Tool

    # environment.environment
    env_mod = ModuleType("tau2.environment.environment")

    class Environment:
        def __init__(self, domain_name: str, policy: str, tools: list[Tool], user_tools: Optional[list[Tool]] = None):
            self.domain_name = domain_name
            self._policy = policy
            self._tools = tools
            self._user_tools = user_tools or []

        def get_policy(self) -> str:
            return self._policy

        def get_tools(self) -> list[Tool]:
            return self._tools

        def get_user_tools(self) -> list[Tool]:
            return self._user_tools

        def set_state(self, **_kwargs: Any) -> None:
            return None

        def get_response(self, tool_call: ToolCall) -> ToolMessage:
            payload = json.dumps({"tool": tool_call.name, "arguments": tool_call.arguments})
            return ToolMessage(id=tool_call.id, role="tool", content=payload, requestor=tool_call.requestor, error=False)

    env_mod.Environment = Environment

    # user.base
    user_base_mod = ModuleType("tau2.user.base")
    user_base_mod.STOP = STOP
    user_base_mod.TRANSFER = TRANSFER
    user_base_mod.OUT_OF_SCOPE = OUT_OF_SCOPE

    def is_valid_user_history_message(_msg: Any) -> bool:
        return True

    user_base_mod.is_valid_user_history_message = is_valid_user_history_message

    # user.user_simulator
    user_sim_mod = ModuleType("tau2.user.user_simulator")

    @dataclass
    class UserState:
        system_messages: list[Any] = field(default_factory=list)
        messages: list[Any] = field(default_factory=list)
        tool_used: bool = False

    class UserSimulator:
        def __init__(self, tools: Optional[list[Tool]] = None, instructions: Optional[str] = None, llm=None, llm_args=None):
            self.tools = tools or []
            self.instructions = instructions or ""
            self._force_tool_call = force_user_tool_call

        def get_init_state(self, message_history: Optional[list[Any]] = None) -> UserState:
            return UserState(messages=list(message_history or []))

        def set_seed(self, seed: int) -> None:
            return None

        def generate_next_message(self, message: Any, state: UserState):
            if self._force_tool_call and not state.tool_used and self.tools:
                state.tool_used = True
                tool_call = ToolCall(
                    id="user_tool_call",
                    name=self.tools[0].name,
                    arguments={"value": "1"},
                    requestor="user",
                )
                return UserMessage(role="user", content=None, tool_calls=[tool_call], cost=0.1), state
            text = "user_response"
            if isinstance(message, AssistantMessage) and message.content:
                if "stop" in message.content.lower():
                    text = STOP
            return UserMessage(role="user", content=text, tool_calls=None, cost=0.1), state

        @classmethod
        def is_stop(cls, message: UserMessage) -> bool:
            return isinstance(message.content, str) and STOP in message.content

    user_sim_mod.UserSimulator = UserSimulator
    user_sim_mod.UserState = UserState

    # agent.llm_agent
    agent_llm_mod = ModuleType("tau2.agent.llm_agent")
    agent_llm_mod.AGENT_INSTRUCTION = "Stub agent instruction."

    # orchestrator.orchestrator
    orchestrator_mod = ModuleType("tau2.orchestrator.orchestrator")
    orchestrator_mod.DEFAULT_FIRST_AGENT_MESSAGE = AssistantMessage(
        role="assistant", content="Hi! How can I help you today?"
    )

    # evaluator.evaluator
    evaluator_mod = ModuleType("tau2.evaluator.evaluator")

    class EvaluationType(str, Enum):
        ALL = "all"
        ALL_WITH_NL_ASSERTIONS = "all_with_nl_assertions"

    def evaluate_simulation(simulation: SimulationRun, task: Task, evaluation_type: EvaluationType, solo_mode: bool, domain: str) -> RewardInfo:
        reward = 1.0 if simulation.termination_reason in {TerminationReason.USER_STOP, TerminationReason.AGENT_STOP} else 0.0
        return RewardInfo(reward=reward, reward_basis=getattr(task.evaluation_criteria, "reward_basis", None), info={"evaluation_type": str(evaluation_type)})

    evaluator_mod.EvaluationType = EvaluationType
    evaluator_mod.evaluate_simulation = evaluate_simulation

    # registry
    registry_mod = ModuleType("tau2.registry")

    def _get_env_constructor(_domain: str):
        def build_env():
            tools = [Tool("lookup")]
            user_tools = [Tool("user_tool")]
            return Environment(domain_name=_domain, policy="policy", tools=tools, user_tools=user_tools)

        return build_env

    registry_mod.registry = SimpleNamespace(get_env_constructor=_get_env_constructor)

    # assemble modules
    modules = {
        "tau2": tau2,
        "tau2.utils": utils_mod,
        "tau2.utils.utils": utils_utils,
        "tau2.data_model": ModuleType("tau2.data_model"),
        "tau2.data_model.message": message_mod,
        "tau2.data_model.tasks": tasks_mod,
        "tau2.data_model.simulation": sim_mod,
        "tau2.environment": ModuleType("tau2.environment"),
        "tau2.environment.tool": tool_mod,
        "tau2.environment.environment": env_mod,
        "tau2.user": ModuleType("tau2.user"),
        "tau2.user.base": user_base_mod,
        "tau2.user.user_simulator": user_sim_mod,
        "tau2.agent": ModuleType("tau2.agent"),
        "tau2.agent.llm_agent": agent_llm_mod,
        "tau2.orchestrator": ModuleType("tau2.orchestrator"),
        "tau2.orchestrator.orchestrator": orchestrator_mod,
        "tau2.evaluator": ModuleType("tau2.evaluator"),
        "tau2.evaluator.evaluator": evaluator_mod,
        "tau2.registry": registry_mod,
    }

    for name, module in modules.items():
        monkeypatch.setitem(sys.modules, name, module)

