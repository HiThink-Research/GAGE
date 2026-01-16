"""Agent runtime utilities."""

from gage_eval.role.agent.hooks import (
    AgentHookContext,
    AgentLoopHook,
    build_hook_chain,
    register_hook,
    register_hook_aliases,
)
from gage_eval.role.agent.human_gateway import HumanGateway
from gage_eval.role.agent.loop import AgentLoop
from gage_eval.role.agent.tool_router import ToolRouter

__all__ = [
    "AgentLoop",
    "ToolRouter",
    "HumanGateway",
    "AgentHookContext",
    "AgentLoopHook",
    "build_hook_chain",
    "register_hook",
    "register_hook_aliases",
]
