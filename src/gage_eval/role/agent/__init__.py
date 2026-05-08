"""Agent hook registry retained for benchmark lifecycle adapters."""

from gage_eval.role.agent.hooks import (
    AgentHookContext,
    AgentLoopHook,
    build_hook_chain,
    register_hook,
    register_hook_aliases,
)

__all__ = [
    "AgentHookContext",
    "AgentLoopHook",
    "build_hook_chain",
    "register_hook",
    "register_hook_aliases",
]
