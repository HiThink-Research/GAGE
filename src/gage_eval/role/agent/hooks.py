"""AgentLoop hook abstractions for task-specific setup and teardown."""

from __future__ import annotations

import importlib
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterable, List, Optional, Protocol, Sequence

_HOOK_REGISTRY: Dict[str, Callable[..., AgentLoopHook]] = {}
_BUILTIN_HOOK_MODULES = ("gage_eval.sandbox.integrations.appworld.hooks",)
_BUILTIN_HOOKS_LOADED = False


class AgentLoopHook(Protocol):
    """Execute a hook with the provided context."""

    def run(self, context: AgentHookContext) -> Optional[Dict[str, Any]]:
        """Run the hook.

        Args:
            context: Hook context carrying sample metadata and runtime handles.

        Returns:
            Optional hook result payload.
        """


@dataclass
class AgentHookContext:
    """Shared context for AgentLoop hooks."""

    sample: Dict[str, Any]
    metadata: Dict[str, Any]
    runtime_handle: Dict[str, Any]
    sandbox_config: Dict[str, Any]
    agent_trace: Optional[List[Dict[str, Any]]] = None
    hook_state: Dict[str, Any] = field(default_factory=dict)

    def resolve_endpoint(self, *keys: str) -> Optional[str]:
        """Resolve endpoint values from runtime handles or sandbox configs."""

        runtime_handle = self.runtime_handle or {}
        sandbox_config = self.sandbox_config or {}
        runtime_configs = sandbox_config.get("runtime_configs") or {}
        for key in keys:
            if key in runtime_handle:
                return str(runtime_handle[key])
            if key in runtime_configs:
                return str(runtime_configs[key])
            if key in sandbox_config:
                return str(sandbox_config[key])
        return None

    def with_agent_trace(self, agent_trace: List[Dict[str, Any]]) -> AgentHookContext:
        """Clone the hook context with agent trace attached."""

        return AgentHookContext(
            sample=self.sample,
            metadata=self.metadata,
            runtime_handle=self.runtime_handle,
            sandbox_config=self.sandbox_config,
            agent_trace=agent_trace,
            hook_state=self.hook_state,
        )


def build_hook_chain(hooks: Optional[Iterable[Any]]) -> List[AgentLoopHook]:
    """Build hook instances from hook specs or callables.

    Args:
        hooks: Hook objects or hook specs.

    Returns:
        Materialized hook list.
    """

    if hooks is None:
        return []
    if isinstance(hooks, dict):
        hooks = [hooks]
    chain: List[AgentLoopHook] = []
    for entry in hooks:
        if entry is None:
            continue
        if isinstance(entry, dict):
            chain.append(_build_hook_from_spec(entry))
            continue
        chain.append(entry)
    return chain


def register_hook(hook_type: str, factory: Callable[..., AgentLoopHook]) -> None:
    """Register a hook factory for a given hook type."""

    _HOOK_REGISTRY[str(hook_type)] = factory


def register_hook_aliases(hook_types: Sequence[str], factory: Callable[..., AgentLoopHook]) -> None:
    """Register multiple hook type aliases for a single factory."""

    for hook_type in hook_types:
        register_hook(hook_type, factory)


def _build_hook_from_spec(spec: Dict[str, Any]) -> AgentLoopHook:
    hook_type = spec.get("type") or spec.get("hook_type")
    params = spec.get("params") or spec.get("config") or {}
    if not hook_type:
        raise ValueError("Hook spec missing type")
    hook_type = str(hook_type)
    _ensure_builtin_hooks_loaded()
    factory = _HOOK_REGISTRY.get(hook_type)
    if factory is None:
        raise ValueError(f"Unknown hook type '{hook_type}'")
    return factory(**params)


def _ensure_builtin_hooks_loaded() -> None:
    global _BUILTIN_HOOKS_LOADED
    if _BUILTIN_HOOKS_LOADED:
        return
    for module_path in _BUILTIN_HOOK_MODULES:
        importlib.import_module(module_path)
    _BUILTIN_HOOKS_LOADED = True
