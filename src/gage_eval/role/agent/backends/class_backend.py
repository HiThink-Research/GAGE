"""Class-based agent backend implementation."""

from __future__ import annotations

import importlib
from typing import Any, Callable, Dict, Optional, Type

from gage_eval.role.agent.backends.base import AgentBackend, normalize_agent_output


class ClassBackend(AgentBackend):
    """Agent backend that calls a Python class method."""

    def __init__(self, config: Dict[str, Any]) -> None:
        agent_class = config.get("agent_class")
        if not agent_class:
            raise ValueError("agent_class is required for ClassBackend")
        method = config.get("method") or "run"
        init_kwargs = config.get("init_kwargs") or {}
        self._agent = self._build_agent(agent_class, init_kwargs)
        self._method = method

    def invoke(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        target = getattr(self._agent, self._method, None)
        if not callable(target):
            raise AttributeError(f"Agent class missing callable method '{self._method}'")
        return normalize_agent_output(target(payload))

    @staticmethod
    def _build_agent(agent_class: Any, init_kwargs: Dict[str, Any]) -> Any:
        if isinstance(agent_class, str):
            cls = _load_class(agent_class)
        else:
            cls = agent_class
        if isinstance(cls, type):
            return cls(**init_kwargs)
        if callable(cls):
            return _CallableAgent(cls)
        raise TypeError(f"Unsupported agent_class type: {type(agent_class)}")


class _CallableAgent:
    def __init__(self, func: Callable[[Dict[str, Any]], Any]) -> None:
        self._func = func

    def run(self, payload: Dict[str, Any]) -> Any:
        return self._func(payload)


def _load_class(path: str) -> Type[Any]:
    if ":" in path:
        module_name, class_name = path.split(":", 1)
    else:
        module_name, class_name = path.rsplit(".", 1)
    module = importlib.import_module(module_name)
    return getattr(module, class_name)
