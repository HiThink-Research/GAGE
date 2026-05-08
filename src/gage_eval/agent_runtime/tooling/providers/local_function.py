from __future__ import annotations

from typing import Any, Callable

from gage_eval.agent_runtime.tooling.contracts import ToolExecutionContext


class LocalFunctionToolProvider:
    """Small adapter for host-local function tools."""

    def __init__(self, function: Callable[..., Any]) -> None:
        self._function = function

    def call(self, arguments: Any, context: ToolExecutionContext) -> Any:
        return self._function(arguments, context)
