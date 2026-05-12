from __future__ import annotations

from typing import Any

from gage_eval.agent_runtime.tooling.contracts import ToolExecutionContext, ToolingError


class EnvironmentToolProvider:
    """Executes tools through the trial-scoped environment lease."""

    def call(self, name: str, arguments: dict[str, Any], context: ToolExecutionContext) -> Any:
        if context.environment_lease is None:
            raise ToolingError(
                "client_execution.tool_router.environment_unavailable",
                "environment lease is unavailable",
                details={"tool": name},
            )
        call_tool = getattr(context.environment_lease, "call_tool", None)
        if callable(call_tool):
            return call_tool(name, arguments)
        exec_tool = getattr(context.environment_lease, "exec_tool", None)
        if callable(exec_tool):
            return exec_tool(name, arguments)
        raise ToolingError(
            "client_execution.tool_router.environment_unavailable",
            "environment lease does not expose a tool execution method",
            details={"tool": name},
        )
