"""Sandbox execution protocol definitions.

Defines runtime_checkable Protocol classes for the three duck-typed contracts
used by third-layer runtimes (e.g. Tau2). These protocols formalize the
exec_tool / get_state / initialize_task interfaces that callers already
access via getattr + callable checks.

Callers are NOT required to switch to isinstance checks; the protocols exist
to document the contract and enable optional static type checking.
"""

from __future__ import annotations

from typing import Any, Dict, Protocol, runtime_checkable


@runtime_checkable
class ToolExecutionProtocol(Protocol):
    """Tool-based runtime execution contract.

    Runtimes implementing this protocol execute semantic tool calls instead
    of shell commands. This is the primary execution entrypoint for
    tool-protocol runtimes such as Tau2.
    """

    def exec_tool(self, name: str, arguments: Any) -> Dict[str, Any]: ...


@runtime_checkable
class StateQueryProtocol(Protocol):
    """Runtime state query contract for judges and evaluators.

    Allows external components (e.g. tau2_eval) to read the runtime's
    internal simulation state without coupling to implementation details.
    """

    def get_state(self) -> Dict[str, Any]: ...


@runtime_checkable
class TaskInitProtocol(Protocol):
    """Task bootstrap contract for runtimes that prepare sample state.

    Runtimes implementing this protocol accept a sample dict, initialize
    internal task/environment state, and return enriched metadata (messages,
    tools, etc.) back to the caller.
    """

    def initialize_task(self, sample: Dict[str, Any]) -> Dict[str, Any]: ...
