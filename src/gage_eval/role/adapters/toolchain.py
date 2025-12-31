"""Toolchain adapter responsible for MCP / browser orchestration."""

from __future__ import annotations

from typing import Dict, List

from gage_eval.registry import registry
from gage_eval.role.adapters.base import RoleAdapter, RoleAdapterState


@registry.asset(
    "roles",
    "toolchain",
    desc="Unified role adapter for external tools/MCP",
    tags=("role", "tool"),
    role_type="toolchain",
)
class ToolchainAdapter(RoleAdapter):
    def __init__(self, adapter_id: str, tools: List[str]) -> None:
        super().__init__(adapter_id=adapter_id, role_type="toolchain", capabilities=tools)

    async def ainvoke(self, payload: Dict[str, str], state: RoleAdapterState) -> Dict[str, List[str]]:
        trace = [f"Executed {tool}" for tool in self.capabilities]
        return {"tool_trace": trace}
