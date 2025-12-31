"""DUT agent adapter (multi-tool agent under sandbox control)."""

from __future__ import annotations

from typing import Dict

from gage_eval.registry import registry
from gage_eval.role.adapters.base import RoleAdapter, RoleAdapterState


@registry.asset(
    "roles",
    "dut_agent",
    desc="具备工具调用能力的 DUT Agent 适配器",
    tags=("role", "agent"),
    role_type="dut_agent",
)
class DUTAgentAdapter(RoleAdapter):
    def __init__(self, adapter_id: str, tools: Dict[str, str]) -> None:
        super().__init__(adapter_id=adapter_id, role_type="dut_agent", capabilities=tuple(tools.keys()))
        self._tools = tools

    async def ainvoke(self, payload: Dict[str, str], state: RoleAdapterState) -> Dict[str, str]:
        return {"agent_trace": list(self._tools.keys()), "answer": payload.get("question", "")}
