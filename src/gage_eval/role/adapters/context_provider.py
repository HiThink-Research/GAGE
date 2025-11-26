"""Context provider adapter (RAG/knowledge retriever)."""

from __future__ import annotations

from typing import Dict, List

from gage_eval.registry import registry
from gage_eval.role.adapters.base import RoleAdapter, RoleAdapterState


@registry.asset(
    "roles",
    "context_provider",
    desc="RAG/知识增强上下文提供角色",
    tags=("role", "context"),
    role_type="context_provider",
)
class ContextProviderAdapter(RoleAdapter):
    def __init__(self, adapter_id: str) -> None:
        super().__init__(adapter_id=adapter_id, role_type="context_provider", capabilities=("text",))

    async def ainvoke(self, payload: Dict[str, str], state: RoleAdapterState) -> Dict[str, List[str]]:
        sample = payload.get("sample", {})
        query = payload.get("query") or sample.get("query") or sample.get("prompt", "")
        return {"context": [f"stub context for {query}"]}
