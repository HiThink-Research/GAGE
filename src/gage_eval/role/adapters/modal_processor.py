"""Modal processor adapter."""

from __future__ import annotations

from typing import Dict

from gage_eval.registry import registry
from gage_eval.role.adapters.base import RoleAdapter, RoleAdapterState


@registry.asset(
    "roles",
    "modal_processor",
    desc="Multimodal pre/post-processing role adapter",
    tags=("role", "modal"),
    role_type="modal_processor",
)
class ModalProcessorAdapter(RoleAdapter):
    def __init__(self, adapter_id: str, modal_type: str) -> None:
        super().__init__(adapter_id=adapter_id, role_type="modal_processor", capabilities=(modal_type,))

    async def ainvoke(self, payload: Dict[str, str], state: RoleAdapterState) -> Dict[str, str]:
        sample = payload.get("sample", {})
        identifier = sample.get("id", payload.get("id", "sample"))
        return {"artifact_uri": f"processed://{self.capabilities[0]}/{identifier}"}
