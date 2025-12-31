"""DUT model adapter implementation."""

from __future__ import annotations

from typing import Any, Dict

from gage_eval.registry import registry
from gage_eval.role.adapters.model_role_adapter import ModelRoleAdapter
from gage_eval.role.model.runtime import TextGenerationMixin


@registry.asset(
    "roles",
    "dut_model",
    desc="面向被测模型（DUT）的标准角色适配器",
    tags=("role", "dut"),
    role_type="dut_model",
) 
class DUTModelAdapter(TextGenerationMixin, ModelRoleAdapter):
    """Baseline DUT model adapter that delegates execution to the configured backend."""

    def __init__(self, adapter_id: str, role_type: str, backend: Any, capabilities=(), prompt_renderer=None, **params) -> None:
        super().__init__(
            adapter_id=adapter_id,
            role_type=role_type,
            capabilities=capabilities,
            backend=backend,
            prompt_renderer=prompt_renderer,
            **params,
        )
