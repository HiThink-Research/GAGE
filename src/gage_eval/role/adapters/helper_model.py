"""Helper model adapter implementation."""

from __future__ import annotations

from typing import Any, Dict

from gage_eval.registry import registry
from gage_eval.role.adapters.model_role_adapter import ModelRoleAdapter


@registry.asset(
    "roles",
    "helper_model",
    desc="Helper role adapter for prompt augmentation / tool orchestration",
    tags=("role", "helper"),
    role_type="helper_model",
)
class HelperModelAdapter(ModelRoleAdapter):
    """Generic helper adapter (prompt rewrite / embedding / retrieval...)."""

    def __init__(
        self,
        adapter_id: str,
        backend: Any,
        capabilities=(),
        mode: str = "default",
        prompt_renderer=None,
        role_type: str = "helper_model",
        **params,
    ) -> None:
        super().__init__(
            adapter_id=adapter_id,
            role_type=role_type or "helper_model",
            capabilities=capabilities,
            backend=backend,
            prompt_renderer=prompt_renderer,
            **params,
        )
        self.mode = mode

    def prepare_backend_request(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        sample = payload.get("sample", {})
        step = payload.get("step", {})
        rendered = self.render_prompt(payload)
        messages = rendered.messages
        if rendered.prompt and not messages:
            messages = [{"role": "user", "content": rendered.prompt}]
        sampling_params = self._compose_sampling_params(
            sample_params=sample.get("sampling_params"),
            runtime_params=payload.get("sampling_params"),
            legacy_params=sample.get("generation_params"),
        )
        return {
            "sample": sample,
            "step": step,
            "mode": step.get("mode") or self.mode,
            "prompt": rendered.prompt or sample.get("prompt") or sample.get("text"),
            "messages": messages,
            "inputs": sample.get("inputs"),
            "sampling_params": sampling_params,
            "prompt_meta": rendered.metadata or {},
        }
