"""Judge model adapter."""

from __future__ import annotations

from typing import Any, Dict

from gage_eval.registry import registry
from gage_eval.role.adapters.model_role_adapter import ModelRoleAdapter


@registry.asset(
    "roles",
    "judge_model",
    desc="Judge/scoring LLM role adapter",
    tags=("role", "judge"),
    role_type="judge_model",
)
class JudgeModelAdapter(ModelRoleAdapter):
    def __init__(
        self,
        adapter_id: str,
        backend: Any,
        capabilities=(),
        prompt_renderer=None,
        role_type: str = "judge_model",
        **params,
    ) -> None:
        super().__init__(
            adapter_id=adapter_id,
            role_type=role_type or "judge_model",
            capabilities=capabilities,
            backend=backend,
            prompt_renderer=prompt_renderer,
            **params,
        )

    def prepare_backend_request(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        sample = payload.get("sample", {})
        model_output = payload.get("model_output", {})
        rendered = self.render_prompt(payload)
        question = rendered.prompt or sample.get("question") or sample.get("prompt") or sample.get("text") or ""
        request = {
            "sample": sample,
            "question": question,
            "answer": model_output.get("answer"),
            "model_output": model_output,
            "messages": rendered.messages,
        }
        if rendered.metadata:
            request["prompt_meta"] = rendered.metadata
        return request
