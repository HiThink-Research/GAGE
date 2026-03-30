from __future__ import annotations

from gage_eval.assets.prompts.renderers import PromptRenderResult
from gage_eval.role.adapters.judge_model import JudgeModelAdapter


class _PromptOnlyRenderer:
    def __init__(self, prompt: str) -> None:
        self._prompt = prompt

    def render(self, context) -> PromptRenderResult:
        return PromptRenderResult(prompt=self._prompt)


def test_judge_model_adapter_wraps_prompt_only_render_output_as_messages() -> None:
    adapter = JudgeModelAdapter(
        adapter_id="judge",
        backend=lambda payload: payload,
        prompt_renderer=_PromptOnlyRenderer("Only answer with A or B."),
    )
    payload = {
        "sample": {
            "messages": [{"role": "user", "content": "original task prompt"}],
        },
        "model_output": {"answer": "The best answer is: A"},
    }

    request = adapter.prepare_backend_request(payload)

    assert request["prompt"] == "Only answer with A or B."
    assert request["messages"] == [{"role": "user", "content": "Only answer with A or B."}]

