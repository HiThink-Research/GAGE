"""Default helper_model prompt definitions."""

from __future__ import annotations

from gage_eval.assets.prompts.assets import PromptTemplateAsset
from gage_eval.registry import registry


def _register_prompt(prompt_id: str, desc: str, template: str, *, renderer: str = "jinja") -> None:
    asset = PromptTemplateAsset(
        prompt_id=prompt_id,
        renderer_type=renderer,
        template=template,
    )
    registry.register(
        "prompts",
        prompt_id,
        asset,
        desc=desc,
        tags=("helper", "support"),
        renderer=renderer,
        has_template=bool(template),
    )


def register_support_prompts() -> None:
    """Register helper_model default prompt assets."""

    answer_cleaner_template = """\
You are a strict answer cleaner.
Question:
{% set question = sample.get("metadata", {}).get("question", sample.get("question", sample.get("prompt", ""))) %}
{{ question }}

Model response:
{% set predict_result = sample.get("predict_result", []) %}
{% if predict_result %}
{% set last_pred = predict_result[-1] %}
{% set model_response = last_pred.get("answer", last_pred.get("response", last_pred.get("text", ""))) %}
{{ model_response }}
{% else %}
{{ "" }}
{% endif %}

Output only the final answer. Do not include explanation or extra text.
If there is no clear final answer, output "None".
"""

    _register_prompt(
        "helper/answer_cleaner@v1",
        desc="Helper answer cleaner prompt",
        template=answer_cleaner_template,
    )
