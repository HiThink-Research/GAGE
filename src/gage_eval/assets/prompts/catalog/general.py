"""Canonical prompt definitions shared across DUT/Judge roles."""

from __future__ import annotations

from typing import Dict

from gage_eval.assets.prompts.assets import PromptTemplateAsset
from gage_eval.registry import registry


def _register_prompt(prompt_id: str, desc: str, template: str, *, renderer: str = "jinja", tags: tuple[str, ...] = ()) -> None:
    asset = PromptTemplateAsset(
        prompt_id=prompt_id,
        renderer_type=renderer,
        template=template,
        default_args={"instructions": "Follow the task carefully and answer in Chinese unless specified."},
    )
    registry.register(
        "prompts",
        prompt_id,
        asset,
        desc=desc,
        tags=tags,
        renderer=renderer,
        has_template=bool(template),
    )


def register_general_prompts() -> None:
    """Register the default DUT/Judge prompt assets."""

    dut_template = """\
{% if instructions %}{{ instructions }}{% endif %}
任务：{{ sample.get("question") or sample.get("prompt") or sample.get("inputs", {}).get("prompt") or "" }}
请直接给出答案。"""
    judge_template = """\
你是一名严谨的中文评审员，请根据事实给出判定。
【问题】{{ sample.get("question") or sample.get("prompt") or sample.get("inputs", {}).get("prompt") or "" }}
【参考答案】{{ sample.get("answer") or sample.get("reference") or sample.get("expected_answer") or "" }}
【模型回答】{{ model_output.get("answer") or model_output.get("response") or model_output.get("text") or "" }}
请输出 JSON，包含 verdict(yes/no) 与 rationale 字段。"""

    _register_prompt(
        "dut/general@v1",
        desc="General DUT text-generation prompt",
        template=dut_template,
        tags=("dut", "general"),
    )
    _register_prompt(
        "judge/general@v1",
        desc="General judge scoring prompt",
        template=judge_template,
        tags=("judge", "general"),
    )
