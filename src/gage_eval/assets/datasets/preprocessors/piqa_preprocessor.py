"""PIQA preprocessors built on MultiChoicePreprocessor."""

from __future__ import annotations

from typing import Any, Dict

from gage_eval.assets.datasets.preprocessors.multi_choice_preprocessor import MultiChoicePreprocessor


class PiqaPreprocessor(MultiChoicePreprocessor):
    """Preprocesses PIQA multiple-choice records into the Sample schema.

    Prompt rendering can be customized via `kwargs` passed to the base class.
    """

    def to_sample(self, record: Dict[str, Any], **kwargs: Any) -> Dict[str, Any]:
        sample = dict(record)
        # NOTE: Support both raw PIQA fields (goal/sol1/sol2/label) and already
        # structured fields (question/choices/answer).
        question = record.get("question") or record.get("goal")
        options = record.get("choices")
        if not options:
            options = [record.get("sol1"), record.get("sol2")]
        answer = record.get("answer", record.get("label"))

        sample["question"] = question
        sample["choices"] = options
        sample["answer"] = answer
        return super().to_sample(
            sample,
            question_field="question",
            options_field="choices",
            answer_field="answer",
            answer_index_base=0,
            **kwargs,
        )


class PiqaStructOnlyPreprocessor(PiqaPreprocessor):
    """PIQA struct-only preprocessing (no prompt rendering).

    This variant keeps `choices`/`metadata` but removes rendered prompts so that
    downstream steps can supply their own prompting strategy.
    """

    def to_sample(self, record: Dict[str, Any], **kwargs: Any) -> Dict[str, Any]:
        sample = super().to_sample(record, **kwargs)
        # NOTE: Keep minimal fields to satisfy envelope validation while removing
        # render flags and prompt inputs.
        sample.pop("prompt", None)
        sample["messages"] = []
        sample["inputs"] = {}
        sample.pop("chat_template_mode", None)
        sample.pop("rendered_by", None)
        sample.pop("template_source", None)
        sample.pop("cache_suffix", None)
        return sample


__all__ = ["PiqaPreprocessor", "PiqaStructOnlyPreprocessor"]
from gage_eval.assets.datasets.utils.rendering import strip_render_flags
