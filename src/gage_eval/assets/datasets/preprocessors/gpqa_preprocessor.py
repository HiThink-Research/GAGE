"""GPQA preprocessors built on MultiChoicePreprocessor."""

from __future__ import annotations

import random
from typing import Any, Dict

from gage_eval.assets.datasets.preprocessors.multi_choice_preprocessor import MultiChoicePreprocessor
from gage_eval.assets.datasets.utils.rendering import (
    contains_multimodal,
    render_messages_with_fallback,
    set_render_flags,
    strip_render_flags,
)


class GpqaPreprocessor(MultiChoicePreprocessor):
    """Preprocess GPQA multiple-choice records.

    Steps:
        1) Extract the question and answer options.
        2) Shuffle options to remove position bias.
        3) Compute the correct answer index.
        4) Delegate to `MultiChoicePreprocessor` for standardized Sample rendering.
    """

    def to_sample(self, record: Dict[str, Any], **kwargs: Any) -> Dict[str, Any]:
        sample = dict(record)

        # STEP 1: Extract fields.
        question = record.get("Question")
        correct_answer = record.get("Correct Answer")
        incorrect_answers = [
            record.get("Incorrect Answer 1"),
            record.get("Incorrect Answer 2"),
            record.get("Incorrect Answer 3"),
        ]

        # NOTE: Keep best-effort behavior for partially missing records.
        if question is None or correct_answer is None:
            pass

        # STEP 2: Build the option list.
        options = [correct_answer] + [opt for opt in incorrect_answers if opt is not None]

        # STEP 3: Shuffle options.
        random.shuffle(options)

        # STEP 4: Determine the correct option index.
        try:
            answer_index = options.index(correct_answer)
        except ValueError:
            answer_index = 0

        # STEP 5: Adapt fields for `MultiChoicePreprocessor`.
        sample["question"] = question
        sample["choices"] = options
        # NOTE: Pass the answer as an integer index.
        sample["answer"] = answer_index

        # STEP 6: Delegate to the base preprocessor for normalization/rendering.
        sample = super().to_sample(
            sample,
            question_field="question",
            options_field="choices",
            answer_field="answer",
            answer_index_base=0,
            **kwargs,
        )
        # NOTE: Try to reuse the tokenizer chat template and emit `input_ids` when possible (llm-eval compatible).
        messages = sample.get("messages")
        if (
            self._tokenizer is not None
            and isinstance(messages, list)
            and messages
            and not contains_multimodal(messages)
        ):
            prompt, source = render_messages_with_fallback(messages, self._tokenizer)
            sample["prompt"] = prompt
            sample["inputs"] = {"prompt": prompt}
            try:
                if hasattr(self._tokenizer, "encode"):
                    sample["inputs"]["input_ids"] = self._tokenizer.encode(prompt)
            except Exception:
                # NOTE: Keep prompt-only on encoding failures to avoid breaking preprocessing.
                pass
            set_render_flags(
                sample,
                mode="preprocess",
                source=source,
                rendered_by="preprocess",
                cache_suffix="-chat_template" if source == "model" else "-plain",
            )
            if getattr(self, "_tokenizer_path", None) and "_tokenizer_path" not in sample:
                sample["_tokenizer_path"] = self._tokenizer_path
        return sample


class GpqaStructOnlyPreprocessor(GpqaPreprocessor):
    """Preprocess GPQA into structured fields only (no prompt rendering)."""

    def to_sample(self, record: Dict[str, Any], **kwargs: Any) -> Dict[str, Any]:
        sample = super().to_sample(record, **kwargs)
        # Keep messages/choices for envelope validation; drop rendered outputs and prompt inputs.
        sample.pop("prompt", None)
        sample["messages"] = []
        sample["inputs"] = {}
        strip_render_flags(sample)
        return sample

__all__ = ["GpqaPreprocessor", "GpqaStructOnlyPreprocessor"]
