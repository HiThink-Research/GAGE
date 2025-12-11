"""Class-based multiple-choice preprocessor (new implementation)."""

from __future__ import annotations

from typing import Any, Dict, List, Sequence, Tuple

from gage_eval.assets.datasets.preprocessors.default_preprocessor import DefaultPreprocessor
from gage_eval.assets.datasets.utils.mapping import (
    extract_field,
    normalize_options,
    resolve_correct_choice,
)
from gage_eval.assets.datasets.utils.rendering import set_render_flags

_CHOICE_ALPHABET: Tuple[str, ...] = tuple("ABCDEFGHIJKLMNOPQRSTUVWXYZ")


class MultiChoicePreprocessor(DefaultPreprocessor):
    """Normalize multiple-choice samples into messages + metadata."""

    def to_sample(
        self,
        record: Dict[str, Any],
        *,
        question_field: str = "question",
        options_field: str | None = None,
        choices_field: str | None = None,
        answer_field: str = "answer",
        answer_index_base: int = 0,
        system_prompt: str | None = None,
        instruction: str | None = "请仅输出正确选项对应的大写字母，例如 'A'。",
        **kwargs: Any,
    ) -> Dict[str, Any]:
        sample = dict(record)
        question = extract_field(sample, question_field)
        if question is None:
            raise ValueError(f"Multiple-choice sample missing question field '{question_field}'")

        option_source = choices_field or options_field or "choices"
        raw_options = extract_field(sample, option_source)
        options = normalize_options(raw_options)
        if len(options) < 2:
            raise ValueError("Multiple-choice sample must provide at least two options")

        option_pairs = [(_CHOICE_ALPHABET[idx], text) for idx, text in enumerate(options)]
        if len(option_pairs) > len(_CHOICE_ALPHABET):
            raise ValueError("Multiple-choice preprocessor supports up to 26 options")

        answer_value = extract_field(sample, answer_field)
        correct_choice = resolve_correct_choice(answer_value, option_pairs, answer_index_base)
        if correct_choice is None:
            raise ValueError("Unable to resolve correct choice from answer field")

        user_prompt = _compose_prompt(question, option_pairs, instruction)
        messages: List[Dict[str, Any]] = []
        if system_prompt:
            messages.append(
                {
                    "role": "system",
                    "content": [{"type": "text", "text": system_prompt.strip()}],
                }
            )
        messages.append(
            {
                "role": "user",
                "content": [{"type": "text", "text": user_prompt}],
            }
        )

        sample["messages"] = messages
        sample["prompt"] = user_prompt
        set_render_flags(sample, mode="preprocess", source="manual", rendered_by="preprocess", cache_suffix="-converted")
        sample["choices"] = [
            {
                "index": idx,
                "label": label,
                "message": {"role": "assistant", "content": [{"type": "text", "text": option_text}]},
            }
            for idx, (label, option_text) in enumerate(option_pairs)
        ]
        metadata = dict(sample.get("metadata") or {})
        metadata.update(
            {
                "option_map": {label: option_text for label, option_text in option_pairs},
                "correct_choice": correct_choice,
                "question_field": question_field,
                "choices_field": option_source,
                "answer_field": answer_field,
            }
        )
        sample["metadata"] = metadata
        sample["inputs"] = sample.get("inputs") or {"prompt": user_prompt}
        return sample


def _compose_prompt(question: Any, option_pairs: Sequence[Tuple[str, str]], instruction: str | None) -> str:
    question_block = str(question).strip()
    option_lines = "\n".join(f"{label}. {text}" for label, text in option_pairs)
    prompt_parts = [question_block, "", option_lines]
    if instruction:
        prompt_parts.extend(["", instruction.strip()])
    return "\n".join(part for part in prompt_parts if part is not None).strip()


__all__ = ["MultiChoicePreprocessor"]
