"""GPQA preprocessors built on MultiChoicePreprocessor."""

from __future__ import annotations

import random
from typing import Any, Dict

from gage_eval.assets.datasets.preprocessors.multi_choice_preprocessor import MultiChoicePreprocessor
from gage_eval.assets.datasets.utils.rendering import strip_render_flags

from gage_eval.assets.datasets.preprocessors.gpqa.utils import (
    create_prompts,
    Example,
    load_examples,
)

from loguru import logger

from gage_eval.assets.datasets.utils.mapping import (
    extract_field,
    normalize_options,
    resolve_correct_choice,
)
from gage_eval.assets.datasets.utils.rendering import set_render_flags

from gage_eval.assets.datasets.sample import (
    SCHEMA_VERSION,
    Sample,
    Message,
    MessageContent
)

_CHOICE_ALPHABET: Tuple[str, ...] = tuple("ABCDEFGHIJKLMNOPQRSTUVWXYZ")


class GpqaDiamondPreprocessor(MultiChoicePreprocessor):
    """Preprocesses GPQA (Diamond) multiple-choice records into the Sample schema.

    The implementation follows a straightforward pipeline:
    1) Load and render the prompt for the given `gpqa_prompt_type`.
    2) Collect the four options and resolve the correct option index.
    3) Build `messages` / `inputs` and attach standardized `choices`.
    4) Populate metadata for downstream metrics and debugging.
    """

    def to_sample(self, record: Dict[str, Any], gpqa_prompt_type, schema_version = SCHEMA_VERSION, **kwargs: Any) -> Dict[str, Any]:
        """Converts a raw GPQA record into a standardized multiple-choice Sample.

        Args:
            record: Raw dataset record (typically a dict emitted by the loader).
            gpqa_prompt_type: Prompt rendering mode passed through to the GPQA utilities.
            **kwargs: Reserved for forward compatibility.

        Returns:
            A dict compatible with the gage-eval Sample schema.
        """
        sample: Dict[str, Any] = dict(record)
        try:
            # STEP 1: Render a single user prompt and collect the option set.
            correct_answer = record.get("Correct Answer")
            examples = load_examples([sample])
            prompts, examples = create_prompts(examples, gpqa_prompt_type)
            assert len(prompts) == 1, "len(prompts) != 1"
            assert len(examples) == 1, "len(examples) != 1"
            user_prompt = prompts[0]
            example = examples[0]

            # STEP 2: Resolve the correct answer index and validate option constraints.
            options = [example.choice1, example.choice2, example.choice3, example.choice4]
            answer_index = options.index(correct_answer)
            message_content = MessageContent(type="text", text=user_prompt)

            options = normalize_options(options)
            if len(options) < 2:
                raise ValueError("Multiple-choice sample must provide at least two options")
            option_pairs = [(_CHOICE_ALPHABET[idx], text) for idx, text in enumerate(options)]
            if len(option_pairs) > len(_CHOICE_ALPHABET):
                raise ValueError("Multiple-choice preprocessor supports up to 26 options")                            
            correct_choice = resolve_correct_choice(answer_index, option_pairs, answer_index_base=0)
            if correct_choice is None:
                raise ValueError("Unable to resolve correct choice from answer field")

            # STEP 3: Build canonical chat messages
            message = Message(role='user', content=[message_content])
            sample_id = str(hash('\n'.join([user_prompt, correct_choice])))
            ret_sample = Sample(
                id = sample_id,
                schema_version = schema_version,
                messages = [message],
                options = options,
                references = [correct_choice],
                label=correct_choice
            )
        except Exception as e:
            logger.error("gpqa_diamond_preprocessor failed: {}", e)
        return ret_sample




__all__ = ["GpqaDiamondPreprocessor"]
