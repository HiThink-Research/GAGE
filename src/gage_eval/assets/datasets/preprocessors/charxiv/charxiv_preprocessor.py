"""CharXiv reasoning preprocessor.

Maps HuggingFace `princeton-nlp/CharXiv` records into the unified `Sample` schema.
We currently focus on the **reasoning** questions:
- `reasoning_q`  -> user question
- `reasoning_a`  -> ground-truth answer (label + references)
"""

from __future__ import annotations

from typing import Any, Dict, List
import re

from gage_eval.assets.datasets.preprocessors.base import BasePreprocessor
from gage_eval.assets.datasets.utils.multimodal import (
    embed_local_image_as_data_url,
    encode_pil_to_data_url,
    resolve_media_path,
)
from gage_eval.assets.datasets.sample import (
    SCHEMA_VERSION,
    Sample,
    Message,
    MessageContent,
)
from gage_eval.assets.datasets.validation import validate_sample_schema
from loguru import logger


class CharXivReasoningPreprocessor(BasePreprocessor):
    """Preprocess CharXiv reasoning records into standardized multimodal Samples.

    Expected HF fields (validation split of `princeton-nlp/CharXiv`):
      - image: PIL image
      - category, year, original_figure_path, original_id, figure_path
      - num_subplots, subplot_row, subplot_col, subplot_loc
      - descriptive_q{1..4}, descriptive_a{1..4}
      - reasoning_q, reasoning_q_source, reasoning_a, reasoning_a_type
    """

    def __init__(self, **kwargs: Any):
        super().__init__(**kwargs)

    def to_sample(
        self,
        record: Dict[str, Any],
        *,
        content_root: str | None = None,
        system_prompt: str | None = None,
        strict_image: bool = False,
        pre_encode_images: bool = True,
        schema_version: str = SCHEMA_VERSION,
        **_: Any,
    ) -> Sample:
        """Convert a single CharXiv reasoning record into a `Sample`."""
        try:
            metadata: Dict[str, Any] = {}

            # ---- 1) Extract reasoning question & answer ----
            question = record.get("reasoning_q")
            if not question:
                raise ValueError("CharXiv record missing 'reasoning_q'")
            answer = record.get("reasoning_a")
            if answer is None:
                raise ValueError("CharXiv record missing 'reasoning_a' for reasoning split")

            question_text = str(question).strip()
            raw_label = str(answer).strip()

            # Keep the raw annotated answer as label (e.g., "(b) OPT"),
            # and also store a normalized core answer (without option letter)
            # so custom metrics can treat "(b)", "OPT", "(b) OPT" as一致答案.
            label = raw_label
            core_answer = raw_label
            m = re.match(r"^\(?[A-Za-z]\)?\s*(.+)$", raw_label)
            if m:
                core_answer = m.group(1).strip()
            metadata["core_answer"] = core_answer

            references: List[str] = [label]

            # ---- 2) Metadata passthrough ----
            meta_keys = [
                "category",
                "year",
                "original_figure_path",
                "original_id",
                "figure_path",
                "num_subplots",
                "subplot_row",
                "subplot_col",
                "subplot_loc",
                "reasoning_q_source",
                "reasoning_a_type",
            ]
            for k in meta_keys:
                if k in record and record[k] is not None:
                    metadata[k] = record[k]

            if content_root:
                metadata["content_root"] = content_root

        except Exception as e:
            logger.warning(f"CharXiv reasoning preprocessing failed: {e}")
            raise

        # ---- 3) Build multimodal `messages` (text + optional image) ----
        img_frag = None
        pil_obj = record.get("decoded_image")
        if pil_obj is not None:
            try:
                url = encode_pil_to_data_url(pil_obj)
                img_frag = MessageContent(
                    **{"type": "image_url", "image_url": {"url": url}}
                )
            except Exception:
                if strict_image:
                    raise

        if img_frag is None and record.get("image"):
            # `image` can be a path or PIL image; we reuse the helper to encode when possible.
            if pre_encode_images:
                # embed_local_image_as_data_url handles string paths; for PIL image we fall back to encode in-place
                embed_local_image_as_data_url(
                    record, image_field="image", content_root=content_root, strict=False
                )
                if isinstance(record.get("image"), str):
                    img_frag = MessageContent(
                        **{"type": "image_url", "image_url": {"url": record["image"]}}
                    )
            else:
                resolved = resolve_media_path(record.get("image"), root=content_root)
                if resolved:
                    img_frag = MessageContent(
                        **{"type": "image_url", "image_url": {"url": resolved}}
                    )

        # If caller did not provide a system prompt, use a default instruction
        # that forces the model to output only the final answer, so that
        # strict `exact_match` metrics can be used reliably.
        if system_prompt is None:
            system_prompt = (
                "You are answering chart-based reasoning questions. "
                "Read the user's question (and image if provided), then reply with "
                "ONLY the final short answer text. Do NOT include explanations, "
                "sentences, or any extra words."
            )

        # User message: reasoning question text (+ image if available)
        content: List[MessageContent] = [
            MessageContent(**{"type": "text", "text": question_text})
        ]
        if img_frag:
            content.append(img_frag)

        messages: List[Message] = []
        if system_prompt:
            messages.append(
                Message(
                    **{
                        "role": "system",
                        "content": [
                            MessageContent(
                                **{
                                    "type": "text",
                                    "text": system_prompt.strip(),
                                }
                            )
                        ],
                    }
                )
            )
        messages.append(Message(**{"role": "user", "content": content}))

        # ---- 4) Sample ID ----
        original_id = record.get("original_id")
        subplot_row = record.get("subplot_row")
        subplot_col = record.get("subplot_col")
        if original_id is not None and subplot_row is not None and subplot_col is not None:
            sample_id = f"{original_id}_r{subplot_row}_c{subplot_col}"
        else:
            # Fallback to hash of question + figure_path
            base = f"{record.get('figure_path', '')}::{question_text}"
            sample_id = str(hash(base))

        # ---- 5) Build Sample ----
        ret_sample = Sample(
            id=sample_id,
            schema_version=schema_version,
            options=[],
            messages=messages,
            references=references,
            label=label,
            metadata=metadata,
        )

        # Validate for early detection of schema issues
        validate_sample_schema(ret_sample)
        return ret_sample


__all__ = ["CharXivReasoningPreprocessor"]

