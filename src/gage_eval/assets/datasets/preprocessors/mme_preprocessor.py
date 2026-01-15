"""MME (Multimodal Evaluation) preprocessors built on BasePreprocessor with image handling."""

from __future__ import annotations

from typing import Any, Dict, List, Optional
from collections import defaultdict

from gage_eval.assets.datasets.preprocessors.base import BasePreprocessor
from gage_eval.assets.datasets.utils.multimodal import (
    embed_local_image_as_data_url,
    encode_pil_to_data_url,
    merge_multimodal_inputs,
    resolve_media_path,
)
from gage_eval.assets.datasets.utils.rendering import set_render_flags
from gage_eval.assets.datasets.sample import (
    SCHEMA_VERSION,
    Sample,
    Message,
    MessageContent,
    sample_from_dict,
)
from gage_eval.assets.datasets.validation import validate_sample_schema
from loguru import logger


class MMEPreprocessor(BasePreprocessor):
    """Preprocess MME records into a standardized multimodal Sample.
    
    MME dataset: one image (question_id) corresponds to two questions.
    This preprocessor groups two questions with the same question_id into one sample,
    matching the reference implementation in calculation.py.
    
    MME dataset fields:
    - question_id: Unique identifier for the image (two questions share the same question_id)
    - image: Image path or PIL Image object
    - question: Question text
    - answer: Answer ("Yes" or "No")
    - category: Category of the question (e.g., "code_reasoning", "artwork")
    """

    def __init__(self, **kwargs: Any):
        super().__init__(**kwargs)
        # Track question index for each question_id to add q1/q2 suffix
        self._question_index: Dict[str, int] = {}

    def to_sample(
        self,
        record: Dict[str, Any],
        *,
        content_root: str | None = None,
        system_prompt: str | None = None,
        strict_image: bool = False,
        pre_encode_images: bool = True,
        schema_version = SCHEMA_VERSION,
        **kwargs: Any,
    ) -> Sample:
        """Process a single question record (each question is a separate sample)."""
        try:
            metadata = {}
            sample = dict(record)
            
            # Extract single question and answer
            question = sample.get("question")
            if question is None:
                raise ValueError("MME sample missing 'question'")
            
            answer = sample.get("answer")
            if answer is None:
                answer = sample.get("label")
            if answer is None:
                raise ValueError("MME sample missing 'answer'")
            
            # Normalize answer to Yes/No
            answer = self._normalize_answer(answer)
            question_text = str(question).strip()
            references = [answer]
            label = answer
            
            category = sample.get("category")
            if category:
                metadata["category"] = category
            
            question_id = sample.get("question_id")
            if question_id:
                metadata["question_id"] = question_id
            
            if content_root:
                metadata["content_root"] = content_root
        except Exception as e:
            logger.warning(f"failed, {e}")
            raise

        # STEP 2: Resolve image fragments (prefer PIL -> data URL).
        img_frag = None
        pil_obj = sample.get("decoded_image")
        if pil_obj is not None:
            try:
                url = encode_pil_to_data_url(pil_obj)
                img_frag = MessageContent(**{"type": "image_url", "image_url": {"url": url}})
                # Do not store image_url in metadata (following mathvista format)
            except Exception:
                if strict_image:
                    raise
        if img_frag is None and sample.get("image"):
            if pre_encode_images:
                embed_local_image_as_data_url(sample, image_field="image", content_root=content_root, strict=False)
                if isinstance(sample.get("image"), str):
                    img_frag = MessageContent(**{"type": "image_url", "image_url": {"url": sample["image"]}})
                    # Do not store image_url in metadata (following mathvista format)
            else:
                resolved = resolve_media_path(sample.get("image"), root=content_root)
                if resolved:
                    img_frag = MessageContent(**{"type": "image_url", "image_url": {"url": resolved}})
                    # Do not store image_url in metadata (following mathvista format)

        # STEP 3: Build `messages` with question and image.
        content: List[MessageContent] = [MessageContent(**{"type": "text", "text": question_text})]
        if img_frag:
            content.append(img_frag)
        
        messages: List[Message] = []
        if system_prompt:
            messages.append(Message(**{"role": "system", "content": [MessageContent(**{"type": "text", "text": system_prompt.strip()})]}))
        messages.append(Message(**{"role": "user", "content": content}))

        # STEP 4: Generate sample ID using question_id + q1/q2 suffix
        # Each question needs a unique ID, but we keep question_id in metadata for grouping
        question_id = sample.get("question_id")
        if question_id:
            metadata["question_id"] = question_id
            question_id_str = str(question_id)
            # Track question index for this question_id (1 or 2)
            if question_id_str not in self._question_index:
                self._question_index[question_id_str] = 0
            self._question_index[question_id_str] += 1
            question_index = self._question_index[question_id_str]
            # Create unique ID: question_id + q1 or q2 suffix
            unique_id = f"{question_id}:q{question_index}"
        else:
            # Fallback: use hash if no question_id available
            unique_id = sample.get("id") or str(hash(question_text))
        
        # STEP 5: Return Sample with single question's answer
        ret_sample = Sample(
            id=unique_id,
            schema_version=schema_version,
            options=[],
            messages=messages,
            references=references,  # Contains single answer
            label=label,
            metadata=metadata
        )
        return ret_sample

    def _normalize_answer(self, answer: Any) -> str:
        """Normalize answer to Yes/No."""
        if answer is None:
            return ""
        answer_str = str(answer).strip()
        if answer_str.lower() in ("yes", "y", "1", "true"):
            return "Yes"
        elif answer_str.lower() in ("no", "n", "0", "false"):
            return "No"
        else:
            return answer_str


__all__ = ["MMEPreprocessor"]
