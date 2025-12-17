"""Class-based DocVQA preprocessor (new implementation)."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

from gage_eval.assets.datasets.utils.multimodal import (
    _derive_root,
    collect_content_fragments,
    embed_local_message_images,
)
from gage_eval.assets.datasets.preprocessors.base import BasePreprocessor
from gage_eval.assets.datasets.utils.mapping import extract_field
from gage_eval.assets.datasets.utils.normalization import list_images, ensure_chat_template_flags
from gage_eval.assets.datasets.utils.answers import parse_list_from_string, enrich_answer_with_options
from gage_eval.assets.datasets.utils.rendering import set_render_flags


class DocVQAPreprocessor(BasePreprocessor):
    """Normalize DocVQA samples with text + image content."""

    def to_sample(
        self,
        record: Dict[str, Any],
        *,
        question_field: str = "question",
        answers_field: str = "choices.0.message.content.0.text",
        content_field: str = "messages.0.content",
        content_root: str | None = None,
        data_path: str | None = None,
        system_prompt: str | None = None,
        instruction: str | None = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        sample = dict(record)
        question = extract_field(sample, question_field)
        if question is None:
            raise ValueError(f"DocVQA sample missing question field '{question_field}'")
        answers_raw = extract_field(sample, answers_field)
        answers = enrich_answer_with_options(question, parse_list_from_string(answers_raw))
        if not answers:
            raise ValueError("DocVQA sample must provide at least one reference answer")

        # 1. Resolve Root Path
        # Ensure dataset metadata path is set if data_path is provided, for _derive_root fallback
        if data_path and "_dataset_metadata" not in sample:
            sample["_dataset_metadata"] = {"path": data_path}
        elif data_path:
            sample.setdefault("_dataset_metadata", {})["path"] = data_path
            
        resolved_root = _derive_root(sample, content_root)
        if resolved_root and isinstance(resolved_root, str) and not resolved_root.startswith(("http://", "https://", "data:")):
            try:
                resolved_root = str(Path(resolved_root).expanduser().resolve())
            except Exception:
                resolved_root = str(Path(resolved_root).expanduser())

        # 2. Construct Content
        text_content = str(question).strip()
        if instruction:
            text_content = f"{text_content}\n\n{instruction.strip()}"
            
        user_content_parts = [{"type": "text", "text": text_content}]
        
        # Collect multimodal fragments
        fragments = collect_content_fragments(sample, content_field=content_field, content_root=resolved_root)
        visual_fragments = [
            frag for frag in fragments 
            if isinstance(frag, dict) and frag.get("type") == "image_url"
        ]
        
        if visual_fragments:
            user_content_parts.extend(visual_fragments)

        # 3. Build Messages
        messages: List[Dict[str, Any]] = []
        if system_prompt:
            messages.append({"role": "system", "content": [{"type": "text", "text": system_prompt.strip()}]})
        
        user_msg_index = len(messages) 
        messages.append({"role": "user", "content": user_content_parts})

        sample["messages"] = messages
        sample["prompt"] = question
        set_render_flags(sample, mode="preprocess", source="manual", rendered_by="preprocess", cache_suffix="-converted")

        # 5. Embed Local Images
        target_content_field = f"messages.{user_msg_index}.content"
        embed_local_message_images(
            sample,
            content_field=target_content_field,
            content_root=resolved_root,
            strict=False,
        )
        ensure_chat_template_flags(sample)

        # 5. Finalize Metadata
        final_images = list_images(sample)
        final_image_url = final_images[0] if final_images else None

        metadata = dict(sample.get("metadata") or {})
        metadata.pop("image_root", None)
        metadata.update({
            "answers": answers,
            "question_field": question_field,
            "answers_field": answers_field,
            "content_field": target_content_field,
        })
        if final_image_url:
            metadata["image_url"] = final_image_url
        if resolved_root:
            metadata["content_root"] = resolved_root

        sample["metadata"] = metadata
        sample["inputs"] = sample.get("inputs") or {"prompt": question}
        return sample


__all__ = ["DocVQAPreprocessor"]
