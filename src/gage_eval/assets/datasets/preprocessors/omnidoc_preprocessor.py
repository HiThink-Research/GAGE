"""Class-based DocVQA preprocessor (new implementation)."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List
import base64
import mimetypes

from gage_eval.assets.datasets.utils.multimodal import (
    _derive_root,
    collect_content_fragments,
    embed_local_message_images,
    embed_local_image_as_data_url
)
from gage_eval.assets.datasets.preprocessors.base import BasePreprocessor
from gage_eval.assets.datasets.utils.mapping import extract_field
from gage_eval.assets.datasets.utils.normalization import list_images, ensure_chat_template_flags
from gage_eval.assets.datasets.utils.answers import parse_list_from_string, enrich_answer_with_options
from gage_eval.assets.datasets.utils.rendering import set_render_flags

def encode_image_to_data_uri(image_path):
    mime_type, _ = mimetypes.guess_type(image_path)
    if mime_type is None:
        mime_type = 'image/jpeg'
    
    with open(image_path, "rb") as image_file:
        base64_data = base64.b64encode(image_file.read()).decode('utf-8')
    
    return f"data:{mime_type};base64,{base64_data}"

class OmniDocPreprocessor(BasePreprocessor):
    """Normalize OminiDoc samples with text + image content."""

    def to_sample(
        self,
        record: Dict[str, Any],
        *,
        question_field: str = "question",
        content_field: str = "image",
        content_root: str | None = None,
        data_path: str | None = None,
        system_prompt: str | None = None,
        instruction: str | None = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        
        sample = dict(record)
        question = extract_field(sample, question_field)
        if question is None:
            raise ValueError(f"OminiDoc sample missing question field '{question_field}'")

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
        
        # 3. Embed Local Images
        fragments = collect_content_fragments(sample, content_field=content_field, content_root=resolved_root)

        converted = embed_local_image_as_data_url(
            sample,
            image_field=content_field,
            strict=False,
            cache_dir=None,
            content_root=content_root,
        )
        
        # 4. Build Messages
        messages: List[Dict[str, Any]] = []
        if system_prompt:
            messages.append({"role": "system", "content": [{"type": "text", "text": system_prompt.strip()}]})
        
        visual_fragments=[{"type": "image_url", "image_url": sample['image']}]
        user_content_parts.extend(visual_fragments)
        messages.append({"role": "user", "content": user_content_parts})

        sample["messages"] = messages
        sample["prompt"] = question
        sample["chat_template_mode"] = "preprocess"
        sample["rendered_by"] = "preprocess"
        sample["template_source"] = "manual"
        sample["cache_suffix"] = "-converted"

        ensure_chat_template_flags(sample)

        # 5. Finalize Metadata
        final_image_name = fragments
        metadata = dict(sample.get("metadata") or {})
        metadata.pop("image_root", None)
        metadata.update({
            "question_field": question_field,
            "content_field": content_field,
        })
        if final_image_name:
            metadata["image_name"] = final_image_name
        if resolved_root:
            metadata["content_root"] = resolved_root
        
        sample["metadata"] = metadata
        sample["inputs"] = sample.get("inputs") or {"prompt": question}
        return sample


__all__ = ["OmniDocPreprocessor"]
