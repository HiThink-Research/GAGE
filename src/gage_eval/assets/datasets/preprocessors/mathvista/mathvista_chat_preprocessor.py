"""MathVista preprocessors built on BasePreprocessor with image handling."""

from __future__ import annotations

from typing import Any, Dict, List, Tuple

from gage_eval.assets.datasets.preprocessors.base import BasePreprocessor
from gage_eval.assets.datasets.utils.multimodal import (
    embed_local_image_as_data_url,
    encode_pil_to_data_url,
    merge_multimodal_inputs,
    resolve_media_path,
)
from gage_eval.assets.datasets.utils.mapping import normalize_options, resolve_correct_choice
from gage_eval.assets.datasets.utils.rendering import set_render_flags, strip_render_flags

from gage_eval.assets.datasets.preprocessors.common.utils import read_json
from gage_eval.assets.datasets.preprocessors.mathvista.this import create_prompt

from loguru import logger as logging

import os

_CHOICE_ALPHABET: Tuple[str, ...] = tuple("ABCDEFGHIJKLMNOPQRSTUVWXYZ")


class MathVistaChatPreprocessor(BasePreprocessor):
    """Preprocess MathVista records into a chat-style multimodal Sample."""

    def to_sample(
        self,
        record: Dict[str, Any],
        *,
        use_caption: bool = False,
        use_ocr: bool = False,
        shot_num: int = 0,
        shot_type: str = 'solution',        
        content_root: str | None = None,
        system_prompt: str | None = None,
        strict_image: bool = False,
        pre_encode_images: bool = True,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        try:
            # STEP 1: Load auxiliary text (caption/OCR) and build a prompt.
            caption_data = {}
            if use_caption:
                current_dir_path = os.path.dirname(os.path.abspath(__file__))
                rel_cap_filename = 'data/texts/captions_bard.json'
                full_path = os.path.join(current_dir_path, rel_cap_filename)
                if os.path.exists(full_path):
                    caption_data = read_json(full_path)["texts"]
            if use_ocr:
                current_dir_path = os.path.dirname(os.path.abspath(__file__))
                rel_ocr_filename = 'data/texts/ocrs_easyocr.json'
                ocr_file = os.path.join(current_dir_path, rel_ocr_filename)
                if os.path.exists(ocr_file):
                    ocr_data = read_json(ocr_file)["texts"]     
            
            sample = dict(record)    
            pid = sample.get("pid")
            sample['caption'] = caption_data.get('pid')
            sample['ocr'] = ocr_data.get('pid')

            prompt = create_prompt(sample, use_caption, use_ocr, shot_num, shot_type)

            choices = normalize_options(sample.get("choices") or [])
            answer = sample.get("answer")
            if answer in (None, ""):
                # NOTE: Fall back to common answer fields used by different dataset variants.
                answer = (
                    sample.get("label")
                    or (sample.get("metadata") or {}).get("answer")
                    or sample.get("final_answer")
                    or sample.get("answer_text")
                )
            if answer is not None:
                sample["answer"] = answer
            # Try to infer `answer_type` when missing (useful for numeric QA).
            answer_type = sample.get("answer_type") or (sample.get("metadata") or {}).get("answer_type")
            if answer_type is None and answer not in (None, ""):
                try:
                    # Best-effort heuristic: treat integers as a special case.
                    float_val = float(str(answer).strip())
                    answer_type = "integer" if float_val.is_integer() else "float"
                except Exception:
                    answer_type = None
            if answer_type:
                sample["answer_type"] = answer_type 
        
            sample['shot_type'] = shot_type
            if content_root:
                sample.setdefault("_dataset_metadata", {})["path"] = content_root

        except Exception as e:
            logging.error("failed, {}", e)
            exit(0)

        # STEP 2: Resolve image fragments (prefer PIL -> data URL).
        img_frag = None
        pil_obj = sample.get("decoded_image")
        if pil_obj is not None:
            try:
                # Encode the PIL object as a data URL so HTTP backends can consume it.
                if pre_encode_images:
                    url = encode_pil_to_data_url(pil_obj)
                    img_frag = {"type": "image_url", "image_url": {"url": url}}
                    sample.setdefault("metadata", {})["image_url"] = url
                else:
                    # If there is no on-disk path, we cannot defer encoding safely.
                    url = encode_pil_to_data_url(pil_obj)
                    img_frag = {"type": "image_url", "image_url": {"url": url}}
                    sample.setdefault("metadata", {})["image_url"] = url
            except Exception:
                if strict_image:
                    raise
        if img_frag is None and sample.get("image"):
            # Fallback: local path -> data URL (supports relative paths). If `pre_encode_images=False`,
            # keep the resolved path and let the backend decide whether to embed/encode.
            if pre_encode_images:
                embed_local_image_as_data_url(sample, image_field="image", content_root=content_root, strict=False)
                if isinstance(sample.get("image"), str):
                    img_frag = {"type": "image_url", "image_url": {"url": sample["image"]}}
                    sample.setdefault("metadata", {})["image_url"] = sample["image"]
            else:
                resolved = resolve_media_path(sample.get("image"), root=content_root)
                if resolved:
                    sample["image"] = resolved
                    img_frag = {"type": "image_url", "image_url": {"url": resolved}}
                    sample.setdefault("metadata", {})["image_url"] = resolved
        if content_root:
            meta = sample.get("metadata") or {}
            meta["content_root"] = content_root
            sample["metadata"] = meta

        # STEP 3: Build `messages` (prompt text + optional image fragments).
        content: List[Dict[str, Any]] = [{"type": "text", "text": str(prompt).strip()}]
        if img_frag:
            content.append(img_frag)
        messages: List[Dict[str, Any]] = []
        if system_prompt:
            messages.append({"role": "system", "content": [{"type": "text", "text": system_prompt.strip()}]})
        messages.append({"role": "user", "content": content})

        sample["messages"] = messages

        # STEP 4: Normalize multiple-choice options (if present).
        if choices:
            if len(choices) > len(_CHOICE_ALPHABET):
                raise ValueError("MathVista multiple-choice supports up to 26 options")
            option_pairs = list(zip(_CHOICE_ALPHABET, choices))
            sample["choices"] = [
                {
                    "index": idx,
                    "label": label,
                    "message": {"role": "assistant", "content": [{"type": "text", "text": opt}]},
                }
                for idx, (label, opt) in enumerate(option_pairs)
            ]
            meta = sample.setdefault("metadata", {})
            meta["option_map"] = {l: opt for l, opt in option_pairs}
            meta["correct_choice"] = resolve_correct_choice(answer, option_pairs, answer_index_base=0)

        # STEP 5: Sync multimodal inputs and set render flags.
        # sample["inputs"] = sample.get("inputs") or {"prompt": sample["prompt"]}
        merge_multimodal_inputs(sample)
        set_render_flags(
            sample,
            mode="preprocess",
            source="manual",
            rendered_by="preprocess",
            cache_suffix="-converted",
            overwrite=False,
        )
        return sample


__all__ = ["MathVistaChatPreprocessor"]



