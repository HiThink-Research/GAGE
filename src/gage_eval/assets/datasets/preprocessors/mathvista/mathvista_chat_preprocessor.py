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

from gage_eval.assets.datasets.preprocessors.mathvista.this import create_prompt

from loguru import logger
from dataclasses import dataclass, asdict, is_dataclass

import os

from gage_eval.assets.datasets.sample import (
    SCHEMA_VERSION,
    Sample,
    Message,
    MessageContent
)

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
        schema_version = SCHEMA_VERSION,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        try:
            references = []
            metadata = {}
            # STEP 1: build a prompt.  
            sample = dict(record)   
            prompt = create_prompt(sample, use_caption, use_ocr, shot_num, shot_type)
            choices = normalize_options(sample.get("choices") or [])
            answer = sample.get("answer")
            answer_type = sample.get("answer_type")            
            if answer_type is None and answer not in (None, ""):
                try:
                    # Best-effort heuristic: treat integers as a special case.
                    float_val = float(str(answer).strip())
                    answer_type = "integer" if float_val.is_integer() else "float"
                except Exception:
                    answer_type = None
            metadata["question_type"] = sample.get("question_type")
            is_multi_choice =  metadata["question_type"] == 'multi_choice'                    
            metadata["answer_type"] = answer_type
            metadata["shot_type"] = shot_type
            metadata['use_caption'] = use_caption
            metadata['use_ocr'] = use_ocr
        except Exception as e:
            logger.warning(f"failed, {e}")

        # STEP 2: Normalize multiple-choice options (if present).
        if is_multi_choice:
            if len(choices) > len(_CHOICE_ALPHABET):
                raise ValueError("MathVista multiple-choice supports up to 26 options")
            option_pairs = list(zip(_CHOICE_ALPHABET, choices))
            answer = resolve_correct_choice(str(answer), option_pairs, answer_index_base=0)

        # STEP 3: Resolve image fragments (prefer PIL -> data URL).
        img_frag = None
        pil_obj = sample.get("decoded_image")
        if pil_obj is not None:
            try:
                # Encode the PIL object as a data URL so HTTP backends can consume it.
                if pre_encode_images:
                    url = encode_pil_to_data_url(pil_obj)
                    img_frag = MessageContent(**{"type": "image_url", "image_url": {"url": url}})
                else:
                    # If there is no on-disk path, we cannot defer encoding safely.
                    url = encode_pil_to_data_url(pil_obj)
                    img_frag = MessageContent(**{"type": "image_url", "image_url": {"url": url}})
            except Exception:
                if strict_image:
                    raise
        if img_frag is None and sample.get("image"):
            # Fallback: local path -> data URL (supports relative paths). If `pre_encode_images=False`,
            # keep the resolved path and let the backend decide whether to embed/encode.
            if pre_encode_images:
                embed_local_image_as_data_url(sample, image_field="image", content_root=content_root, strict=False)
                if isinstance(sample.get("image"), str):
                    img_frag = MessageContent(**{"type": "image_url", "image_url": {"url": sample["image"]}})
            else:
                resolved = resolve_media_path(sample.get("image"), root=content_root)
                if resolved:
                    sample["image"] = resolved
                    img_frag = MessageContent(**{"type": "image_url", "image_url": {"url": resolved}})
        if content_root:
            metadata["content_root"] = content_root

        # STEP 4: Build `messages` (prompt text + optional image fragments).
        content: List[Dict[str, Any]] = [MessageContent(**{"type": "text", "text": str(prompt).strip()})]
        if img_frag:
            content.append(img_frag)
        messages: List[Dict[str, Any]] = []
        if system_prompt:
            messages.append(Message(**{"role": "system", "content": [MessageContent(**{"type": "text", "text": system_prompt.strip()})]}))
        messages.append(Message(**{"role": "user", "content": content}))
        ret_sample = Sample(
            id = sample.get("pid"),
            schema_version = schema_version,
            options = choices,
            messages = messages,
            references = [answer],
            label=answer,
            metadata = metadata            
        )
        return ret_sample


__all__ = ["MathVistaChatPreprocessor"]

from PIL import Image
import random
import numpy as np

def generate_random_image_v2(width, height):
    # 生成 [height, width, 3] 的随机 uint8 数组
    data = np.random.randint(0, 256, (height, width, 3), dtype=np.uint8)
    img = Image.fromarray(data, 'RGB')
    return img

if __name__ == '__main__':
    pid = "1"
    question = "how many dogs in this image?"
    decoded_image =generate_random_image_v2(32, 32)
    answer = "1"
    question_type = "free_form"
    answer_type = "integer"    
    sample = {
            "unit": 'g',
            "image": "fake.jpg",
            "precision": 1,
            "choices": None,
            "query": "test",
            "pid":pid,
            "question": question,
            "decoded_image": decoded_image,
            "answer": answer,
            "question_type": question_type,
            "answer_type": answer_type,
            "caption": None,
            "ocr": None
        }
    pre = MathVistaChatPreprocessor()        
    ret = pre.to_sample(sample)
