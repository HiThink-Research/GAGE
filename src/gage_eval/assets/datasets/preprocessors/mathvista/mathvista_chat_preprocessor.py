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
    """MathVista 多模态预处理器：题干 + 图片 + 可选多选项."""

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
                # 回退常见字段：label/metadata.answer/final_answer/answer_text
                answer = (
                    sample.get("label")
                    or (sample.get("metadata") or {}).get("answer")
                    or sample.get("final_answer")
                    or sample.get("answer_text")
                )
            if answer is not None:
                sample["answer"] = answer
            # answer_type 尝试回填
            answer_type = sample.get("answer_type") or (sample.get("metadata") or {}).get("answer_type")
            if answer_type is None and answer not in (None, ""):
                try:
                    # 简单推断类型
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

        img_frag = None
        pil_obj = sample.get("decoded_image")
        if pil_obj is not None:
            try:
                # step2.1: PIL 对象编码为 data URL，方便 HTTP 后端直接消费
                if pre_encode_images:
                    url = encode_pil_to_data_url(pil_obj)
                    img_frag = {"type": "image_url", "image_url": {"url": url}}
                    sample.setdefault("metadata", {})["image_url"] = url
                else:
                    # 无路径时无法延后，仍回退为 data URL
                    url = encode_pil_to_data_url(pil_obj)
                    img_frag = {"type": "image_url", "image_url": {"url": url}}
                    sample.setdefault("metadata", {})["image_url"] = url
            except Exception:
                if strict_image:
                    raise
        if img_frag is None and sample.get("image"):
            # step2.2: 回退图片路径 -> data URL（支持相对路径）；如设置 pre_encode_images=False，则仅解析路径
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

        # step3: 消息拼装（文本 + 图像片段）
        # step3.1: 基础文本片段
        content: List[Dict[str, Any]] = [{"type": "text", "text": str(prompt).strip()}]
        # step3.2: 追加图像片段（若存在）
        if img_frag:
            content.append(img_frag)
        # step3.3: 组织 system/user 消息
        messages: List[Dict[str, Any]] = []
        if system_prompt:
            messages.append({"role": "system", "content": [{"type": "text", "text": system_prompt.strip()}]})
        messages.append({"role": "user", "content": content})

        sample["messages"] = messages

        # step4: 选项/答案归一化
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

        # step5: inputs/多模态对齐 + 标记
        # sample["inputs"] = sample.get("inputs") or {"prompt": sample["prompt"]}
        # step5.1: merge_multimodal_inputs 归并消息/inputs 的媒体引用并去重
        merge_multimodal_inputs(sample)
        # step5.2: set_render_flags 统一设置渲染标记/缓存后缀
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




