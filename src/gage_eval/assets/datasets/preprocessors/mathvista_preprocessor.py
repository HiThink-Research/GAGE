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

_CHOICE_ALPHABET: Tuple[str, ...] = tuple("ABCDEFGHIJKLMNOPQRSTUVWXYZ")


class MathVistaPreprocessor(BasePreprocessor):
    """MathVista 多模态预处理器：题干 + 图片 + 可选多选项."""

    def to_sample(
        self,
        record: Dict[str, Any],
        *,
        content_root: str | None = None,
        system_prompt: str | None = None,
        strict_image: bool = False,
        pre_encode_images: bool = True,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        # step1: 确定题干/答案/内容字段，并补齐路径上下文。
        sample = dict(record)
        question = sample.get("question")
        if question is None:
            raise ValueError("MathVista sample missing 'question'")
        choices = normalize_options(sample.get("choices") or [])
        answer = sample.get("answer")

        if content_root:
            sample.setdefault("_dataset_metadata", {})["path"] = content_root

        # step2: 图片处理，优先 PIL -> data URL
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
        content: List[Dict[str, Any]] = [{"type": "text", "text": str(question).strip()}]
        # step3.2: 追加图像片段（若存在）
        if img_frag:
            content.append(img_frag)
        # step3.3: 组织 system/user 消息
        messages: List[Dict[str, Any]] = []
        if system_prompt:
            messages.append({"role": "system", "content": [{"type": "text", "text": system_prompt.strip()}]})
        messages.append({"role": "user", "content": content})
        sample["messages"] = messages
        sample["prompt"] = sample.get("prompt") or str(question)

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
        sample["inputs"] = sample.get("inputs") or {"prompt": sample["prompt"]}
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


__all__ = ["MathVistaPreprocessor"]


class MathVistaStructOnlyPreprocessor(MathVistaPreprocessor):
    """MathVista 结构化预处理（不拼接 Prompt，保留 inputs/multimodal/metadata）。"""

    def to_sample(
        self,
        record: Dict[str, Any],
        *,
        content_root: str | None = None,
        system_prompt: str | None = None,
        strict_image: bool = False,
        pre_encode_images: bool = True,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        sample = super().to_sample(
            record,
            content_root=content_root,
            system_prompt=system_prompt,
            strict_image=strict_image,
            pre_encode_images=pre_encode_images,
            **kwargs,
        )
        # 仅保留多模态/choices/metadata，移除渲染产物，让外置 Prompt 渲染
        sample.pop("prompt", None)
        sample["messages"] = []
        sample["inputs"] = sample.get("inputs") or {}
        strip_render_flags(sample)
        return sample


__all__ = ["MathVistaPreprocessor", "MathVistaStructOnlyPreprocessor"]
