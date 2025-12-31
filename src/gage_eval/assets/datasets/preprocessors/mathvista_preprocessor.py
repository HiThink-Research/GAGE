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
    """Preprocess MathVista records into a standardized multimodal Sample."""

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
        # STEP 1: Resolve core fields and dataset context.
        sample = dict(record)
        question = sample.get("question")
        if question is None:
            raise ValueError("MathVista sample missing 'question'")
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

        if content_root:
            sample.setdefault("_dataset_metadata", {})["path"] = content_root

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

        # STEP 3: Build `messages` (text + optional image fragments).
        content: List[Dict[str, Any]] = [{"type": "text", "text": str(question).strip()}]
        if img_frag:
            content.append(img_frag)
            # Ensure the prompt contains the `<image>` placeholder for multimodal backends.
            question_with_image = f"<image>\\n{question}" if "<image>" not in str(question) else str(question)
            sample["prompt"] = sample.get("prompt") or question_with_image
        messages: List[Dict[str, Any]] = []
        if system_prompt:
            messages.append({"role": "system", "content": [{"type": "text", "text": system_prompt.strip()}]})
        messages.append({"role": "user", "content": content})
        sample["messages"] = messages
        sample["prompt"] = sample.get("prompt") or str(question)

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
        sample["inputs"] = sample.get("inputs") or {"prompt": sample["prompt"]}
        merge_multimodal_inputs(sample)
        set_render_flags(
            sample,
            mode="preprocess",
            source="manual",
            rendered_by="preprocess",
            cache_suffix="-converted",
            overwrite=False,
        )

        # STEP 6: Optionally tokenize the prompt (produces `prompt_token_ids` for token-based backends).
        tokenize_prompt = kwargs.get("tokenize_prompt") or self.kwargs.get("tokenize_prompt")
        tokenizer = kwargs.get("tokenizer") or self.kwargs.get("tokenizer")
        strict_tokenize = kwargs.get("strict_tokenize") or self.kwargs.get("strict_tokenize")
        if tokenize_prompt and tokenizer and hasattr(tokenizer, "apply_chat_template"):
            try:
                rendered_prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                tokenized = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True)
                token_ids = tokenized[0] if isinstance(tokenized, list) else tokenized
                if rendered_prompt:
                    sample["prompt"] = str(rendered_prompt)
                    sample["inputs"]["prompt"] = sample["prompt"]
                if token_ids is not None:
                    sample["inputs"]["prompt_token_ids"] = token_ids
                # Mark that a model chat template has been applied, so backends can skip re-rendering.
                set_render_flags(
                    sample,
                    mode="preprocess",
                    source="model",
                    rendered_by="preprocess",
                    cache_suffix="-chat_template",
                    overwrite=True,
                )
                tok_path = (
                    kwargs.get("tokenizer_path")
                    or kwargs.get("tokenizer_name")
                    or self.kwargs.get("tokenizer_path")
                    or self.kwargs.get("tokenizer_name")
                )
                if tok_path and "_tokenizer_path" not in sample:
                    sample["_tokenizer_path"] = tok_path
            except Exception as exc:
                if strict_tokenize:
                    raise
                sample.setdefault("metadata", {})["tokenize_warning"] = str(exc)
        return sample


__all__ = ["MathVistaPreprocessor"]


class MathVistaStructOnlyPreprocessor(MathVistaPreprocessor):
    """Preprocess MathVista into structured fields only (no prompt rendering)."""

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
        # Keep multimodal/choices/metadata only. Remove rendered outputs for external prompt rendering.
        sample.pop("prompt", None)
        sample["messages"] = []
        sample["inputs"] = sample.get("inputs") or {}
        strip_render_flags(sample)
        return sample


__all__ = ["MathVistaPreprocessor", "MathVistaStructOnlyPreprocessor"]
