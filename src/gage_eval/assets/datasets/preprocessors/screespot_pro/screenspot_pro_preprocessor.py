"""ScreenSpot-Pro preprocessor for GUI grounding tasks."""

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

# ScreenSpot-Pro class labels (26 classes representing different professional applications)
SCREENSPOT_CLASSES = [
    "android_studio_mac", "autocad_windows", "blender_windows", "davinci_resolve_macos",
    "eviews_windows", "excel_macos", "fl_studio_windows", "illustrator_windows",
    "inventor_windows", "linux_ubuntu", "macos_sonoma", "matlab_windows",
    "origin_windows", "photoshop_windows", "powerpoint_windows", "premiere_windows",
    "pycharm_macos", "quartus_windows", "solidworks_windows", "stata_windows",
    "unreal_engine_windows", "vivado_windows", "visual_studio_code_macos",
    "vmware_fusion_macos", "windows_11", "word_macos"
]


class ScreenSpotProPreprocessor(BasePreprocessor):
    """Preprocess ScreenSpot-Pro records into a standardized multimodal Sample.

    ScreenSpot-Pro dataset: GUI grounding for professional high-resolution computer use.
    Each sample contains an image of a professional application interface and a class label
    indicating which application the interface belongs to.

    Dataset fields:
    - image: Image path or PIL Image object
    - label: Class index (0-25) corresponding to different professional applications
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
        schema_version = SCHEMA_VERSION,
        **kwargs: Any,
    ) -> Sample:
        """Process a single ScreenSpot-Pro record."""
        try:
            metadata = {}

            # Extract label
            label_idx = record.get("label")
            # `likaixin/ScreenSpot-Pro` uses an int label (0-25).
            # `lmms-lab/ScreenSpot-Pro` is a GUI-grounding dataset and uses string fields like `application`
            # (plus bbox/img_size/instruction). Support both.
            if label_idx is None:
                app = record.get("application")
                if app is None:
                    raise ValueError("ScreenSpot-Pro sample missing required 'label' or 'application'")
                class_name = str(app)
                references = [class_name]
                label = class_name
                metadata["application"] = class_name
            else:
                if not isinstance(label_idx, int) or not (0 <= label_idx < len(SCREENSPOT_CLASSES)):
                    raise ValueError(f"Invalid label index: {label_idx}, must be 0-{len(SCREENSPOT_CLASSES)-1}")
                # Get class name and prepare answer
                class_name = SCREENSPOT_CLASSES[label_idx]
                references = [class_name]
                label = class_name

            # Add metadata
            if label_idx is not None:
                metadata["label_index"] = label_idx
            metadata["class_name"] = class_name
            # If available, pass through GUI-grounding fields so metrics can use them.
            # Note: the HuggingFace dataset `likaixin/ScreenSpot-Pro` only has (image, label).
            # Other ScreenSpot variants may provide these fields.
            if "bbox" in record:
                metadata["bbox"] = record.get("bbox")
            if "img_size" in record:
                metadata["img_size"] = record.get("img_size")
            if "platform" in record:
                metadata["platform"] = record.get("platform")
            if "group" in record:
                metadata["group"] = record.get("group")
            if "ui_type" in record:
                metadata["ui_type"] = record.get("ui_type")
            if content_root:
                metadata["content_root"] = content_root

        except Exception as e:
            logger.warning(f"ScreenSpot-Pro preprocessing failed: {e}")
            raise

        # STEP 2: Resolve image fragments (prefer PIL -> data URL).
        img_frag = None
        pil_obj = record.get("decoded_image")
        if pil_obj is not None:
            try:
                url = encode_pil_to_data_url(pil_obj)
                img_frag = MessageContent(**{"type": "image_url", "image_url": {"url": url}})
            except Exception:
                if strict_image:
                    raise
        if img_frag is None and record.get("image"):
            if pre_encode_images:
                embed_local_image_as_data_url(record, image_field="image", content_root=content_root, strict=False)
                if isinstance(record.get("image"), str):
                    img_frag = MessageContent(**{"type": "image_url", "image_url": {"url": record["image"]}})
            else:
                resolved = resolve_media_path(record.get("image"), root=content_root)
                if resolved:
                    img_frag = MessageContent(**{"type": "image_url", "image_url": {"url": resolved}})

        # STEP 3: Build messages with question and image.
        # Use ScreenSpot-Pro-GUI-Grounding `models/minicpmv.py:56` prompt (forces bbox format).
        instruction = (
            record.get("instruction")
            or record.get("prompt_to_evaluate")
            or record.get("command")
            or record.get("text")
            or "Locate the UI element described in the instruction"
        )
        question_text = (
            f'What is the bounding box of the UI element corresponding to the user instruction "{instruction}"? '
            "Output in the format of x1 x2 y1 y2."
        )
        content: List[MessageContent] = [MessageContent(**{"type": "text", "text": question_text})]
        if img_frag:
            content.append(img_frag)

        messages: List[Message] = []
        if system_prompt:
            messages.append(Message(**{"role": "system", "content": [MessageContent(**{"type": "text", "text": system_prompt.strip()})]}))
        messages.append(Message(**{"role": "user", "content": content}))

        # STEP 4: Generate sample ID
        sample_id = record.get("id") or str(hash(str(record.get("image", "")) + str(label_idx)))

        # STEP 5: Return Sample
        ret_sample = Sample(
            id=sample_id,
            schema_version=schema_version,
            options=[],  # No multiple choice options
            messages=messages,
            references=references,  # Contains the correct class name
            label=label,
            metadata=metadata
        )
        return ret_sample


__all__ = ["ScreenSpotProPreprocessor"]