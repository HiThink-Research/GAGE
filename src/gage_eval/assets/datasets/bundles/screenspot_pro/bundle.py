"""ScreenSpot-Pro Resource Providers"""

from __future__ import annotations

from typing import Any, Dict

from gage_eval.assets.datasets.bundles.base import BaseBundle
from loguru import logger


class ScreenSpotProBundle(BaseBundle):
    """Provide ScreenSpot-Pro related resources"""

    def __init__(self, **kwargs: Any):
        super().__init__(**kwargs)

    def load(self) -> None:
        """Load any required resources (ScreenSpot-Pro doesn't need additional resources)."""
        pass

    def provide(self, sample: Dict[str, Any], **kwargs: Any) -> Any:
        """Provide resources for ScreenSpot-Pro sample (ensure decoded_image is available).

        HuggingFace datasets library automatically decodes Image features to PIL Image
        objects in streaming mode. The field name is 'image', not 'decoded_image'.
        We need to copy 'image' to 'decoded_image' for the preprocessor to use.
        """
        sample_dict = dict(sample)

        # HuggingFace datasets library automatically decodes Image features to PIL Image
        # in streaming mode. The decoded PIL Image is stored in the 'image' field.
        # We need to ensure 'decoded_image' exists for the preprocessor.
        if "decoded_image" not in sample_dict and "image" in sample_dict:
            image_value = sample_dict.get("image")
            if image_value is not None:
                try:
                    from PIL import Image
                    # In streaming mode, HuggingFace datasets library automatically decodes
                    # Image features to PIL Image objects
                    if isinstance(image_value, Image.Image):
                        # Direct PIL Image: copy to decoded_image
                        sample_dict["decoded_image"] = image_value
                    # Fallback: handle other formats (dict, bytes, etc.)
                    elif isinstance(image_value, dict):
                        # HuggingFace Image feature format: {"bytes": b"...", "path": "..."}
                        if "bytes" in image_value:
                            from io import BytesIO
                            sample_dict["decoded_image"] = Image.open(BytesIO(image_value["bytes"])).convert("RGB")
                        elif "path" in image_value:
                            sample_dict["decoded_image"] = Image.open(image_value["path"]).convert("RGB")
                    elif isinstance(image_value, bytes):
                        from io import BytesIO
                        sample_dict["decoded_image"] = Image.open(BytesIO(image_value)).convert("RGB")
                except Exception as e:
                    logger.warning(f"ScreenSpot-Pro bundle: Failed to decode image: {e}")
                    # Keep original image field for preprocessor fallback

        return sample_dict