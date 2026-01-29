"""CharXiv bundle.

Ensures the HuggingFace `image` feature is available as `decoded_image`
so downstream preprocessors can treat it uniformly with other vision datasets.
"""

from __future__ import annotations

from typing import Any, Dict

from gage_eval.assets.datasets.bundles.base import BaseBundle
from loguru import logger


class CharXivBundle(BaseBundle):
    """Provide CharXiv-related dataset resources."""

    def __init__(self, **kwargs: Any):
        super().__init__(**kwargs)

    def load(self) -> None:
        """CharXiv does not require extra resources to be loaded."""
        return None

    def provide(self, sample: Dict[str, Any], **kwargs: Any) -> Dict[str, Any]:
        """Ensure `decoded_image` is present when HF streaming yields a PIL `image`."""
        sample_dict = dict(sample)

        if "decoded_image" not in sample_dict and "image" in sample_dict:
            image_value = sample_dict.get("image")
            if image_value is not None:
                try:
                    from PIL import Image  # type: ignore

                    if isinstance(image_value, Image.Image):
                        sample_dict["decoded_image"] = image_value
                except Exception as e:
                    logger.warning(f"CharXiv bundle: failed to decode image: {e}")

        return sample_dict


__all__ = ["CharXivBundle"]

