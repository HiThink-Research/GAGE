"""Class-based MMMU multimodal preprocessor."""

from __future__ import annotations

from typing import Any, Dict

from gage_eval.assets.datasets.utils.multimodal import merge_multimodal_inputs
from gage_eval.assets.datasets.preprocessors.base import BasePreprocessor


class MMMUMultimodalPreprocessor(BasePreprocessor):
    """Ensure MMMU samples have multi_modal_data/image filled from messages."""

    def to_sample(self, record: Dict[str, Any], **kwargs: Any) -> Dict[str, Any]:
        sample = dict(record)
        merge_multimodal_inputs(sample)
        sample.setdefault("chat_template_mode", "preprocess")
        sample.setdefault("rendered_by", "preprocess")
        sample.setdefault("template_source", "manual")
        sample.setdefault("cache_suffix", "-converted")
        sample.setdefault("inputs", {})
        return sample


__all__ = ["MMMUMultimodalPreprocessor"]
